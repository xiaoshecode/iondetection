<#
.SYNOPSIS
Run the ion-recognition pipeline from any working directory.
.DESCRIPTION
classical performs inference only. masked and diffusion validate the explicit
train/validation splits, train a new model, then infer using training/best.pt.
No dependencies are installed and no global settings are changed. An existing
nonempty output directory is refused. All relative paths are project-relative.
.EXAMPLE
.\scripts\run_pipeline.ps1 -Method classical
.EXAMPLE
.\scripts\run_pipeline.ps1 -Method masked -Device cpu
.EXAMPLE
.\scripts\run_pipeline.ps1 -Method diffusion -ConfirmStaticRepeats
#>
[CmdletBinding()]
param(
    [ValidateSet('classical', 'masked', 'diffusion')]
    [string]$Method = 'classical',
    [string]$Python = '.venv\Scripts\python.exe',
    [string]$TrainRoot = 'data/train',
    [string]$ValRoot = 'data/val',
    [string]$InputPath = 'data/test',
    [string]$Config = 'configs/default.yaml',
    [string]$Device = 'auto',
    [string]$OutputRoot,
    [switch]$ConfirmStaticRepeats
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot

function Get-ProjectPath {
    param([Parameter(Mandatory = $true)][string]$Value)
    if ([System.IO.Path]::IsPathRooted($Value)) {
        return [System.IO.Path]::GetFullPath($Value)
    }
    return [System.IO.Path]::GetFullPath((Join-Path $projectRoot $Value))
}

function Invoke-IonCommand {
    param(
        [Parameter(Mandatory = $true)][string]$Stage,
        [Parameter(Mandatory = $true)][string[]]$Arguments
    )
    Write-Host "`n[$Stage]"
    # Argument splatting preserves spaces without constructing shell commands.
    & $pythonExecutable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Stage '$Stage' failed (Python exit code $LASTEXITCODE). Later stages were not run."
    }
}

$locationChanged = $false
try {
    if ($Method -eq 'diffusion' -and -not $ConfirmStaticRepeats) {
        throw 'Diffusion requires static repeated exposures of each scene. After verifying fixed positions, bright/dark states, exposure and gain within every group, add -ConfirmStaticRepeats. For unrelated images choose -Method masked.'
    }

    $pythonCandidate = Get-ProjectPath $Python
    if (Test-Path -LiteralPath $pythonCandidate -PathType Leaf) {
        $pythonExecutable = $pythonCandidate
    }
    elseif ($Python -notmatch '[\\/]') {
        $pythonExecutable = (Get-Command -Name $Python -CommandType Application -ErrorAction Stop).Source
    }
    else {
        throw "Python was not found at '$pythonCandidate'. Follow README installation steps or pass -Python with an existing Python executable."
    }

    $configPath = Get-ProjectPath $Config
    $inputSource = Get-ProjectPath $InputPath
    if (-not (Test-Path -LiteralPath $configPath -PathType Leaf)) {
        throw "Config file does not exist: $configPath"
    }
    if (-not (Test-Path -LiteralPath $inputSource)) {
        throw "Inference input does not exist: $inputSource. Put grayscale images into data/test, or specify -InputPath."
    }

    $datasetSources = @($inputSource)
    if ($Method -ne 'classical') {
        $trainSource = Get-ProjectPath $TrainRoot
        $validationSource = Get-ProjectPath $ValRoot
        foreach ($source in @($trainSource, $validationSource)) {
            if (-not (Test-Path -LiteralPath $source -PathType Container)) {
                throw "Dataset directory does not exist: $source"
            }
        }
        $datasetSources += @($trainSource, $validationSource)
    }

    if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
        $stamp = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
        $OutputRoot = "runs/pipeline_${Method}_$stamp"
    }
    $destination = Get-ProjectPath $OutputRoot
    foreach ($source in $datasetSources) {
        if (Test-Path -LiteralPath $source -PathType Container) {
            $sourcePrefix = $source.TrimEnd([System.IO.Path]::DirectorySeparatorChar, [System.IO.Path]::AltDirectorySeparatorChar) + [System.IO.Path]::DirectorySeparatorChar
            if ($destination.Equals($source, [System.StringComparison]::OrdinalIgnoreCase) -or $destination.StartsWith($sourcePrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
                throw "Output must be outside every input dataset directory: $destination"
            }
        }
    }
    if (Test-Path -LiteralPath $destination) {
        if (-not (Test-Path -LiteralPath $destination -PathType Container)) {
            throw "Output path is a file: $destination"
        }
        if (@(Get-ChildItem -LiteralPath $destination -Force).Count -gt 0) {
            throw "Output directory is not empty: $destination. Choose a new -OutputRoot or omit it for a timestamped run."
        }
    }

    Push-Location -LiteralPath $projectRoot
    $locationChanged = $true
    New-Item -ItemType Directory -Path $destination -Force | Out-Null
    Write-Host "Project: $projectRoot"
    Write-Host "Method:  $Method"
    Write-Host "Python:  $pythonExecutable"
    Write-Host "Output:  $destination"

    $inferArguments = @('-m', 'ionrecog', 'infer', '--method', $Method,
        '--input', $inputSource, '--output', (Join-Path $destination 'predictions'),
        '--config', $configPath, '--device', $Device)

    if ($Method -ne 'classical') {
        Invoke-IonCommand -Stage 'Inspect datasets' -Arguments @('-m', 'ionrecog', 'inspect',
            '--method', $Method, '--train', $trainSource, '--val', $validationSource,
            '--output', (Join-Path $destination 'dataset_report.json'))

        $trainingOutput = Join-Path $destination 'training'
        $trainingArguments = @('-m', 'ionrecog', 'train', '--method', $Method,
            '--train', $trainSource, '--val', $validationSource, '--output', $trainingOutput,
            '--config', $configPath, '--device', $Device)
        if ($Method -eq 'diffusion') {
            $trainingArguments += '--confirm-static'
        }
        Invoke-IonCommand -Stage 'Train model' -Arguments $trainingArguments

        $checkpoint = Join-Path $trainingOutput 'best.pt'
        if (-not (Test-Path -LiteralPath $checkpoint -PathType Leaf)) {
            throw "Training returned successfully but its best checkpoint is missing: $checkpoint"
        }
        $inferArguments += @('--checkpoint', $checkpoint)
    }

    Invoke-IonCommand -Stage 'Infer ion locations' -Arguments $inferArguments
    Write-Host "`nCompleted. Results: $(Join-Path $destination 'predictions')"
    Write-Host 'Candidate counts and training losses are not measured experimental detection accuracy.'
}
catch {
    Write-Error -Message $_.Exception.Message -ErrorAction Continue
    exit 1
}
finally {
    if ($locationChanged) {
        Pop-Location
    }
}
