param(
    [int]$Port = 8087,
    [string]$Device = 'Vulkan1',
    [int]$ContextSize = 8192,
    [int]$ImageMinTokens = 1024,
    [int]$ImageMaxTokens = 1024,
    [switch]$Background
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$serverPath = Join-Path $projectRoot '.vlm-runtime\llama-b12069\llama-server.exe'
$modelPath = Join-Path $projectRoot 'checkpoints\vlm\Qwen3-VL-4B-Instruct-Q4_K_M.gguf'
$projectorPath = Join-Path $projectRoot 'checkpoints\vlm\mmproj-F16.gguf'
foreach ($requiredPath in @($serverPath, $modelPath, $projectorPath)) {
    if (-not (Test-Path -LiteralPath $requiredPath -PathType Leaf)) {
        throw "Missing local runtime/model file: $requiredPath. See docs/VLM_RUNTIME.md."
    }
}
if ($Port -lt 1024 -or $Port -gt 65535) { throw 'Port must be between 1024 and 65535.' }
if ($ContextSize -lt 2048 -or $ImageMinTokens -lt 64 -or $ImageMaxTokens -lt $ImageMinTokens) { throw 'Require ContextSize >=2048 and 64 <= ImageMinTokens <= ImageMaxTokens.' }
$portProbe = New-Object System.Net.Sockets.TcpClient
$portBusy = $false
try {
    $portProbe.Connect('127.0.0.1', $Port)
    $portBusy = $true
} catch [System.Net.Sockets.SocketException] {
    # Connection refused is expected when no existing service occupies the port.
} finally {
    $portProbe.Dispose()
}
if ($portBusy) { throw "Port $Port is already in use. Stop the previous server before restarting; active_server.json was not changed." }

# Inspect --list-devices on another computer: Vulkan1 is this machine's RTX 3060.
# Bind only to loopback. Images never leave this computer through this server.
$serverArgs = @(
    '-m', $modelPath, '--mmproj', $projectorPath,
    '--alias', 'ion-qwen3-vl-4b', '--host', '127.0.0.1', '--port', "$Port",
    '--device', $Device, '-ngl', '99', '-c', "$ContextSize", '-np', '1',
    '--image-min-tokens', "$ImageMinTokens", '--image-max-tokens', "$ImageMaxTokens", '--flash-attn', 'on',
    '-ctk', 'q8_0', '-ctv', 'q8_0', '--jinja'
)
$versionProcess = New-Object System.Diagnostics.Process
$versionProcess.StartInfo.FileName = $serverPath
$versionProcess.StartInfo.Arguments = '--version'
$versionProcess.StartInfo.UseShellExecute = $false
$versionProcess.StartInfo.CreateNoWindow = $true
$versionProcess.StartInfo.RedirectStandardOutput = $true
$versionProcess.StartInfo.RedirectStandardError = $true
[void]$versionProcess.Start()
$runtimeVersion = ($versionProcess.StandardOutput.ReadToEnd() + $versionProcess.StandardError.ReadToEnd()).Trim()
$versionProcess.WaitForExit()
if ($versionProcess.ExitCode -ne 0) { throw 'Could not query llama-server version.' }
$versionProcess.Dispose()
# Hash actual local weights so a substituted file cannot reuse stale metadata.
# No PID or timestamp: identical launches must produce identical metadata hashes.
$runtimeMetadata = [ordered]@{
    schema_version = 1
    model_alias = 'ion-qwen3-vl-4b'
    model_sha256 = (Get-FileHash -LiteralPath $modelPath -Algorithm SHA256).Hash.ToLowerInvariant()
    mmproj_sha256 = (Get-FileHash -LiteralPath $projectorPath -Algorithm SHA256).Hash.ToLowerInvariant()
    llama_cpp_version = $runtimeVersion
    runtime_backend = 'Vulkan'
    device = $Device
    context_size = $ContextSize
    image_min_tokens = $ImageMinTokens
    image_max_tokens = $ImageMaxTokens
    parallel_slots = 1
    kv_cache_type_k = 'q8_0'
    kv_cache_type_v = 'q8_0'
    endpoint = "http://127.0.0.1:$Port/v1"
    launch_args = $serverArgs
}
$metadataPath = Join-Path $projectRoot '.vlm-runtime\active_server.json'
$metadataTempPath = $metadataPath + '.tmp'
[System.IO.File]::WriteAllText($metadataTempPath, (($runtimeMetadata | ConvertTo-Json -Depth 8) + [Environment]::NewLine), (New-Object System.Text.UTF8Encoding($false)))
Move-Item -LiteralPath $metadataTempPath -Destination $metadataPath -Force
if ($Background) {
    $logRoot = Join-Path $projectRoot '.vlm-runtime'
    $stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
    $stdoutPath = Join-Path $logRoot "server_$stamp.stdout.log"
    $stderrPath = Join-Path $logRoot "server_$stamp.stderr.log"
    # Start-Process joins ArgumentList; quote path arguments containing spaces.
    $quotedArgs = $serverArgs | ForEach-Object { '"' + $_.Replace('"', '\"') + '"' }
    $serverProcess = Start-Process -FilePath $serverPath -ArgumentList $quotedArgs -WindowStyle Hidden -PassThru -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath -WorkingDirectory $projectRoot
    Write-Output "PID: $($serverProcess.Id)"
    Write-Output "Endpoint: http://127.0.0.1:$Port/v1"
    Write-Output "Model: ion-qwen3-vl-4b"
    Write-Output "Log: $stderrPath"
    Write-Output 'Wait for /health to return status ok before inference.'
} else {
    & $serverPath @serverArgs
    if ($LASTEXITCODE -ne 0) { throw "llama-server exited with code $LASTEXITCODE" }
}
