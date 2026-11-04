@echo off
cd /d "%~dp0"
where python >nul 2>nul
if errorlevel 1 (
    echo Python was not found. Install Python with numpy and matplotlib.
    pause
    exit /b 1
)
python "%~dp0view_ion.py" %*
if errorlevel 1 pause
