@echo off
REM HybriDock-Pep - Windows launcher.
REM
REM HybriDock-Pep does not have a native Windows install path: RAPiDock's CUDA
REM stack and conda are only supported on Linux/WSL2 and macOS (see
REM the platform table in INSTALL.md). This script sets up WSL2 for you if you
REM don't have it, then runs the real installer (install.sh) inside WSL2 -
REM which gives you full CUDA passthrough to your NVIDIA GPU, exactly like a
REM native Linux install.
REM
REM Just double-click this file, or run it from a terminal:
REM     install.bat

setlocal enabledelayedexpansion

echo ============================================================
echo   HybriDock-Pep - Windows setup (via WSL2)
echo ============================================================
echo.

where wsl >nul 2>nul
if errorlevel 1 (
    echo WSL is not available on this machine. HybriDock-Pep needs WSL2 to run
    echo on Windows. Install it from an elevated PowerShell/CMD with:
    echo.
    echo     wsl --install
    echo.
    echo Then reboot and re-run this script.
    pause
    exit /b 1
)

wsl -l -v >nul 2>nul
if errorlevel 1 (
    echo WSL is installed but no Linux distro is set up yet. Installing Ubuntu...
    wsl --install -d Ubuntu
    echo.
    echo A reboot is likely required. After rebooting, open the new Ubuntu
    echo terminal once to finish its first-time setup ^(create a username/password^),
    echo then re-run this script.
    pause
    exit /b 0
)

echo WSL2 found. Handing off to install.sh inside WSL2...
echo.

REM Translate this script's own directory (the repo root) to a WSL path and
REM run install.sh there, so it works whether the repo was cloned on the
REM Windows side (C:\...) or already inside the WSL filesystem.
REM %~dp0 always ends in a trailing backslash; passed straight through as
REM "...\" the backslash escapes the closing quote before it reaches WSL's
REM argv, so `wsl wslpath -a` gets a mangled/unterminated argument and fails
REM (confirmed: errorlevel 2, no output) - leaving WSL_REPO_DIR empty and
REM silently handing off install.sh to the wrong directory. Strip it first.
set "REPO_DIR=%~dp0"
if "%REPO_DIR:~-1%"=="\" set "REPO_DIR=%REPO_DIR:~0,-1%"
for /f "usebackq delims=" %%p in (`wsl wslpath -a "%REPO_DIR%"`) do set "WSL_REPO_DIR=%%p"

if "%WSL_REPO_DIR%"=="" (
    echo Failed to translate the repo path into WSL. Make sure this script is
    echo run from inside the cloned repo folder and try again.
    pause
    exit /b 1
)

wsl bash -lc "cd '%WSL_REPO_DIR%' && bash install.sh"

echo.
echo Done. Future runs: open an Ubuntu/WSL terminal and run:
echo     cd '%WSL_REPO_DIR%' ^&^& ./launch_ui.sh
pause
