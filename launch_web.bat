@echo off
REM HybriDock-Pep - start the web app on Windows (it runs inside WSL2, which install.bat set up).
REM Double-click this file. Your browser opens at http://127.0.0.1:8000 ; close this window to stop the app.
setlocal enabledelayedexpansion

where wsl >nul 2>nul
if errorlevel 1 (
    echo WSL2 is not installed. Run install.bat first - it sets everything up.
    pause
    exit /b 1
)

REM %~dp0 ends in a backslash that would escape the closing quote on its way to WSL: strip it first.
set "REPO_DIR=%~dp0"
if "%REPO_DIR:~-1%"=="\" set "REPO_DIR=%REPO_DIR:~0,-1%"
for /f "usebackq delims=" %%p in (`wsl wslpath -a "%REPO_DIR%"`) do set "WSL_REPO_DIR=%%p"
if "%WSL_REPO_DIR%"=="" (
    echo Could not translate this folder into WSL. Run this file from inside the cloned repo folder.
    pause
    exit /b 1
)

REM open the browser a few seconds from now, once the app is listening (WSL2 forwards localhost to Windows)
start "" cmd /c "timeout /t 6 >nul & start http://127.0.0.1:8000"
wsl bash -lc "cd '%WSL_REPO_DIR%' && ./launch_web.sh --no-browser"
echo.
echo HybriDock-Pep has stopped.
pause
