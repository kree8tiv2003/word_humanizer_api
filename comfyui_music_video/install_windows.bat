@echo off
REM Drag ANY of these onto this file:
REM   - your ComfyUI folder (has custom_nodes inside)
REM   - the portable folder (ComfyUI_windows_portable)
REM   - a shared folder that only has input / models / output
REM Or just double-click it: the installer will search for ComfyUI itself.
REM Extra options: --models-dir D:\AI\models   --with-claude   --no-lora   --dry-run
setlocal
set "HERE=%~dp0"
set "SCRIPT=%HERE%install_music_video.py"
set "TARGET=%~1"
set "EXTRA="
if not "%~1"=="" shift
:collect
if "%~1"=="" goto findpy
set "EXTRA=%EXTRA% %1"
shift
goto collect

:findpy
REM Every candidate is test-run first, so a broken Python (e.g. a ComfyUI .venv whose
REM base Python was deleted) is skipped instead of stopping the installer.
set PY=
call :try "%HERE%python\python.exe"
if not "%TARGET%"=="" (
  call :try "%TARGET%\python_embeded\python.exe"
  call :try "%TARGET%\..\python_embeded\python.exe"
  call :try "%TARGET%\.venv\Scripts\python.exe"
  call :try "%TARGET%\..\.venv\Scripts\python.exe"
)
call :try "%USERPROFILE%\Documents\ComfyUI\.venv\Scripts\python.exe"
if not defined PY (
  py -3 -c "import sys" >nul 2>nul && set PY=py -3
)
if not defined PY (
  python -c "import sys" >nul 2>nul && set PY=python
)
if not defined PY call :download_python
if not defined PY (
  echo.
  echo Could not get Python automatically.
  echo Please install Python from https://www.python.org/downloads/
  echo ^(on the first installer screen tick "Add python.exe to PATH"^), then run this file again.
  pause
  exit /b 1
)
echo Using Python: %PY%
echo.
if "%TARGET%"=="" (
  %PY% "%SCRIPT%" %EXTRA%
) else (
  %PY% "%SCRIPT%" --comfyui "%TARGET%" %EXTRA%
)
pause
exit /b 0

:try
if defined PY exit /b 0
if not exist "%~1" exit /b 0
"%~1" -c "import sys, ssl" >nul 2>nul
if errorlevel 1 (
  echo Skipping a Python that doesn't work: %~1
  exit /b 0
)
set PY="%~1"
exit /b 0

:download_python
echo.
echo No working Python found - downloading a small private copy (about 11 MB)...
echo It stays inside this folder and does not change anything else on your PC.
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ProgressPreference='SilentlyContinue'; [Net.ServicePointManager]::SecurityProtocol=[Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -UseBasicParsing -Uri 'https://www.python.org/ftp/python/3.12.10/python-3.12.10-embed-amd64.zip' -OutFile '%HERE%python.zip'; Expand-Archive -Force -Path '%HERE%python.zip' -DestinationPath '%HERE%python'; Remove-Item '%HERE%python.zip'"
call :try "%HERE%python\python.exe"
exit /b 0
