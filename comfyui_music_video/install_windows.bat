@echo off
REM Drag ANY of these onto this file:
REM   - your ComfyUI folder (has custom_nodes inside)
REM   - the portable folder (ComfyUI_windows_portable)
REM   - a shared folder that only has input / models / output
REM Or just double-click it: the installer will search for ComfyUI itself.
REM Extra options: --models-dir D:\AI\models   --with-claude   --no-lora   --dry-run
setlocal
set "SCRIPT=%~dp0install_music_video.py"
set "TARGET=%~1"
set "EXTRA="
if not "%~1"=="" shift
:collect
if "%~1"=="" goto findpy
set "EXTRA=%EXTRA% %1"
shift
goto collect

:findpy
REM PY keeps its own quotes so paths with spaces work.
set PY=
if not "%TARGET%"=="" (
  if exist "%TARGET%\python_embeded\python.exe" set PY="%TARGET%\python_embeded\python.exe"
  if exist "%TARGET%\..\python_embeded\python.exe" set PY="%TARGET%\..\python_embeded\python.exe"
  if exist "%TARGET%\.venv\Scripts\python.exe" set PY="%TARGET%\.venv\Scripts\python.exe"
  if exist "%TARGET%\..\.venv\Scripts\python.exe" set PY="%TARGET%\..\.venv\Scripts\python.exe"
)
if not defined PY if exist "%USERPROFILE%\Documents\ComfyUI\.venv\Scripts\python.exe" set PY="%USERPROFILE%\Documents\ComfyUI\.venv\Scripts\python.exe"
if not defined PY (
  py -3 --version >nul 2>nul && set PY=py -3
)
if not defined PY (
  python --version >nul 2>nul && set PY=python
)
if not defined PY (
  echo Looking for the Python that came with ComfyUI...
  for /f "usebackq delims=" %%P in (`powershell -NoProfile -Command "Get-ChildItem -Path $env:USERPROFILE -Filter python.exe -Recurse -Depth 6 -ErrorAction SilentlyContinue ^| Where-Object { $_.FullName -match 'comfy' } ^| Select-Object -First 1 -ExpandProperty FullName"`) do set PY="%%P"
)
if not defined PY (
  echo.
  echo Could not find Python. Install it from https://www.python.org/downloads/
  echo ^(tick "Add python.exe to PATH"^), then run this file again.
  pause
  exit /b 1
)
echo Using Python: %PY%
if "%TARGET%"=="" (
  %PY% "%SCRIPT%" %EXTRA%
) else (
  %PY% "%SCRIPT%" --comfyui "%TARGET%" %EXTRA%
)
pause
