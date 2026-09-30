@echo off
REM Usage: install_windows.bat "C:\path\to\ComfyUI" [--models-dir D:\AI\models] [--with-claude] [--no-lora]
REM Or drag your ComfyUI folder onto this file.
setlocal
set "SCRIPT=%~dp0install_music_video.py"
set "COMFY=%~1"
if "%COMFY%"=="" (
  echo Drag your ComfyUI folder onto this file, or run: install_windows.bat "C:\path\to\ComfyUI"
  pause
  exit /b 1
)
shift
set "EXTRA="
:collect
if "%~1"=="" goto run
set "EXTRA=%EXTRA% %1"
shift
goto collect
:run
set PY=
REM PY holds its own quotes so paths with spaces work
if exist "%COMFY%\..\python_embeded\python.exe" set PY="%COMFY%\..\python_embeded\python.exe"
if exist "%COMFY%\python_embeded\python.exe" set PY="%COMFY%\python_embeded\python.exe"
if exist "%COMFY%\.venv\Scripts\python.exe" set PY="%COMFY%\.venv\Scripts\python.exe"
if not defined PY (
  where py >nul 2>nul && set PY=py -3
)
if not defined PY set PY=python
%PY% "%SCRIPT%" --comfyui "%COMFY%" %EXTRA%
pause
