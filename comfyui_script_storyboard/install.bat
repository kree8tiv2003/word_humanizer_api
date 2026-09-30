@echo off
rem Double-click to install the extra Python packages this node pack uses.
rem Works with ComfyUI Portable and the ComfyUI Desktop app on Windows.
setlocal
cd /d "%~dp0"
set "PY="
if exist "..\..\..\python_embeded\python.exe" set "PY=..\..\..\python_embeded\python.exe"
if not defined PY if exist "..\..\.venv\Scripts\python.exe" set "PY=..\..\.venv\Scripts\python.exe"
if not defined PY set "PY=python"
echo Installing with: %PY%
"%PY%" -m pip install -r requirements.txt
echo.
echo Done. Restart ComfyUI.
pause
