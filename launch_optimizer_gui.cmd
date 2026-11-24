@echo off
setlocal
pushd "%~dp0" || exit /b 1
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" run_gui.py
) else (
    py -3 run_gui.py
)
set "OPTIMIZER_EXIT=%ERRORLEVEL%"
popd
if not "%OPTIMIZER_EXIT%"=="0" pause
exit /b %OPTIMIZER_EXIT%
