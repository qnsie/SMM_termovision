@echo off
setlocal
set "VENV_PYW=%~dp0.venv\Scripts\pythonw.exe"
set "VENV_PY=%~dp0.venv\Scripts\python.exe"

if exist "%VENV_PYW%" (
    start "" "%VENV_PYW%" "%~dp0app.py"
    exit /b 0
)

if exist "%VENV_PY%" (
    "%VENV_PY%" "%~dp0app.py"
    exit /b %ERRORLEVEL%
)

echo Nie znaleziono srodowiska .venv. Uruchom:
echo   python -m venv .venv
echo   .\.venv\Scripts\pip install -r requirements.txt
pause
exit /b 1
