@echo off
REM ============================================================
REM  VisionLab-D455
REM  Stereo Ranging and Object Detection Experiment Platform
REM  One-click launcher for Windows (double-click friendly)
REM
REM  This file is kept strictly ASCII-only with CRLF line endings.
REM  Reason: cmd.exe reads batch files using the console code page and
REM  re-reads them by byte offset, so non-ASCII content or LF-only line
REM  endings corrupt the parse. The Chinese UI lives in start.ps1.
REM
REM  Interpreter selection never hard-codes a path and never trusts a
REM  folder name: every candidate is probed for the required packages,
REM  and only an interpreter that actually passes is accepted. This
REM  avoids picking a sibling project venv that lacks Flask.
REM ============================================================
setlocal EnableExtensions
cd /d "%~dp0"
title VisionLab-D455

set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"

REM ---------- 1) gather candidates in priority order ----------
set "PY="
set "ANY="
set "PROBE_OK="

call :probe "%VISIONLAB_PYTHON%"
call :probe "%~dp0.venv\Scripts\python.exe"
call :probe "%~dp0venv\Scripts\python.exe"
for /d %%D in ("%~dp0..\*") do call :probe "%%D\.venv\Scripts\python.exe"
for /d %%D in ("%~dp0..\*") do call :probe "%%D\venv\Scripts\python.exe"
for /f "delims=" %%I in ('where python 2^>nul') do call :probe "%%I"
for /f "delims=" %%I in ('where py 2^>nul') do call :probe "%%I"

REM  nothing passed the probe: fall back to any existing interpreter
if not defined PY if defined ANY set "PY=%ANY%"
if not defined PY goto :nopython

REM ---------- 2) banner ----------
echo ============================================================
echo   VisionLab-D455
echo   Stereo Ranging and Object Detection Experiment Platform
echo ============================================================
echo   Python   : %PY%
echo   Workdir  : %~dp0
echo   Login    : admin / admin123
echo ============================================================
echo.
if not "%PROBE_OK%"=="1" (
    echo   [WARN] This interpreter is missing core dependencies.
    echo          Install them with:
    echo            "%PY%" -m pip install -r requirements.txt
    echo          Or point VISIONLAB_PYTHON at a prepared python.exe.
    echo.
)
echo   Tips:
echo     start.bat --no-camera      use the built-in synthetic scene
echo     start.bat --port 8080      change the listening port
echo     start.bat --no-browser     do not open a browser
echo     start.bat --host 0.0.0.0   listen on all interfaces
echo.
echo   Starting. Press Ctrl+C to stop.
echo.

REM ---------- 3) launch ----------
"%PY%" app.py %*
set "EXITCODE=%ERRORLEVEL%"

if not "%EXITCODE%"=="0" (
    echo.
    echo [ERROR] Platform exited with code %EXITCODE%
    echo Running the environment self-check to help locate the problem ...
    echo.
    "%PY%" tools\self_check.py
    echo.
    echo Common causes: the port is already in use ^(use --port^),
    echo missing dependencies, or an unwritable storage directory.
)

echo.
pause
exit /b %EXITCODE%


REM ============================================================
REM  :probe <path>
REM  Accept the path if it exists and can import the core packages.
REM  Remembers the first existing interpreter as a last-resort fallback.
REM ============================================================
:probe
if defined PY exit /b
if "%~1"=="" exit /b
if not exist "%~1" exit /b
if not defined ANY set "ANY=%~1"
set "CAND=%~1"
set "HIT="
REM  Do not use "for /f ('...')" here: cmd strips the outer quotes of a
REM  command whose first token is quoted, so the probe would silently
REM  fail and the launcher would fall back to a broken interpreter.
REM  A plain redirect plus "set /p" is reliable.
"%CAND%" -c "import flask,numpy,cv2,yaml,sqlite3,PIL;print(1)" >"%TEMP%\vl_probe.txt" 2>nul
if exist "%TEMP%\vl_probe.txt" set /p HIT=<"%TEMP%\vl_probe.txt"
del "%TEMP%\vl_probe.txt" >nul 2>nul
if not "%HIT%"=="1" exit /b
set "PY=%CAND%"
set "PROBE_OK=1"
exit /b


:nopython
echo.
echo [ERROR] No Python interpreter was found.
echo.
echo   Install Python 3.10+ and make sure it is on PATH, or set the
echo   environment variable VISIONLAB_PYTHON to the full path of
echo   python.exe, for example:
echo.
echo     set VISIONLAB_PYTHON=C:\Python312\python.exe
echo     start.bat
echo.
pause
exit /b 1
