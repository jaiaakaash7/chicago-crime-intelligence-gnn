@echo off
setlocal enabledelayedexpansion

echo ============================================================
echo   CHICAGO CRIME INTELLIGENCE SYSTEM
echo   GraphSAGE Predictive Intelligence & Network Analytics
echo ============================================================
echo.

cd /d "%~dp0"

:: Activate virtual environment if available
if exist ".venv\Scripts\activate.bat" (
    echo [INFO] Activating virtual environment (.venv)...
    call .venv\Scripts\activate.bat
) else (
    echo [INFO] Using system Python environment...
)

echo.
echo [INFO] Starting Local Gradio Dashboard on http://127.0.0.1:7860 ...
echo [INFO] Mode: STRICTLY LOCAL (share=False, no cloud deployment)
echo.
echo Press Ctrl+C in this window to stop the server when finished.
echo ============================================================
echo.

python dashboard\app.py

pause
