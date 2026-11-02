@echo off
setlocal
title AI Based Deforestation Monitoring System - Command Center
color 0A

REM Ensure script always executes in its own directory
cd /d "%~dp0"

REM Set optimal environment variables
set "MPLCONFIGDIR=%TEMP%\matplotlib"
if not exist "%MPLCONFIGDIR%" mkdir "%MPLCONFIGDIR%" >nul 2>&1
set "PYTHONDONTWRITEBYTECODE=1"
set "PYTHONIOENCODING=utf-8"
set "STREAMLIT_BROWSER_GATHER_USAGE_STATS=false"

echo ==============================================================================
echo        AI DEFORESTATION MONITORING ^& EARTH OBSERVATION SYSTEM
echo        Multi-Temporal Satellite Change Detection ^& Deep Learning AI
echo ==============================================================================
echo.

REM 0. Check Virtual Environment
if exist ".venv\Scripts\activate.bat" (
    echo [ENV] Activating local virtual environment .venv
    call ".venv\Scripts\activate.bat"
) else if exist "venv\Scripts\activate.bat" (
    echo [ENV] Activating local virtual environment venv
    call "venv\Scripts\activate.bat"
)

REM 1. Check Python installation
python --version >nul 2>&1
if %errorlevel% neq 0 (
    color 0C
    echo [ERROR] Python was not found on your system PATH.
    echo Please install Python 3.10+ and ensure 'Add Python to PATH' is selected.
    echo Download: https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
)

for /f "tokens=*" %%i in ('python --version') do set "PYTHON_VER=%%i"
echo [1/4] Environment: %PYTHON_VER% verified.
echo.

REM 2. Check Core Dependencies
echo [2/4] Verifying required Python libraries...
python -c "import streamlit, torch, cv2, pandas, sklearn, PIL, matplotlib, plotly, folium, reportlab" >nul 2>&1
if %errorlevel% neq 0 (
    echo [WARNING] Missing dependencies detected. Installing from requirements.txt...
    pip install -r requirements.txt
    if %errorlevel% neq 0 (
        color 0C
        echo [ERROR] Dependency installation failed. Run 'pip install -r requirements.txt' manually.
        pause
        exit /b 1
    )
) else (
    echo       All core deep-learning and remote sensing packages verified.
)
echo.

REM 3. Verify Model Checkpoints Status
echo [3/4] Verifying AI Model Weights ^& Checkpoints...
set "REAL_MODELS=0"

if exist "models\checkpoints\unet_deforestation.pth" (
    set "REAL_MODELS=1"
    echo       [OK] ForestUNet Checkpoint:    models\checkpoints\unet_deforestation.pth
) else (
    echo       [NOTICE] ForestUNet Checkpoint not found.
)

if exist "weights\siamese_best.pt" (
    echo       [OK] Siamese ChangeNet:        weights\siamese_best.pt
)

if exist "weights\random_forest.joblib" (
    echo       [OK] Random Forest Model:      weights\random_forest.joblib
)

if "%REAL_MODELS%"=="1" (
    echo       -- Active Status: REAL MODEL INFERENCE [PyTorch 4-stage U-Net]
) else (
    echo       -- Active Status: DEMO MODE [Vegetation Index Proxy active]
    echo          To train the model: python train.py --epochs 6
)
echo.

REM 4. Start Streamlit Application
echo [4/4] Launching Interactive Glassmorphic Web Dashboard...
echo ------------------------------------------------------------------------------
echo Dashboard URL:      http://localhost:8501
echo Auto-opening browser in 2 seconds...
echo.
echo Quick Utilities:
echo   - Run Full Test Suite:         pytest tests\test_audit.py
echo   - Run System Pipeline Audit:   python test_system.py
echo   - Train Neural Checkpoint:     python train.py --epochs 6
echo   - Evaluate Benchmarks:         python models\run_benchmark.py
echo.
echo Press Ctrl+C in this terminal to safely stop the server.
echo ------------------------------------------------------------------------------
echo.

REM Launch background browser opener
start "" powershell -WindowStyle Hidden -Command "Start-Sleep -Seconds 2; if (Test-Path 'C:\Program Files\Google\Chrome\Application\chrome.exe') { Start-Process 'C:\Program Files\Google\Chrome\Application\chrome.exe' 'http://localhost:8501' } elseif (Test-Path 'C:\Program Files (x86)\Google\Chrome\Application\chrome.exe') { Start-Process 'C:\Program Files (x86)\Google\Chrome\Application\chrome.exe' 'http://localhost:8501' } else { Start-Process 'http://localhost:8501' }"

REM Start Streamlit server
python -m streamlit run app.py --server.port 8501 --server.headless false --browser.gatherUsageStats false

pause
