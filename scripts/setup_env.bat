@echo off
REM EdgeSight Windows Environment Setup Script
REM One-click setup for development environment

title EdgeSight Setup
setlocal EnableDelayedExpansion

echo ================================================
echo EdgeSight - Windows Environment Setup
echo ================================================
echo ================================================
echo.

REM Check for Administrator privileges
net session >nul 2>&1
if %errorLevel% NEQ 0 (
    echo [ERROR] This script requires Administrator privileges.
    echo Please right-click the script or your terminal and select "Run as Administrator".
    pause
    exit /b 1
)

REM Check Python version
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    echo Please install Python 3.11 from https://python.org
    pause
    exit /b 1
)

for /f "tokens=*" %%a in ('python -c "import sys; print(sys.version_info.major, sys.version_info.minor)"') do set PYTHON_VERSION=%%a
for /f "tokens=1,2" %%a in ("%PYTHON_VERSION%") do (
    set MAJOR=%%a
    set MINOR=%%b
)

if %MAJOR% LSS 3 (
    echo [ERROR] Python 3.11+ required, found %MAJOR%.%MINOR%
    pause
    exit /b 1
)

if %MAJOR%==3 if %MINOR% LSS 11 (
    echo [WARNING] Python %MAJOR%.%MINOR% found. Python 3.11+ recommended.
)

echo [OK] Python %MAJOR%.%MINOR% detected

REM Check CMake
cmake --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] CMake is not installed or not in PATH
    echo Please install CMake 3.28+ from https://cmake.org
    pause
    exit /b 1
)
echo [OK] CMake detected

REM Check Visual Studio / MSVC
cl >nul 2>&1
if errorlevel 1 (
    where cl >nul 2>&1
    if errorlevel 1 (
        echo [WARNING] MSVC compiler not found in PATH
        echo Make sure to run this script from a Visual Studio Developer Command Prompt
        echo Or run: "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
    )
) else (
    echo [OK] MSVC compiler detected
)

echo.
echo ================================================
echo Phase 1: Installing Python Dependencies
echo ================================================
echo.

if not exist requirements.txt (
    echo [ERROR] requirements.txt not found
    pause
    exit /b 1
)

echo Installing Python packages...
pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Failed to install Python dependencies
    pause
    exit /b 1
)
echo [OK] Python dependencies installed

echo.
echo ================================================
echo Phase 2: Creating Directory Structure
echo ================================================
echo.

for %%d in (
    "data\raw\urfall"
    "data\raw\le2i"
    "data\processed"
    "model\checkpoints"
    "model\exported"
    "model\results"
    "inference\benchmark\results"
    "app\logs"
    "app\assets\icons"
    "build"
) do (
    if not exist %%d (
        mkdir %%d
        echo Created: %%d
    )
)

echo [OK] Directory structure created

echo.
echo ================================================
echo Phase 3: Building C++ Components
echo ================================================
echo.

if not exist build mkdir build
cd build

echo Configuring CMake...
cmake .. -DBUILD_TESTS=ON -DBUILD_APP=ON
if errorlevel 1 (
    echo [ERROR] CMake configuration failed
    cd ..
    pause
    exit /b 1
)

echo Building Release configuration...
cmake --build . --config Release
if errorlevel 1 (
    echo [ERROR] Build failed
    cd ..
    pause
    exit /b 1
)

cd ..
echo [OK] C++ components built successfully

echo.
echo ================================================
echo Phase 4: Running Tests
echo ================================================
echo.

echo Running Python syntax validation...
python -m py_compile data\download_datasets.py data\preprocess.py data\dataset.py
python -m py_compile model\architecture.py model\train.py model\evaluate.py
python -m py_compile model\export_onnx.py model\quantize.py
python -m py_compile inference\python_infer.py inference\benchmark\benchmark.py
if errorlevel 1 (
    echo [WARNING] Some Python files have syntax issues
) else (
    echo [OK] Python syntax validation passed
)

if exist build\bin\test_engine.exe (
    echo Running C++ unit tests...
    build\bin\test_engine.exe
    if errorlevel 1 (
        echo [WARNING] Some C++ tests failed ^(may need trained model^)
    ) else (
        echo [OK] C++ tests passed
    )
) else (
    echo [INFO] C++ test binary not found, skipping
)

echo.
echo ================================================
echo Setup Complete!
echo ================================================
echo.
echo Next steps:
echo   1. Download datasets:   python data\download_datasets.py
echo   2. Preprocess data:     python data\preprocess.py
echo   3. Train model:         python model\train.py --epochs 30
echo   4. Export to ONNX:      python model\export_onnx.py
echo   5. Run benchmarks:      python scripts\run_benchmark.py
echo   6. Start GUI app:       build\bin\EdgeSight.exe
echo.
echo For quick test with existing model:
echo   python inference\python_infer.py --model model\exported\fallnet_fp32.onnx
echo.
echo Documentation: README.md
echo.

pause
