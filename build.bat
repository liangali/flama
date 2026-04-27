@echo off
REM Flama Build Script for Windows
REM This script automates the CMake configuration and build process

setlocal enabledelayedexpansion

echo ===============================================
echo Flama Build Script
echo ===============================================
echo.

REM ==========================================
REM Define dependency paths (modify as needed)
REM ==========================================
if not defined OPENVINO_ROOT (
    for /d %%D in ("D:\library\openvino\openvino_toolkit_windows_2026.1*") do set "OPENVINO_ROOT=%%~fD"
)
if not defined OPENVINO_GENAI_ROOT (
    for /d %%D in ("D:\library\openvino.genai\openvino_genai_windows_2026.1*") do set "OPENVINO_GENAI_ROOT=%%~fD"
)

if defined OPENVINO_ROOT (
    set "OPENVINO_DIR=%OPENVINO_ROOT%\runtime\cmake"
    set "TBB_DIR=%OPENVINO_ROOT%\runtime\3rdparty\tbb\lib\cmake\TBB"
) else (
    set "OPENVINO_DIR="
    set "TBB_DIR="
)

if defined OPENVINO_GENAI_ROOT (
    set "OPENVINO_GENAI_DIR=%OPENVINO_GENAI_ROOT%\runtime\cmake"
) else (
    set "OPENVINO_GENAI_DIR="
)

set "VPL_DIR=%cd%\thirdparty\_vplinstall\lib\cmake\vpl"

echo ==========================================
echo Configuration Paths
echo ==========================================
echo OpenVINO_ROOT:      %OPENVINO_ROOT%
echo OpenVINO_GenAI_ROOT:%OPENVINO_GENAI_ROOT%
echo OpenVINO_DIR:       %OPENVINO_DIR%
echo OpenVINOGenAI_DIR:  %OPENVINO_GENAI_DIR%
echo TBB_DIR:            %TBB_DIR%
echo VPL_DIR:            %VPL_DIR%
echo.

REM Check if vcpkg is installed
if not defined VCPKG_ROOT (
    echo ERROR: VCPKG_ROOT environment variable is not set
    echo Please set it to your vcpkg installation directory
    echo Example: setx VCPKG_ROOT "C:\vcpkg"
    pause
    exit /b 1
)

echo VCPKG_ROOT: %VCPKG_ROOT%
echo.

REM ==========================================
REM Validate dependency paths
REM ==========================================
echo ==========================================
echo Validating Dependency Paths...
echo ==========================================

set "PATHS_VALID=1"

if not exist "%OPENVINO_DIR%" (
    echo ERROR: OpenVINO_DIR not found: %OPENVINO_DIR%
    set "PATHS_VALID=0"
)

if not exist "%OPENVINO_GENAI_DIR%" (
    echo ERROR: OpenVINOGenAI_DIR not found: %OPENVINO_GENAI_DIR%
    set "PATHS_VALID=0"
)

if not exist "%TBB_DIR%" (
    echo ERROR: TBB_DIR not found: %TBB_DIR%
    set "PATHS_VALID=0"
)

if not exist "%VPL_DIR%" (
    echo ERROR: VPL_DIR not found: %VPL_DIR%
    set "PATHS_VALID=0"
)

if %PATHS_VALID% equ 0 (
    echo.
    echo ERROR: One or more dependency paths do not exist!
    echo Please modify the path definitions at the beginning of this script.
    echo.
    pause
    exit /b 1
)

echo All dependency paths validated successfully!
echo.

REM Check if build directory exists
if exist build (
    echo Build directory exists. Cleaning...
    rmdir /s /q build
)

REM Create build directory
echo Creating build directory...
mkdir build
cd build

echo.
echo ===============================================
echo Configuring CMake...
echo ===============================================
cmake .. -G "Visual Studio 17 2022" -A x64 ^
  -DCMAKE_TOOLCHAIN_FILE="%VCPKG_ROOT%\scripts\buildsystems\vcpkg.cmake" ^
    -DCMAKE_CXX_FLAGS="/utf-8" ^
  -DOpenVINO_DIR="%OPENVINO_DIR%" ^
  -DOpenVINOGenAI_DIR="%OPENVINO_GENAI_DIR%" ^
  -DTBB_DIR="%TBB_DIR%" ^
  -DVPL_DIR="%VPL_DIR%" ^
  -DCMAKE_BUILD_TYPE=Release

if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: CMake configuration failed
    echo Please check the error messages above
    pause
    exit /b 1
)

echo.
echo ===============================================
echo Building Flama...
echo ===============================================
cmake --build . --config Release

if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: Build failed
    echo Please check the error messages above
    pause
    exit /b 1
)

echo.
echo ===============================================
echo Build Successful!
echo ===============================================
echo.
echo Executable location: build\bin\Release\flama.exe
echo.
echo To run the application:
echo   cd build\bin\Release
echo   flama.exe "path\to\video.mp4" hw
echo.
pause
