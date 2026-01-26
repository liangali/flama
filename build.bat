@echo off
REM Flama Build Script for Windows
REM This script automates the CMake configuration and build process

echo ===============================================
echo Flama Build Script
echo ===============================================
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
