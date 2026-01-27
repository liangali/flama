# PowerShell script to copy all required DLLs to the Release directory

$ErrorActionPreference = "Stop"

# Target directory
$targetDir = "D:\code\flama_code\flama\build\bin\Release"

# Ensure target directory exists
if (-not (Test-Path $targetDir)) {
    New-Item -ItemType Directory -Path $targetDir -Force | Out-Null
}

Write-Host "Copying DLLs to $targetDir..." -ForegroundColor Green

# 1. OpenVINO Runtime DLLs
$openvinoDir = "D:\library\openvino\openvino_toolkit_windows_2025.4.2.20430.85e49f27be1_x86_64\runtime\bin\intel64\Release"
if (Test-Path $openvinoDir) {
    Write-Host "  Copying OpenVINO Runtime DLLs..." -ForegroundColor Cyan
    Copy-Item "$openvinoDir\*.dll" -Destination $targetDir -Force
}

# 2. TBB DLLs
$tbbDir = "D:\library\openvino\openvino_toolkit_windows_2025.4.2.20430.85e49f27be1_x86_64\runtime\3rdparty\tbb\bin"
if (Test-Path $tbbDir) {
    Write-Host "  Copying TBB DLLs..." -ForegroundColor Cyan
    Copy-Item "$tbbDir\*.dll" -Destination $targetDir -Force
}

# 3. OpenVINO GenAI DLLs
$genaiDir = "D:\library\openvino.genai\openvino_genai_windows_2025.4.2.0_x86_64\runtime\bin\intel64\Release"
if (Test-Path $genaiDir) {
    Write-Host "  Copying OpenVINO GenAI DLLs..." -ForegroundColor Cyan
    Copy-Item "$genaiDir\*.dll" -Destination $targetDir -Force
}

# 4. VPL DLLs
$vplDir = "D:\code\flama_code\flama\thirdparty\_vplinstall\bin"
if (Test-Path $vplDir) {
    Write-Host "  Copying VPL DLLs..." -ForegroundColor Cyan
    Copy-Item "$vplDir\*.dll" -Destination $targetDir -Force
}

# 5. vcpkg DLLs (FFmpeg, etc.)
$vcpkgDir = "D:\code\flama_code\flama\build\vcpkg_installed\x64-windows\bin"
if (Test-Path $vcpkgDir) {
    Write-Host "  Copying vcpkg DLLs (FFmpeg, etc.)..." -ForegroundColor Cyan
    Copy-Item "$vcpkgDir\*.dll" -Destination $targetDir -Force
}

Write-Host ""
Write-Host "DLL copy completed!" -ForegroundColor Green
Write-Host "Total DLLs in target directory:" -ForegroundColor Yellow
$dllCount = (Get-ChildItem "$targetDir\*.dll" | Measure-Object).Count
Write-Host "  $dllCount DLL files" -ForegroundColor Yellow
