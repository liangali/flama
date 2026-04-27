# PowerShell script to copy all required DLLs to the Release directory

$ErrorActionPreference = "Stop"

# Target directory
$targetDir = Join-Path $PSScriptRoot "build\bin\Release"

function Get-LatestMatchingDir([string]$pattern) {
    $items = Get-ChildItem -Path $pattern -Directory -ErrorAction SilentlyContinue |
        Sort-Object Name -Descending
    if ($items -and $items.Count -gt 0) {
        return $items[0].FullName
    }
    return $null
}

# Ensure target directory exists
if (-not (Test-Path $targetDir)) {
    New-Item -ItemType Directory -Path $targetDir -Force | Out-Null
}

Write-Host "Copying DLLs to $targetDir..." -ForegroundColor Green

# 1. OpenVINO Runtime DLLs
$openvinoRoot = if ($env:OPENVINO_ROOT) { $env:OPENVINO_ROOT } else { Get-LatestMatchingDir "D:\library\openvino\openvino_toolkit_windows_2026.1*" }
$openvinoDir = if ($openvinoRoot) { Join-Path $openvinoRoot "runtime\bin\intel64\Release" } else { $null }
if ($openvinoDir -and (Test-Path $openvinoDir)) {
    Write-Host "  Copying OpenVINO Runtime DLLs..." -ForegroundColor Cyan
    Copy-Item "$openvinoDir\*.dll" -Destination $targetDir -Force
}

# 2. TBB DLLs
$tbbDir = if ($openvinoRoot) { Join-Path $openvinoRoot "runtime\3rdparty\tbb\bin" } else { $null }
if ($tbbDir -and (Test-Path $tbbDir)) {
    Write-Host "  Copying TBB DLLs..." -ForegroundColor Cyan
    Copy-Item "$tbbDir\*.dll" -Destination $targetDir -Force
}

# 3. OpenVINO GenAI DLLs
$genaiRoot = if ($env:OPENVINO_GENAI_ROOT) { $env:OPENVINO_GENAI_ROOT } else { Get-LatestMatchingDir "D:\library\openvino.genai\openvino_genai_windows_2026.1*" }
$genaiDir = if ($genaiRoot) { Join-Path $genaiRoot "runtime\bin\intel64\Release" } else { $null }
if ($genaiDir -and (Test-Path $genaiDir)) {
    Write-Host "  Copying OpenVINO GenAI DLLs..." -ForegroundColor Cyan
    Copy-Item "$genaiDir\*.dll" -Destination $targetDir -Force
}

# 4. VPL DLLs
$vplDir = Join-Path $PSScriptRoot "thirdparty\_vplinstall\bin"
if (Test-Path $vplDir) {
    Write-Host "  Copying VPL DLLs..." -ForegroundColor Cyan
    Copy-Item "$vplDir\*.dll" -Destination $targetDir -Force
}

# 5. vcpkg DLLs (FFmpeg, etc.)
$vcpkgDir = Join-Path $PSScriptRoot "build\vcpkg_installed\x64-windows\bin"
if (Test-Path $vcpkgDir) {
    Write-Host "  Copying vcpkg DLLs (FFmpeg, etc.)..." -ForegroundColor Cyan
    Copy-Item "$vcpkgDir\*.dll" -Destination $targetDir -Force
}

Write-Host ""
Write-Host "DLL copy completed!" -ForegroundColor Green
Write-Host "Total DLLs in target directory:" -ForegroundColor Yellow
$dllCount = (Get-ChildItem "$targetDir\*.dll" | Measure-Object).Count
Write-Host "  $dllCount DLL files" -ForegroundColor Yellow
