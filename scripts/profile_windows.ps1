param(
    [Parameter(Mandatory = $true)]
    [string]$Executable,
    [string]$Output = '',
    [string]$CudaToolkitRoot = '',
    [string]$Ncu = 'E:\NsightCompute-2026.3.1\nsight_compute-windows-x86_64-2026.3.1.2-archive\nsight-compute\2026.3.1\ncu.bat'
)
$ErrorActionPreference = 'Stop'
if (-not (Test-Path -LiteralPath $Executable -PathType Leaf)) { throw "Executable not found: $Executable" }
if (-not (Test-Path -LiteralPath $Ncu -PathType Leaf)) { throw "Nsight Compute not found: $Ncu" }
if (-not $Output) {
    $projectRoot = Split-Path -Parent $PSScriptRoot
    $Output = Join-Path $projectRoot ("validation\nsight-{0}-{1}\decodeforge" -f (Get-Date -Format 'yyyyMMdd-HHmmss'), [guid]::NewGuid().ToString('N').Substring(0, 8))
}
$outputDir = Split-Path -Parent $Output
if ($outputDir) { New-Item -ItemType Directory -Path $outputDir -Force | Out-Null }
$oldPath = $env:PATH
try {
    $cudaRoot = if ($CudaToolkitRoot) { $CudaToolkitRoot } elseif (Test-Path -LiteralPath 'E:\DecodeForge-CUDA-13.4\bin\nvcc.exe') { 'E:\DecodeForge-CUDA-13.4' } else { $env:CUDA_PATH }
    if ($cudaRoot -and (Test-Path -LiteralPath (Join-Path $cudaRoot 'bin'))) {
        $env:PATH = "$(Join-Path $cudaRoot 'bin\x64');$(Join-Path $cudaRoot 'bin');$env:PATH"
    }
    & $Ncu --set basic --launch-count 1 --export $Output $Executable gpu-check --n 1024 --k 256 --batch 1 --group 64
    $code = $LASTEXITCODE
} finally {
    $env:PATH = $oldPath
}
if ($code -ne 0) { throw "Nsight Compute failed with exit code $code; check the diagnostic above (GPU counter permission may be required)." }
