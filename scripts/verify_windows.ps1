param(
    [switch]$CpuOnly,
    [switch]$SkipSanitizer,
    [string]$CudaToolkitRoot = '',
    [string]$SanitizerPath = ''
)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$stamp = (Get-Date -Format 'yyyyMMdd-HHmmss-ffff') + '-' + [guid]::NewGuid().ToString('N').Substring(0,8)
$out = Join-Path $root "validation\windows-$stamp"
$build = Join-Path $root "build\windows-$stamp"
New-Item -ItemType Directory -Path $out -ErrorAction Stop | Out-Null
function Invoke-Logged([string]$Name, [scriptblock]$Command) {
    & $Command 2>&1 | Tee-Object -FilePath (Join-Path $out "$Name.log")
    if ($LASTEXITCODE -ne 0) { throw "$Name failed with exit code $LASTEXITCODE" }
}
if ($CpuOnly) {
    Invoke-Logged 'configure' { cmake -S $root -B $build -G 'Visual Studio 18 2026' -A x64 -DDF_ENABLE_CUDA=OFF -DBUILD_TESTING=ON }
    Invoke-Logged 'build' { cmake --build $build --config Release --parallel }
    Invoke-Logged 'tests' { ctest --test-dir $build -C Release --output-on-failure }
} else {
    if (-not $CudaToolkitRoot) {
        if (Test-Path -LiteralPath 'E:\DecodeForge-CUDA-13.4\bin\nvcc.exe') { $CudaToolkitRoot = 'E:\DecodeForge-CUDA-13.4' }
        elseif (Test-Path -LiteralPath 'E:\DecodeForge-rc1-toolkit\bin\nvcc.exe') { $CudaToolkitRoot = 'E:\DecodeForge-rc1-toolkit' }
        else { $CudaToolkitRoot = $env:CUDA_PATH }
    }
    if (-not (Test-Path -LiteralPath (Join-Path $CudaToolkitRoot 'bin\nvcc.exe'))) { throw 'Supply -CudaToolkitRoot pointing to a CUDA toolkit.' }
    if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
        $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
        $vs = & $vswhere -latest -version '[18.0,19.0)' -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
        if (-not $vs) { throw 'Visual Studio 2026 with x64 C++ tools was not found.' }
        $devcmd = Join-Path $vs 'Common7\Tools\VsDevCmd.bat'
        $variables = & cmd.exe /d /s /c "`"`"$devcmd`" -arch=x64 -host_arch=x64 >nul && set`""
        if ($LASTEXITCODE -ne 0) { throw 'Visual Studio developer environment initialization failed.' }
        foreach ($line in $variables) { if ($line -match '^([^=]+)=(.*)$') { [Environment]::SetEnvironmentVariable($matches[1], $matches[2], 'Process') } }
    }
    $env:PATH = "$(Join-Path $CudaToolkitRoot 'bin\x64');$(Join-Path $CudaToolkitRoot 'bin');$env:PATH"
    $nvcc = Join-Path $CudaToolkitRoot 'bin\nvcc.exe'
    Invoke-Logged 'device' { nvidia-smi }
    Invoke-Logged 'configure' { cmake -S $root -B $build -G Ninja -DCMAKE_BUILD_TYPE=Release -DDF_ENABLE_CUDA=ON -DBUILD_TESTING=ON "-DCMAKE_CUDA_COMPILER=$nvcc" "-DCUDAToolkit_ROOT=$CudaToolkitRoot" }
    Invoke-Logged 'build' { cmake --build $build --parallel }
    Invoke-Logged 'tests' { ctest --test-dir $build --output-on-failure }
    Invoke-Logged 'device-check' { & (Join-Path $build 'decodeforge_cli.exe') device }
    Invoke-Logged 'benchmark' { & (Join-Path $build 'decodeforge_cli.exe') bench-all --repeats 100 --warmup 20 --csv (Join-Path $out 'w4a16.csv') }
    foreach ($dtype in @('fp16','bf16')) {
        Invoke-Logged "gemm-$dtype" { & (Join-Path $build 'decodeforge_ops_cli.exe') gemm-bench --m 256 --n 256 --k 512 --dtype $dtype --csv (Join-Path $out "gemm-$dtype.csv") }
    }
    Invoke-Logged 'norm' { & (Join-Path $build 'decodeforge_ops_cli.exe') norm-bench --rows 128 --cols 1024 --csv (Join-Path $out 'norm.csv') }
    if ($SkipSanitizer) {
        Write-Warning 'Sanitizer skipped by request; this build is not fully GPU-safety qualified.'
    } else {
        & (Join-Path $PSScriptRoot 'verify_windows_sanitizer.ps1') -BuildDirectory $build -CudaToolkitRoot $CudaToolkitRoot -SanitizerPath $SanitizerPath
    }
}
Write-Host "Logs: $out"
