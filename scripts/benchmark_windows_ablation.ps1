param(
    [Parameter(Mandatory=$true)][string]$BuildDirectory,
    [string]$CudaToolkitRoot = 'E:\DecodeForge-CUDA-13.4',
    [int]$Device = 0,
    [int]$Repeats = 200,
    [int]$Warmup = 30
)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$build = (Resolve-Path -LiteralPath $BuildDirectory -ErrorAction Stop).Path
$exe = Join-Path $build 'decodeforge_cli.exe'
if (-not (Test-Path -LiteralPath $exe)) { throw "No CLI in $build" }
$env:PATH = "$(Join-Path $CudaToolkitRoot 'bin\x64');$(Join-Path $CudaToolkitRoot 'bin');$env:PATH"
$out = Join-Path $root ("validation\ablation-" + (Get-Date -Format 'yyyyMMdd-HHmmss-ffff'))
New-Item -Path $out -ItemType Directory -Force | Out-Null
"date=$(Get-Date -Format o)" | Out-File (Join-Path $out 'environment.txt') -Encoding utf8
& nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,driver_version --format=csv |
    Out-File (Join-Path $out 'environment.txt') -Append -Encoding utf8
& (Join-Path $CudaToolkitRoot 'bin\nvcc.exe') --version |
    Out-File (Join-Path $out 'environment.txt') -Append -Encoding utf8
foreach ($batch in @(1,2,4,8)) {
    foreach ($k in @(256,1024,4096)) {
        $n = if ($k -eq 4096) { 2048 } else { 1024 }
        $file = Join-Path $out "b$batch-k$k.csv"
        Write-Host "[ABLATION] B=$batch N=$n K=$k, seven paired trials per kernel" -ForegroundColor Cyan
        & $exe bench --n $n --k $k --batch $batch --group 64 --device $Device --warmup $Warmup --repeats $Repeats --csv $file
        if ($LASTEXITCODE -ne 0) { throw "Benchmark failed: B=$batch K=$k" }
    }
}
$rows = Get-ChildItem -LiteralPath $out -Filter 'b*-k*.csv' | ForEach-Object { Import-Csv $_.FullName }
$rows | Select-Object n,k,batch,group,grouped_fused_ms,grouped_scalar_ms,paired_speedup_vs_scalar,selected_kernel |
    Export-Csv (Join-Path $out 'summary.csv') -NoTypeInformation -Encoding utf8
Write-Host "[ABLATION] Reports written to $out"
