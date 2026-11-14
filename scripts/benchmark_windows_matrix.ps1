param(
    [Parameter(Mandatory=$true)][string]$BuildDirectory,
    [Parameter(Mandatory=$true)][string]$CudaToolkitRoot,
    [int]$Device = 0,
    [ValidateRange(1,100)][int]$Rounds = 2,
    [ValidateRange(1,100000)][int]$Repeats = 100,
    [ValidateRange(1,100000)][int]$Warmup = 20,
    [switch]$FocusShape
)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$build = (Resolve-Path -LiteralPath $BuildDirectory).Path
$exe = Join-Path $build 'decodeforge_cli.exe'
if (-not (Test-Path -LiteralPath $exe)) { throw "Missing executable: $exe" }
if (-not (Test-Path -LiteralPath (Join-Path $CudaToolkitRoot 'bin/nvcc.exe'))) { throw 'CUDA toolkit not found.' }
$env:PATH = "$(Join-Path $CudaToolkitRoot 'bin/x64');$(Join-Path $CudaToolkitRoot 'bin');$env:PATH"
$out = Join-Path $root ("validation\matrix-raw-" + (Get-Date -Format 'yyyyMMdd-HHmmss-fffffff'))
New-Item -ItemType Directory -Path $out -ErrorAction Stop | Out-Null
@("date=$(Get-Date -Format o)", "build=$build", "toolkit=$CudaToolkitRoot", "rounds=$Rounds repeats=$Repeats warmup=$Warmup focus=$FocusShape", 'GPU clocks unlocked; measurements executed sequentially.') | Set-Content -LiteralPath (Join-Path $out 'environment.txt') -Encoding utf8
& nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,clocks.sm,clocks.mem,driver_version --format=csv | Add-Content -LiteralPath (Join-Path $out 'environment.txt')
if ($LASTEXITCODE -ne 0) { throw 'nvidia-smi failed' }
& (Join-Path $CudaToolkitRoot 'bin/nvcc.exe') --version | Add-Content -LiteralPath (Join-Path $out 'environment.txt')
if ($LASTEXITCODE -ne 0) { throw 'nvcc version failed' }
$shapes = if ($FocusShape) { @(@{N=1024; K=256; B=1; G=64}) } else {
    foreach ($g in @(16,32,64,128)) { foreach ($b in @(1,2,4,8)) {
        foreach ($pair in @(@(65,129),@(256,256),@(1024,1024),@(2048,4096))) {
            @{N=$pair[0]; K=$pair[1]; B=$b; G=$g}
        }
    } }
}
for ($round=1; $round -le $Rounds; $round++) {
    foreach ($s in $shapes) {
        $tag = 'r{0:D3}-n{1}-k{2}-b{3}-g{4}' -f $round,$s.N,$s.K,$s.B,$s.G
        Write-Host "[BENCH] $tag"
        & $exe bench --n $s.N --k $s.K --batch $s.B --group $s.G --device $Device --warmup $Warmup --repeats $Repeats --csv (Join-Path $out "$tag.csv") --samples (Join-Path $out "$tag-samples.csv") --choice-samples (Join-Path $out "$tag-choice.csv") --sample-run $round 2>&1 | Tee-Object -FilePath (Join-Path $out "$tag.log")
        if ($LASTEXITCODE -ne 0) { throw "Benchmark failed: $tag (see $out)" }
    }
}
$files = Get-ChildItem -LiteralPath $out -Filter 'r*.csv' | Where-Object { $_.Name -notlike '*-samples.csv' -and $_.Name -notlike '*-choice.csv' }
$rows = foreach ($file in $files) { Import-Csv -LiteralPath $file.FullName | Select-Object @{Name='run';Expression={$file.BaseName}},* }
$rows | Export-Csv -LiteralPath (Join-Path $out 'summary.csv') -NoTypeInformation -Encoding utf8
Write-Host "[BENCH] Reports: $out"
