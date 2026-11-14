param(
    [Parameter(Mandatory=$true)][string]$BaselineBuild,
    [Parameter(Mandatory=$true)][string]$CandidateBuild,
    [Parameter(Mandatory=$true)][string]$CudaToolkitRoot,
    [ValidateRange(1,20)][int]$Rounds=2,
    [ValidateRange(1,100000)][int]$Repeats=100,
    [ValidateRange(1,100000)][int]$Warmup=20,
    [switch]$RegressionShapes,
    [switch]$ThresholdHoldout,
    [switch]$RefactorRegressions,
    [switch]$BoundaryRegressions,
    [switch]$DirectRegressionsOnly,
    [switch]$MatrixRegressions
)
$ErrorActionPreference='Stop'
$root=Split-Path -Parent $PSScriptRoot
$exes=@{baseline=(Join-Path (Resolve-Path -LiteralPath $BaselineBuild).Path 'decodeforge_cli.exe'); candidate=(Join-Path (Resolve-Path -LiteralPath $CandidateBuild).Path 'decodeforge_cli.exe')}
foreach($exe in $exes.Values) { if(-not (Test-Path -LiteralPath $exe -PathType Leaf)) { throw "Missing executable: $exe" } }
if (-not (Test-Path -LiteralPath (Join-Path $CudaToolkitRoot 'bin/nvcc.exe'))) { throw 'CUDA toolkit not found.' }
$env:PATH="$(Join-Path $CudaToolkitRoot 'bin/x64');$(Join-Path $CudaToolkitRoot 'bin');$env:PATH"
$out=Join-Path $root ('validation\build-ab-' + (Get-Date -Format 'yyyyMMdd-HHmmss-fffffff'))
New-Item -ItemType Directory -Path $out -ErrorAction Stop | Out-Null
@("date=$(Get-Date -Format o)","baseline=$($exes.baseline)","candidate=$($exes.candidate)","toolkit=$CudaToolkitRoot", "rounds=$Rounds repeats=$Repeats warmup=$Warmup; clocks unlocked") | Set-Content -LiteralPath (Join-Path $out 'environment.txt') -Encoding utf8
Get-FileHash -Algorithm SHA256 -LiteralPath $exes.baseline,$exes.candidate | Format-List | Out-File -LiteralPath (Join-Path $out 'executable-hashes.txt') -Encoding utf8
'round,shape_index,n,k,batch,group,position,variant,timestamp' | Set-Content -LiteralPath (Join-Path $out 'schedule.csv') -Encoding ascii
& nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,clocks.sm,clocks.mem,driver_version --format=csv | Add-Content -LiteralPath (Join-Path $out 'environment.txt')
if($LASTEXITCODE -ne 0) { throw 'nvidia-smi failed' }
if(@($RegressionShapes,$ThresholdHoldout,$RefactorRegressions,$BoundaryRegressions,$DirectRegressionsOnly,$MatrixRegressions | Where-Object { $_ }).Count -gt 1){throw 'Choose one shape set.'}
$shapes=if($RegressionShapes) {
    @(@{N=256;K=256;B=4;G=64},@{N=1024;K=1024;B=2;G=16},@{N=65;K=129;B=4;G=32})
} elseif($RefactorRegressions) {
    @(@{N=1024;K=256;B=1;G=64},@{N=256;K=256;B=1;G=16},@{N=256;K=256;B=2;G=32},@{N=256;K=256;B=4;G=16},@{N=256;K=256;B=8;G=16},@{N=256;K=256;B=8;G=128})
} elseif($BoundaryRegressions) {
    @(@{N=1537;K=2047;B=1;G=32},@{N=513;K=1023;B=1;G=128},@{N=513;K=1536;B=1;G=128},@{N=513;K=1536;B=4;G=128},@{N=513;K=2047;B=1;G=128},@{N=513;K=2047;B=1;G=32},@{N=513;K=2049;B=1;G=32})
} elseif($DirectRegressionsOnly) {
    @(@{N=513;K=1025;B=1;G=128},@{N=1024;K=256;B=1;G=64})
} elseif($MatrixRegressions) {
    @(@{N=256;K=256;B=2;G=64},@{N=65;K=129;B=1;G=128},@{N=1024;K=1024;B=1;G=128},@{N=256;K=256;B=1;G=128},@{N=65;K=129;B=2;G=16},@{N=65;K=129;B=8;G=64})
} elseif($ThresholdHoldout) {
    @(foreach($n in @(513,1537)) { foreach($k in @(1023,1025,1536,2047,2048,2049,3073)) { foreach($b in @(1,4)) { foreach($g in @(32,128)) { @{N=$n;K=$k;B=$b;G=$g} } } } })
} else {
    @(foreach($g in @(16,32,64,128)) { foreach($b in @(1,2,4,8)) { foreach($pair in @(@(65,129),@(256,256),@(1024,1024),@(2048,4096))) { @{N=$pair[0];K=$pair[1];B=$b;G=$g} } } })
    @{N=1024;K=256;B=1;G=64}
}
for($round=1;$round -le $Rounds;$round++) {
  $shapeIndex=0
  foreach($s in $shapes) {
    $shapeIndex++
    # Alternate order for every shape across rounds, regardless of shape count.
    $order=if(($round+$shapeIndex)%2 -eq 0) {@('baseline','candidate')} else {@('candidate','baseline')}
    $position=0
    foreach($kind in $order) {
        $position++
        $tag='r{0:D3}-n{1}-k{2}-b{3}-g{4}-{5}' -f $round,$s.N,$s.K,$s.B,$s.G,$kind
        "$round,$shapeIndex,$($s.N),$($s.K),$($s.B),$($s.G),$position,$kind,$(Get-Date -Format o)" | Add-Content -LiteralPath (Join-Path $out 'schedule.csv') -Encoding ascii
        Write-Host "[AB] $tag"
        & ($exes[$kind]) bench --n $s.N --k $s.K --batch $s.B --group $s.G --warmup $Warmup --repeats $Repeats --csv (Join-Path $out "$tag.csv") --choice-samples (Join-Path $out "$tag-choice.csv") --sample-run $round 2>&1 | Tee-Object -FilePath (Join-Path $out "$tag.log")
        if($LASTEXITCODE -ne 0) {throw "Benchmark failed: $tag (see $out)"}
    }
} }
Write-Host "[AB] Reports: $out"
