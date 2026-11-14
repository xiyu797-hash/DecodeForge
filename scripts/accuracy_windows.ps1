param([Parameter(Mandatory=$true)][string]$BuildDirectory,[Parameter(Mandatory=$true)][string]$CudaToolkitRoot)
$ErrorActionPreference='Stop'
$root=Split-Path -Parent $PSScriptRoot
$build=(Resolve-Path -LiteralPath $BuildDirectory).Path
$exe=Join-Path $build 'decodeforge_accuracy_cli.exe'
if(-not(Test-Path -LiteralPath $exe -PathType Leaf)){throw "Missing accuracy executable: $exe"}
if(-not(Test-Path -LiteralPath (Join-Path $CudaToolkitRoot 'bin/nvcc.exe') -PathType Leaf)){throw 'CUDA toolkit not found'}
$out=Join-Path $root ('validation\accuracy-'+(Get-Date -Format 'yyyyMMdd-HHmmss')+'-'+[guid]::NewGuid().ToString('N').Substring(0,8))
New-Item -ItemType Directory -Path $out -ErrorAction Stop|Out-Null
Get-FileHash -LiteralPath $exe -Algorithm SHA256|Format-List|Out-File (Join-Path $out 'binary-hash.txt') -Encoding utf8
& nvidia-smi --query-gpu=name,driver_version,temperature.gpu,utilization.gpu,memory.used --format=csv|Out-File (Join-Path $out 'environment.txt') -Encoding utf8
if($LASTEXITCODE -ne 0){throw "nvidia-smi failed; $out"}
$oldPath=$env:PATH
try {
    $env:PATH="$(Join-Path $CudaToolkitRoot 'bin/x64');$(Join-Path $CudaToolkitRoot 'bin');$oldPath"
    $saved=$ErrorActionPreference
    try {
        $ErrorActionPreference='Continue'
        & $exe 2> (Join-Path $out 'stderr.log') | Set-Content -LiteralPath (Join-Path $out 'accuracy.csv') -Encoding utf8
        $code=$LASTEXITCODE
    } finally {$ErrorActionPreference=$saved}
    "exit_code=$code"|Set-Content (Join-Path $out 'exit.txt') -Encoding ascii
    if($code -ne 0){throw "Accuracy executable failed ($code); see $out"}
    $rows=@(Import-Csv -LiteralPath (Join-Path $out 'accuracy.csv'))
    if($rows.Count -ne 256 -or @($rows|Where-Object {[int]$_.mismatched -ne 0}).Count -ne 0){throw "Accuracy report incomplete or has mismatches; see $out"}
    Write-Host "[ACCURACY] 256 shape/path rows passed. Logs: $out"
} finally {$env:PATH=$oldPath}
