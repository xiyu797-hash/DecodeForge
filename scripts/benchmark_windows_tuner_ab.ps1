param([Parameter(Mandatory=$true)][string]$BaselineBuild,[Parameter(Mandatory=$true)][string]$CandidateBuild,[string]$CudaToolkitRoot='E:\DecodeForge-rc1-toolkit',[ValidateRange(1,100)][int]$Rounds=20,[switch]$AdditionalShapes)
$ErrorActionPreference='Stop'
$exes=@{A=(Join-Path (Resolve-Path -LiteralPath $BaselineBuild).Path 'decodeforge_cli.exe');B=(Join-Path (Resolve-Path -LiteralPath $CandidateBuild).Path 'decodeforge_cli.exe')}
foreach($exe in $exes.Values){if(-not(Test-Path -LiteralPath $exe -PathType Leaf)){throw "Missing exe: $exe"}}
$root=Split-Path -Parent $PSScriptRoot
$out=Join-Path $root ('validation\tuner-ab-'+(Get-Date -Format 'yyyyMMdd-HHmmss-fffffff'))
New-Item -ItemType Directory -Path $out -ErrorAction Stop|Out-Null
$env:PATH="$(Join-Path $CudaToolkitRoot 'bin/x64');$(Join-Path $CudaToolkitRoot 'bin');$env:PATH"
@("date=$(Get-Date -Format o)","A=$($exes.A)","B=$($exes.B)","rounds=$Rounds",'clocks unlocked; CPU/GPU activity not continuously monitored')|Set-Content (Join-Path $out 'environment.txt') -Encoding utf8
Get-FileHash -LiteralPath $exes.A,$exes.B -Algorithm SHA256|Export-Csv (Join-Path $out 'hashes.csv') -NoTypeInformation
& nvidia-smi --query-gpu=name,driver_version,temperature.gpu,utilization.gpu,clocks.sm,memory.used --format=csv | Add-Content (Join-Path $out 'environment.txt')
if($LASTEXITCODE -ne 0){throw 'nvidia-smi failed'}
$shapes=if($AdditionalShapes){@(@{N=256;K=256;B=1;G=16},@{N=1024;K=1024;B=1;G=64},@{N=1024;K=1024;B=4;G=128},@{N=2048;K=4096;B=8;G=32})}else{@(@{N=1024;K=256;B=1;G=64})}
'round,shape,position,variant,timestamp'|Set-Content (Join-Path $out 'schedule.csv') -Encoding ascii
for($round=1;$round -le $Rounds;$round++){$index=0;foreach($s in $shapes){$index++;$order=if(($round+$index)%2 -eq 0){@('A','B')}else{@('B','A')};$pos=0;foreach($kind in $order){$pos++;$shape='n{0}-k{1}-b{2}-g{3}' -f $s.N,$s.K,$s.B,$s.G;$tag='r{0:D3}-{1}-{2}' -f $round,$shape,$kind;"$round,$shape,$pos,$kind,$(Get-Date -Format o)"|Add-Content (Join-Path $out 'schedule.csv') -Encoding ascii;Write-Host "[TUNER AB] $tag";& ($exes[$kind]) bench --n $s.N --k $s.K --batch $s.B --group $s.G --warmup 20 --repeats 100 --sample-run $round --tune-samples (Join-Path $out "$tag-tune.csv") --choice-samples (Join-Path $out "$tag-choice.csv") --csv (Join-Path $out "$tag-summary.csv") 2>&1|Tee-Object -FilePath (Join-Path $out "$tag.log");if($LASTEXITCODE -ne 0){throw "Failed $tag; evidence $out"}}}}
Write-Host "[TUNER AB] Reports: $out"
