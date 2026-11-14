param([Parameter(Mandatory=$true)][string]$ExperimentDirectory)
$ErrorActionPreference='Stop'
$dir=(Resolve-Path -LiteralPath $ExperimentDirectory).Path
$baseline=@(Get-ChildItem -LiteralPath $dir -Filter '*-baseline-choice.csv' | Sort-Object Name)
if($baseline.Count -eq 0){throw 'No baseline samples found.'}
function Median([double[]]$values) {
    $sorted=@($values|Sort-Object)
    if($sorted.Count -eq 0){throw 'No values for median.'}
    $i=[int][math]::Floor($sorted.Count/2)
    if($sorted.Count%2){return $sorted[$i]}
    return ($sorted[$i-1]+$sorted[$i])/2.0
}
$rows=foreach($f in $baseline){
    $candidate=$f.FullName.Replace('-baseline-choice.csv','-candidate-choice.csv')
    if(-not (Test-Path -LiteralPath $candidate -PathType Leaf)){throw "Missing candidate for $($f.Name)"}
    $a=@(Import-Csv -LiteralPath $f.FullName); $b=@(Import-Csv -LiteralPath $candidate)
    if($a.Count -ne 9 -or $b.Count -ne 9){throw 'Expected nine samples per process.'}
    foreach($field in @('n','k','batch','group','run')){if($a[0].$field -ne $b[0].$field){throw "Shape mismatch in $($f.Name)"}}
    $ad=Median @($a|ForEach-Object {[double]$_.direct_ms}); $bd=Median @($b|ForEach-Object {[double]$_.direct_ms})
    $ap=Median @($a|ForEach-Object {[double]$_.paired_ms}); $bp=Median @($b|ForEach-Object {[double]$_.paired_ms})
    if($bd -le 0 -or $bp -le 0){throw 'Nonpositive candidate timing.'}
    [pscustomobject]@{n=$a[0].n;k=$a[0].k;batch=$a[0].batch;group=$a[0].group;run=$a[0].run;baseline_direct_ms=$ad;candidate_direct_ms=$bd;direct_baseline_over_candidate=$ad/$bd;baseline_paired_ms=$ap;candidate_paired_ms=$bp;paired_baseline_over_candidate=$ap/$bp}
}
$out=Join-Path $dir ('ab-analysis-'+(Get-Date -Format 'yyyyMMdd-HHmmss-fffffff')+'.csv')
$rows|Export-Csv -LiteralPath $out -NoTypeInformation -Encoding utf8
$md=Median @($rows|ForEach-Object {$_.direct_baseline_over_candidate}); $mp=Median @($rows|ForEach-Object {$_.paired_baseline_over_candidate})
$worseD=@($rows|Where-Object {$_.direct_baseline_over_candidate -lt 1}).Count
$worseP=@($rows|Where-Object {$_.paired_baseline_over_candidate -lt 1}).Count
Write-Host "Pairs=$($rows.Count) median_Direct_baseline_over_candidate=$md candidate_slower_Direct=$worseD median_Paired_baseline_over_candidate=$mp candidate_slower_Paired=$worseP"
Write-Host "Analysis: $out"
Write-Host 'Ratios compare separate processes, not interleaved launches; no clocks were locked and only the recorded rounds were measured.'
