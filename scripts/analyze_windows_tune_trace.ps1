param([Parameter(Mandatory=$true)][string]$ExperimentDirectory)
$ErrorActionPreference='Stop'
$dir=(Resolve-Path -LiteralPath $ExperimentDirectory).Path
$files=@(Get-ChildItem -LiteralPath $dir -Filter '*-tune.csv'|Sort-Object Name)
if($files.Count -eq 0){throw 'No tuner traces found.'}
$results=foreach($file in $files){
    $t=@(Import-Csv -LiteralPath $file.FullName)
    $choice=$file.FullName.Replace('-tune.csv','-choice.csv')
    if(-not(Test-Path -LiteralPath $choice)){throw "Missing holdout: $choice"}
    $c=@(Import-Csv -LiteralPath $choice)
    if($t.Count -ne 9 -or $c.Count -ne 9){throw "Expected 9 paired rows: $($file.Name)"}
    foreach($field in @('n','k','batch','group','run','selected')){
        if(@($t|Where-Object {$_.$field -ne $t[0].$field}).Count -ne 0 -or @($c|Where-Object {$_.$field -ne $t[0].$field}).Count -ne 0){throw "Inconsistent $field in $($file.Name)"}
    }
    for($i=0;$i -lt 9;$i++){
        $first=if($i%2 -eq 0){'direct'}else{'grouped_paired'}
        if($t[$i].first_kernel -ne $first -or [int]$t[$i].trial -ne $i+1){throw "Invalid tuner order in $($file.Name)"}
        foreach($r in @($t[$i].direct_over_grouped,$c[$i].direct_over_paired)){if(-not [double]::IsFinite([double]$r) -or [double]$r -le 0){throw 'Invalid timing ratio'}}
    }
    $rt=@($t|ForEach-Object {[double]$_.direct_over_grouped}|Sort-Object)
    $rc=@($c|ForEach-Object {[double]$_.direct_over_paired}|Sort-Object)
    $selected=$t[0].selected
    $expected=if($rt[4] -le 1){'direct'}else{'grouped_paired'}
    $hold=if($rc[4] -le 1){'direct'}else{'grouped_paired'}
    if($selected -ne $expected){throw "Trace does not reproduce decision: $($file.Name)"}
    $loss=if($selected -eq $hold){0.0}elseif($selected -eq 'direct'){($rc[4]-1)*100}else{(1/$rc[4]-1)*100}
    [pscustomobject]@{run=$t[0].run;n=$t[0].n;k=$t[0].k;batch=$t[0].batch;group=$t[0].group;selected=$selected;internal_median_ratio=$rt[4];holdout_median_ratio=$rc[4];holdout_winner=$hold;agrees=($selected -eq $hold);observed_loss_pct=$loss;wall_ms=[double]$t[0].autotune_wall_ms}
}
$out=Join-Path $dir ('trace-analysis-'+(Get-Date -Format 'yyyyMMdd-HHmmss-fffffff')+'.csv')
$results|Export-Csv -LiteralPath $out -NoTypeInformation -Encoding utf8
$wall=@($results|ForEach-Object {$_.wall_ms}|Sort-Object);$i=[int][math]::Floor($wall.Count/2)
$median=if($wall.Count%2){$wall[$i]}else{($wall[$i-1]+$wall[$i])/2}
Write-Host "Runs=$($results.Count) agreement=$(@($results|Where-Object agrees).Count) max_observed_loss_pct=$(($results|Measure-Object observed_loss_pct -Maximum).Maximum) median_autotune_wall_ms=$median"
Write-Host "Analysis: $out"
Write-Host 'Holdout is an independent later measurement, not noise-free ground truth.'
