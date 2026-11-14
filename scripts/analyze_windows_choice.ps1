param([Parameter(Mandatory=$true)][string]$ExperimentDirectory)
$ErrorActionPreference = 'Stop'
$dir = (Resolve-Path -LiteralPath $ExperimentDirectory).Path
$files = @(Get-ChildItem -LiteralPath $dir -Filter '*-choice.csv' | Sort-Object Name)
if ($files.Count -eq 0) { throw 'No choice sample files found.' }
function Median([double[]]$values) {
    $sorted = @($values | Sort-Object)
    if ($sorted.Count -eq 0) { throw 'No values for median.' }
    $i = [int][math]::Floor($sorted.Count / 2)
    if ($sorted.Count % 2) { return $sorted[$i] }
    return ($sorted[$i-1] + $sorted[$i]) / 2.0
}
$results = foreach ($file in $files) {
    $s = @(Import-Csv -LiteralPath $file.FullName)
    if ($s.Count -ne 9) { throw "Expected nine choice trials in $($file.Name)." }
    $first = $s[0]
    if (@($s | Where-Object { $_.selected -ne $first.selected -or $_.run -ne $first.run -or $_.n -ne $first.n -or $_.k -ne $first.k -or $_.batch -ne $first.batch -or $_.group -ne $first.group }).Count -ne 0) { throw "Inconsistent metadata: $($file.Name)." }
    $ratios = @($s | ForEach-Object { [double]$_.direct_over_paired })
    $ratio = Median $ratios
    $winner = if ($ratio -lt 1) { 'direct' } else { 'grouped_paired' }
    $loss = if ($first.selected -eq $winner) { 0.0 } elseif ($first.selected -eq 'direct') { ($ratio - 1.0)*100.0 } elseif ($first.selected -eq 'grouped_paired') { (1.0/$ratio - 1.0)*100.0 } else { throw 'Unexpected selected kernel.' }
    $directMedian = Median @($s | ForEach-Object { [double]$_.direct_ms })
    $pairedMedian = Median @($s | ForEach-Object { [double]$_.paired_ms })
    # Derived from the actual holdout timings of the selected candidate; it is
    # not an additional timing of the run_auto() dispatcher.
    $selectedMedian = if ($first.selected -eq 'direct') { $directMedian } else { $pairedMedian }
    [pscustomobject]@{
        file=$file.Name; n=$first.n; k=$first.k; batch=$first.batch; group=$first.group; run=$first.run
        direct_median_ms=$directMedian; paired_median_ms=$pairedMedian; selected_holdout_median_ms=$selectedMedian
        median_direct_over_paired=$ratio; reference_winner=$winner; selected=$first.selected
        agrees=($first.selected -eq $winner); observed_loss_pct=[math]::Max(0.0,$loss)
        near_tie_1pct=([math]::Abs($ratio-1.0) -le 0.01); autotune_wall_ms=[double]$first.autotune_wall_ms
        min_ratio=($ratios | Measure-Object -Minimum).Minimum; max_ratio=($ratios | Measure-Object -Maximum).Maximum
    }
}
$out = Join-Path $dir ('choice-analysis-' + (Get-Date -Format 'yyyyMMdd-HHmmss-fffffff') + '.csv')
$results | Export-Csv -LiteralPath $out -NoTypeInformation -Encoding utf8
$agreed = @($results | Where-Object { $_.agrees }).Count
$ties = @($results | Where-Object { $_.near_tie_1pct }).Count
$maxLoss = ($results | Measure-Object -Property observed_loss_pct -Maximum).Maximum
$medianTune = Median @($results | ForEach-Object { $_.autotune_wall_ms })
Write-Host "Runs=$($results.Count) agreement=$agreed near_tie_1pct=$ties max_observed_loss_pct=$maxLoss median_autotune_wall_ms=$medianTune"
Write-Host "Analysis: $out"
Write-Host 'Reference is the median ratio of nine subsequent paired measurements, not an absolute ground truth; clocks and GPU load are not controlled.'
