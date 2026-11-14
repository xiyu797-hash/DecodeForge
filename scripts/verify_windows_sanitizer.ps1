param(
    [Parameter(Mandatory=$true)][string]$BuildDirectory,
    [string]$CudaToolkitRoot = 'E:\DecodeForge-CUDA-13.4',
    [string]$SanitizerPath = '',
    [switch]$DryRun
)
$ErrorActionPreference = 'Stop'
$root = (Resolve-Path -LiteralPath (Split-Path -Parent $PSScriptRoot)).ProviderPath.TrimEnd('\')
$build = (Resolve-Path -LiteralPath $BuildDirectory -ErrorAction Stop).ProviderPath.TrimEnd('\')
$prefix = $root + '\'
if (-not $build.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) {
    throw 'BuildDirectory must be inside this project so the temporary alias can cover the test binary.'
}
if (-not $SanitizerPath) {
    $bundled = Join-Path $root 'build\tools\cuda-sanitizer-13.4.3\cuda_sanitizer_api-windows-x86_64-13.4.92-archive\compute-sanitizer\compute-sanitizer.exe'
    $SanitizerPath = if (Test-Path -LiteralPath $bundled -PathType Leaf) { $bundled } else { Join-Path $CudaToolkitRoot 'compute-sanitizer\compute-sanitizer.exe' }
}
$san = (Resolve-Path -LiteralPath $SanitizerPath -ErrorAction Stop).ProviderPath
foreach ($test in @('decodeforge_gpu_tests','decodeforge_ops_gpu_tests')) {
    if (-not (Test-Path -LiteralPath (Join-Path $build "$test.exe") -PathType Leaf)) { throw "Missing test binary: $test in $build" }
}
if (-not $san.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase) -and $san -match '[^\x20-\x7e]|\+') {
    throw 'Sanitizer is outside the project and its path is not ASCII-safe; pass an ASCII-safe -SanitizerPath.'
}

# QueryDosDeviceW avoids locale-dependent decoding of subst.exe output. Select
# only unused letters and remove only the alias that still points to our root.
if (-not ('DecodeForgeDosDevice' -as [type])) {
    Add-Type -TypeDefinition @'
using System;
using System.Text;
using System.Runtime.InteropServices;
public static class DecodeForgeDosDevice {
  [DllImport("kernel32.dll", CharSet=CharSet.Unicode, SetLastError=true)]
  public static extern uint QueryDosDevice(string name, StringBuilder target, int capacity);
}
'@
}
function Get-DosTarget([string]$letter) {
    $buffer = [Text.StringBuilder]::new(32768)
    $n = [DecodeForgeDosDevice]::QueryDosDevice("${letter}:", $buffer, $buffer.Capacity)
    if ($n -eq 0) {
        $errorCode = [Runtime.InteropServices.Marshal]::GetLastWin32Error()
        if ($errorCode -eq 2) { return $null }
        throw "QueryDosDevice failed for ${letter}: error $errorCode"
    }
    return $buffer.ToString()
}
$letter = @('X','Y','Z','W','V','U','T','S','R','Q','P','O','N','M','L','K','J','I','H','G','F','D') |
    Where-Object { $null -eq (Get-DosTarget $_) } | Select-Object -First 1
if (-not $letter) { throw 'No unused drive letter available for temporary project alias.' }
$expected = '\??\' + $root
$out = Join-Path $root ("validation\sanitizer-" + (Get-Date -Format 'yyyyMMdd-HHmmss-ffff') + '-' + [guid]::NewGuid().ToString('N').Substring(0,8))
New-Item -ItemType Directory -Path $out -ErrorAction Stop | Out-Null
$summary = Join-Path $out 'summary.csv'
'sanitizer,test,exit_code,elapsed_seconds,log' | Set-Content -LiteralPath $summary -Encoding ascii
$oldPath = $env:PATH
$created = $false
try {
    & subst.exe "${letter}:" $root
    if ($LASTEXITCODE -ne 0) { throw "Failed to create temporary drive ${letter}:" }
    $created = $true
    if ((Get-DosTarget $letter) -ne $expected) { throw 'Temporary mapping does not match the project root.' }
    $alias = "${letter}:\"
    $aliasBuild = Join-Path $alias $build.Substring($prefix.Length)
    $aliasSan = if ($san.StartsWith($prefix,[StringComparison]::OrdinalIgnoreCase)) { Join-Path $alias $san.Substring($prefix.Length) } else { $san }
    $env:PATH = "$(Join-Path $CudaToolkitRoot 'bin\x64');$(Join-Path $CudaToolkitRoot 'bin');$oldPath"
    if ($DryRun) {
        Write-Host "[DRY RUN] Alias and target verified; no sanitizer executed. Logs: $out"
    } else {
        $savedPreference = $ErrorActionPreference
        try {
            $ErrorActionPreference = 'Continue'
            & $aliasSan --version 2>&1 | Tee-Object -FilePath (Join-Path $out 'version.log')
            $versionCode = $LASTEXITCODE
        } finally { $ErrorActionPreference = $savedPreference }
        if ($versionCode -ne 0) { throw "Sanitizer version check failed: $versionCode" }
        if (-not (Select-String -LiteralPath (Join-Path $out 'version.log') -Pattern 'NVIDIA.*Compute Sanitizer' -Quiet)) {
            throw 'Tool version output does not identify NVIDIA Compute Sanitizer.'
        }
        foreach ($tool in @('memcheck','racecheck')) {
            foreach ($test in @('decodeforge_gpu_tests','decodeforge_ops_gpu_tests')) {
                $log = Join-Path $out "$tool-$test.log"
                Write-Host "[SANITIZER] $tool / $test"
                $clock = [Diagnostics.Stopwatch]::StartNew()
                $savedPreference = $ErrorActionPreference
                try {
                    $ErrorActionPreference = 'Continue'
                    & $aliasSan --tool $tool --error-exitcode 1 (Join-Path $aliasBuild "$test.exe") 2>&1 | Tee-Object -FilePath $log
                    $code = $LASTEXITCODE
                } finally { $ErrorActionPreference = $savedPreference }
                $clock.Stop()
                "$tool,$test,$code,$([Math]::Round($clock.Elapsed.TotalSeconds,3)),$(Split-Path -Leaf $log)" | Add-Content -LiteralPath $summary -Encoding ascii
                if ($code -ne 0) { throw "$tool / $test failed (exit=$code); log=$log" }
                $expectedSummary = if ($tool -eq 'memcheck') { 'ERROR SUMMARY:\s*0 errors' } else { 'RACECHECK SUMMARY:\s*0 hazards.*\(0 errors,\s*0 warnings\)' }
                if (-not (Select-String -LiteralPath $log -Pattern $expectedSummary -Quiet)) { throw "$tool / $test did not report a clean sanitizer summary; log=$log" }
            }
        }
        Write-Host "[SANITIZER] All four runs completed with exit=0. Logs: $out"
    }
} finally {
    $env:PATH = $oldPath
    if ($created) {
        if ((Get-DosTarget $letter) -eq $expected) {
            & subst.exe "${letter}:" /D
            if ($LASTEXITCODE -ne 0) { throw "Failed to remove temporary drive ${letter}:" }
        } else { Write-Warning "Temporary drive ${letter}: no longer points to this project; not removing it." }
    }
}
