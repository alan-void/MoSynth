<#
.SYNOPSIS
    Launches a MoSynth synthesis benchmark sweep in Unity and reports where the results landed.

.DESCRIPTION
    Headless by default. Unity must not already have this project open -- only one Editor may hold a
    project at a time, so close it first.

    -Visible launches a normal Editor window instead. In that mode the sweep does NOT quit Unity when
    it finishes (a developer wants their Editor back), so this script blocks until you close it. If
    the Editor is already open, use the MoSynth > Benchmark > Run Sweep menu item instead of this
    script.

.EXAMPLE
    .un-benchmark.ps1
    .un-benchmark.ps1 -Config "Assets/Benchmarks/Ablation.asset" -Output "Benchmarks/ablation"
#>
param(
    [string]$Config = "Assets/Benchmarks/DefaultBenchmark.asset",
    [string]$Output,
    [string]$ProjectPath = "E:\UnityProjects\MoSynth",
    [string]$UnityExe = "C:\Program Files\Unity\Hub\Editor\6000.4.4f1\Editor\Unity.exe",
    [switch]$Visible
)

if (-not $Output) {
    $Output = "Benchmarks/" + (Get-Date -Format yyyyMMdd_HHmmss)
}

if (-not (Test-Path -LiteralPath $UnityExe)) {
    Write-Error "Unity executable not found at '$UnityExe'. Pass -UnityExe with the correct path."
    exit 1
}

$resolvedOutput = $Output
if (-not [System.IO.Path]::IsPathRooted($resolvedOutput)) {
    $resolvedOutput = Join-Path $ProjectPath $resolvedOutput
}

New-Item -ItemType Directory -Force -Path $resolvedOutput | Out-Null

$logFile = Join-Path $resolvedOutput "unity.log"
$resultsCsv = Join-Path $resolvedOutput "results.csv"

$unityArgs = @(
    "-projectPath", $ProjectPath,
    "-executeMethod", "AnimationTools.Editor.SynthesisBenchmarkCli.Run",
    "-benchmarkConfig", $Config,
    "-benchmarkOutput", $Output,
    "-logFile", $logFile
)

if (-not $Visible) {
    $unityArgs += "-batchmode"
    $unityArgs += "-nographics"
}

# Deliberately no -quit: batchmode Unity stays alive after -executeMethod returns, and the sweep
# needs that because it runs asynchronously across play-mode frames. SynthesisBenchmarkCli.Run only
# starts it; SynthesisBenchmarkDriver calls EditorApplication.Exit once the report is written.

if ($Visible) {
    Write-Host "Visible mode: the sweep will not quit Unity when it finishes; close the Editor to release this script."
}

$process = Start-Process -FilePath $UnityExe -ArgumentList $unityArgs -PassThru -Wait -NoNewWindow
$exitCode = $process.ExitCode

Write-Host "Unity exited with code $exitCode"
Write-Host "Results CSV: $resultsCsv"

if ($exitCode -ne 0) {
    Write-Host "Run failed. Tail of $logFile :"
    if (Test-Path -LiteralPath $logFile) {
        Get-Content -LiteralPath $logFile -Tail 40
    }
    else {
        Write-Host "(log file not found)"
    }
}

exit $exitCode
