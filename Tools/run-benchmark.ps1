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

    Unity is found as described in common.ps1 (MOSYNTH_UNITY_EXE, else the Unity Hub default).

    -Graphics keeps batchmode but drops -nographics. A config with recordVideo on needs it: the
    recorder renders through a camera, and -nographics gives it no device to render with.

    -Manifest runs the sweep through Reproduction.ReproductionPipeline.Benchmark instead, so every
    method that loads a per-seed checkpoint runs once per training seed the manifest lists, in place
    of the seeds the method lists itself.

.EXAMPLE
    .\run-benchmark.ps1
    .\run-benchmark.ps1 -Config "Assets/Benchmarks/Ablation.asset" -Output "Benchmarks/ablation"
    .\run-benchmark.ps1 -Config "Assets/Benchmarks/ShowcaseBenchmark.asset" -Graphics
    .\run-benchmark.ps1 -Config "Assets/Benchmarks/Ablation.asset" -Manifest "Assets/Benchmarks/PaperReproduction.asset"
#>
param(
    [string]$Config = "Assets/Benchmarks/DefaultBenchmark.asset",
    [string]$Output,
    [string]$UnityExe,
    [string]$Manifest,
    [switch]$Visible,
    [switch]$Graphics
)

. "$PSScriptRoot\common.ps1"

$projectPath = Get-MoSynthProjectPath
$UnityExe = Get-MoSynthUnityExe -Override $UnityExe

if (-not $Output) {
    $Output = "Benchmarks/" + (Get-Date -Format yyyyMMdd_HHmmss)
}

$resolvedOutput = $Output
if (-not [System.IO.Path]::IsPathRooted($resolvedOutput)) {
    $resolvedOutput = Join-Path $projectPath $resolvedOutput
}
New-Item -ItemType Directory -Force -Path $resolvedOutput | Out-Null

$logFile = Join-Path $resolvedOutput "unity.log"
$resultsCsv = Join-Path $resolvedOutput "results.csv"

if ($Visible) {
    Write-Host "Visible mode: the sweep will not quit Unity when it finishes; close the Editor to release this script."
}

# Deliberately no -quit: batchmode Unity stays alive after -executeMethod returns, and the sweep
# needs that because it runs asynchronously across play-mode frames. SynthesisBenchmarkCli.Run only
# starts it; SynthesisBenchmarkDriver calls EditorApplication.Exit once the report is written.
$entryPoint = "AnimationTools.Editor.SynthesisBenchmarkCli.Run"
$sweepArguments = @("-benchmarkConfig", $Config, "-benchmarkOutput", $Output)
if ($Manifest) {
    $entryPoint = "Reproduction.ReproductionPipeline.Benchmark"
    $sweepArguments += @("-reproManifest", $Manifest)
}

$exitCode = Invoke-MoSynthUnity -UnityExe $UnityExe -LogFile $logFile -Visible:$Visible -Graphics:$Graphics -Arguments (
    @("-executeMethod", $entryPoint) + $sweepArguments)

Write-Host "Unity exited with code $exitCode"
Write-Host "Results CSV: $resultsCsv"
exit $exitCode
