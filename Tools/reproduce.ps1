<#
.SYNOPSIS
    Reproduces the paper's results from raw motion capture: fetch, retarget, databases, train,
    benchmark, report.

.DESCRIPTION
    Each stage can be run on its own with -Stage, and each skips work whose output already exists,
    so an interrupted run is resumed by running it again. Unity must be closed for every stage: the
    retarget writes FBX files Unity would otherwise import with fresh GUIDs, and the Unity stages
    need the project to themselves.

      fetch      download the raw datasets and check the Mixamo character (Tools/Data/fetch_datasets.py)
      retarget   rebuild Assets/LFS from the raw data in Blender (Tools/Retargeting/reproduce_all.py)
      databases  validate every config, then build the pose and feature databases
      train      train PFNN, LMM and motion-field models, logging wall-clock time per run
      benchmark  run every sweep listed in the reproduction manifest
      report     summarise the sweeps into tables

    Paths: see common.ps1 for MOSYNTH_UNITY_EXE and MOSYNTH_PYTHON_VENV; Blender is found by
    run_batch_all.py, overridable with MOSYNTH_BLENDER_EXE.

.EXAMPLE
    .\Tools\reproduce.ps1                     # everything
    .\Tools\reproduce.ps1 -Stage train,benchmark,report
#>
param(
    [ValidateSet("fetch", "retarget", "databases", "train", "benchmark", "report", "all")]
    [string[]]$Stage = @("all"),
    [string]$Manifest = "Assets/Benchmarks/PaperReproduction.asset",
    [string]$Output = ("Benchmarks/reproduction_" + (Get-Date -Format yyyyMMdd_HHmmss)),
    [string]$UnityExe
)

$ErrorActionPreference = "Stop"
. "$PSScriptRoot\common.ps1"

$projectPath = Get-MoSynthProjectPath
$python = Get-MoSynthPythonExe
$outputPath = Join-Path $projectPath $Output
New-Item -ItemType Directory -Force -Path $outputPath | Out-Null

function Test-Stage([string]$name) {
    return ($Stage -contains "all") -or ($Stage -contains $name)
}

function Invoke-Checked([string]$label, [scriptblock]$command) {
    Write-Host "=== $label"
    & $command
    if ($LASTEXITCODE -ne 0) { throw "$label failed with exit code $LASTEXITCODE" }
}

function Invoke-PipelineStep([string]$method) {
    $exe = Get-MoSynthUnityExe -Override $UnityExe
    $code = Invoke-MoSynthUnity -UnityExe $exe -LogFile (Join-Path $outputPath "$method.log") -Arguments @(
        "-executeMethod", "Reproduction.ReproductionPipeline.$method",
        "-reproManifest", $Manifest,
        "-reproLog", $Output,
        "-quit")
    if ($code -ne 0) { throw "ReproductionPipeline.$method failed with exit code $code" }
}

# The manifest lists its benchmarks by GUID; each GUID is resolved through the .meta beside the asset.
function Get-ManifestBenchmarks {
    $manifestPath = Join-Path $projectPath $Manifest
    $guids = Select-String -LiteralPath $manifestPath -Pattern "guid: ([0-9a-f]{32})" -AllMatches |
        ForEach-Object { $_.Matches } | ForEach-Object { $_.Groups[1].Value }
    foreach ($guid in $guids) {
        $meta = Get-ChildItem -Path (Join-Path $projectPath "Assets/Benchmarks") -Filter "*.asset.meta" -Recurse |
            Where-Object { Select-String -LiteralPath $_.FullName -Pattern "guid: $guid" -Quiet } |
            Select-Object -First 1
        if ($meta -and ($meta.FullName -notlike "*$(Split-Path $Manifest -Leaf).meta")) {
            $asset = $meta.FullName.Substring(0, $meta.FullName.Length - ".meta".Length)
            $asset.Substring($projectPath.Length + 1).Replace("\", "/")
        }
    }
}

Push-Location $projectPath
try {
    if (Test-Stage "fetch") {
        Invoke-Checked "fetch datasets" { & $python Tools/Data/fetch_datasets.py }
    }
    if (Test-Stage "retarget") {
        Invoke-Checked "retarget" { & $python Tools/Retargeting/reproduce_all.py }
    }
    if (Test-Stage "databases") {
        Invoke-PipelineStep "Validate"
        Invoke-PipelineStep "BuildDatabases"
    }
    if (Test-Stage "train") {
        Invoke-PipelineStep "Train"
    }
    if (Test-Stage "benchmark") {
        foreach ($benchmark in Get-ManifestBenchmarks) {
            $name = [System.IO.Path]::GetFileNameWithoutExtension($benchmark)
            $sweep = "$Output/$name"
            if (Test-Path -LiteralPath (Join-Path $projectPath "$sweep/results.csv")) {
                Write-Host "=== sweep ${name}: already done, skipping"
                continue
            }
            Invoke-Checked "sweep $name" { & "$PSScriptRoot\run-benchmark.ps1" -Config $benchmark -Output $sweep -UnityExe $UnityExe }
        }
    }
    if (Test-Stage "report") {
        $sweeps = Get-ChildItem -LiteralPath $outputPath -Directory |
            Where-Object { Test-Path -LiteralPath (Join-Path $_.FullName "results.csv") } |
            ForEach-Object { $_.FullName }
        if (-not $sweeps) { throw "No finished sweeps under $outputPath" }
        Push-Location (Join-Path $projectPath "Python")
        try {
            foreach ($format in @("markdown", "csv")) {
                $extension = @{ markdown = "md"; csv = "csv" }[$format]
                $table = Join-Path $outputPath "summary.$extension"
                & $python -m benchmark.report @sweeps --format $format | Set-Content -LiteralPath $table -Encoding utf8
                if ($LASTEXITCODE -ne 0) { throw "benchmark.report failed" }
                Write-Host "=== report: $table"
            }
        }
        finally { Pop-Location }
    }
}
finally { Pop-Location }
