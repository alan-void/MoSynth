<#
.SYNOPSIS
    Shared path discovery for the MoSynth tool scripts. Dot-source it: . "$PSScriptRoot\common.ps1"

.DESCRIPTION
    Every machine-specific path resolves from an environment variable first and a conventional
    default second, so no script carries one machine's layout:
      MOSYNTH_UNITY_EXE     Unity.exe; default is the Unity Hub install of ProjectVersion.txt's editor
      MOSYNTH_PYTHON_VENV   venv whose python runs the Python/ tools; default is python on PATH
      MOSYNTH_BLENDER_EXE   read by Tools/Retargeting/run_batch_all.py, not here
#>

function Get-MoSynthProjectPath {
    return (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
}

function Get-MoSynthUnityVersion {
    $versionFile = Join-Path (Get-MoSynthProjectPath) "ProjectSettings\ProjectVersion.txt"
    $line = Get-Content -LiteralPath $versionFile | Where-Object { $_ -like "m_EditorVersion:*" } | Select-Object -First 1
    return ($line -split ":", 2)[1].Trim()
}

function Get-MoSynthUnityExe {
    param([string]$Override)

    $candidates = @($Override, $env:MOSYNTH_UNITY_EXE,
        "C:\Program Files\Unity\Hub\Editor\$(Get-MoSynthUnityVersion)\Editor\Unity.exe")
    foreach ($candidate in $candidates) {
        if ($candidate -and (Test-Path -LiteralPath $candidate)) { return $candidate }
    }
    throw "Unity $(Get-MoSynthUnityVersion) not found. Install it through Unity Hub, or set MOSYNTH_UNITY_EXE / pass -UnityExe."
}

function Get-MoSynthPythonExe {
    if ($env:MOSYNTH_PYTHON_VENV) {
        $venvPython = Join-Path $env:MOSYNTH_PYTHON_VENV "Scripts\python.exe"
        if (Test-Path -LiteralPath $venvPython) { return $venvPython }
        throw "MOSYNTH_PYTHON_VENV is set to '$env:MOSYNTH_PYTHON_VENV' but it holds no Scripts\python.exe."
    }
    $onPath = Get-Command python -ErrorAction SilentlyContinue
    if ($onPath) { return $onPath.Source }
    throw "No Python found. Set MOSYNTH_PYTHON_VENV to the venv made from Python/requirements.txt."
}

# Runs one synchronous Unity batchmode step. Returns Unity's exit code and tails the log on failure.
function Invoke-MoSynthUnity {
    param(
        [Parameter(Mandatory)][string]$UnityExe,
        [Parameter(Mandatory)][string]$LogFile,
        [Parameter(Mandatory)][string[]]$Arguments,
        [switch]$Visible
    )

    $unityArgs = @("-projectPath", (Get-MoSynthProjectPath), "-logFile", $LogFile) + $Arguments
    if (-not $Visible) { $unityArgs += @("-batchmode", "-nographics") }
    # Start-Process joins arguments with spaces and does not quote them.
    $unityArgs = $unityArgs | ForEach-Object { if ($_ -match "\s") { "`"$_`"" } else { $_ } }

    $process = Start-Process -FilePath $UnityExe -ArgumentList $unityArgs -PassThru -Wait -NoNewWindow
    if ($process.ExitCode -ne 0) {
        Write-Host "Unity exited with code $($process.ExitCode). Tail of $LogFile :"
        if (Test-Path -LiteralPath $LogFile) { Get-Content -LiteralPath $LogFile -Tail 40 | Out-Host }
    }
    return $process.ExitCode
}
