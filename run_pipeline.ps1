# run_pipeline.ps1
# One-command pipeline runner for Windows PowerShell.
#
# Usage:
#   pwsh run_pipeline.ps1                          # most severe crash
#   pwsh run_pipeline.ps1 -EventId synshrp2_90000001
#   pwsh run_pipeline.ps1 -Random                  # random event (testing)
#   pwsh run_pipeline.ps1 -Random -IncludeNormal -Seed 42
#   pwsh run_pipeline.ps1 -SkipBuild               # re-plot without rebuilding

param(
    [string] $EventId          = "",
    [switch] $SkipBuild,
    [switch] $Random,
    [switch] $IncludeNormal,
    [Nullable[int]] $Seed      = $null,
    [string] $DataDir          = ".\data",
    [string] $OutputDir        = ".\output",
    [string] $ScriptsDir       = ".\scripts"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Write-Step  { param($msg) Write-Host "`n>>  $msg" -ForegroundColor Cyan  }
function Write-OK    { param($msg) Write-Host "OK  $msg"   -ForegroundColor Green }
function Write-Warn  { param($msg) Write-Host "!!  $msg"   -ForegroundColor Yellow}
function Write-Fatal { param($msg) Write-Host "XX  $msg"   -ForegroundColor Red; exit 1 }

Write-Step "Checking Python"
try { $v = python --version 2>&1; Write-OK $v }
catch { Write-Fatal "Python not found. Install from https://python.org" }

Write-Step "Installing dependencies"
python -m pip install numpy pandas scipy plotly --quiet --disable-pip-version-check
Write-OK "Dependencies ready"

if (-not $SkipBuild) {
    Write-Step "Building datasets (crashes / normal / validation)"
    New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
    python "$ScriptsDir\build_crash_dataset.py" `
        --synshrp2_dir "$DataDir\synshrp2" `
        --ciss_dir     "$DataDir\ciss"     `
        --har_dir      "$DataDir\har"      `
        --output_dir   $OutputDir
    Write-OK "Output written to $OutputDir"
} else {
    Write-Warn "Skipping dataset build (-SkipBuild)"
}

Write-Step "Generating visualization"
$vizArgs = @("$ScriptsDir\visualize_crash.py","--output_dir",$OutputDir)
if ($EventId -ne "") { $vizArgs += @("--event_id", $EventId) }
if ($Random) { $vizArgs += "--random" }
if ($IncludeNormal) { $vizArgs += "--include_normal" }
if ($null -ne $Seed) { $vizArgs += @("--seed", $Seed) }
python @vizArgs

Write-Step "Opening output"
$f = Get-ChildItem "$OutputDir\plot_*.html" -ErrorAction SilentlyContinue |
     Sort-Object LastWriteTime -Descending | Select-Object -First 1
if ($f) { Write-OK "Opening $($f.FullName)"; Start-Process $f.FullName }

Write-Host "`nDone. Output folder: $OutputDir`n" -ForegroundColor Green
