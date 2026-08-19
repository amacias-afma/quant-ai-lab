# Overnight run - H5 capacity study, staged with an executable gate.
#
#   .\run_overnight.ps1
#
# Stage A (VAL only, ~1.5 h, ZERO test-set evaluations)
#     Expanded capacity pilot: 5 tickers x 20 seeds. Tests whether unanchored seed
#     dispersion grows with model capacity.
#
# Stage B (full panel with TEST, ~5-7 h) - RUNS ONLY IF STAGE A PASSES ITS GATE.
#
# Exit codes from the pilot are a scientific signal, NOT a generic failure flag:
#     0   gate passed        -> run stage B
#     10  premise is false   -> skip stage B, report the null (this IS a result)
#     any other value        -> the script crashed, nothing was concluded
# Conflating the last two once let a UnicodeEncodeError print itself as "GATE FAILED".
#
# Everything is logged to logs\ with timestamps.

$ErrorActionPreference = "Continue"

# Force UTF-8 for Python I/O: the Windows console defaults to cp1252 and will otherwise
# crash on any non-ASCII character a script happens to print.
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

$stamp  = Get-Date -Format "yyyyMMdd_HHmmss"
$logDir = "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$logA = Join-Path $logDir "stageA_pilot_$stamp.log"
$logB = Join-Path $logDir "stageB_panel_$stamp.log"

Write-Host "=========================================================="
Write-Host " STAGE A - capacity pilot (VALIDATION only, no TEST reads)"
Write-Host " started $(Get-Date -Format 'HH:mm:ss')  ->  $logA"
Write-Host "=========================================================="

python run_capacity_pilot.py `
    --tickers "^GSPC,NVDA,BTC-USD,SQM,CL=F" `
    --alpha 0.01 `
    --seeds 20 `
    --weights 0,0.5 `
    --outdir outputs/pilot_capacity_wide 2>&1 | Tee-Object -FilePath $logA

$gate = $LASTEXITCODE

if ($gate -eq 10) {
    Write-Host ""
    Write-Host "##########################################################"
    Write-Host "# H5a GATE FAILED ON THE EVIDENCE (exit 10)."
    Write-Host "# Bigger models are NOT more seed-unstable, so the anchor"
    Write-Host "# has nothing to stabilise. Stage B deliberately SKIPPED."
    Write-Host "# No test-set evaluations were spent. This is a RESULT:"
    Write-Host "# report the null."
    Write-Host "##########################################################"
    exit 10
}

if ($gate -ne 0) {
    Write-Host ""
    Write-Host "##########################################################"
    Write-Host "# STAGE A CRASHED (exit $gate) - this is NOT a finding."
    Write-Host "# Nothing was concluded about H5a. Check the traceback in:"
    Write-Host "#   $logA"
    Write-Host "# Stage B skipped. Fix the error and re-run."
    Write-Host "##########################################################"
    exit $gate
}

Write-Host ""
Write-Host "=========================================================="
Write-Host " STAGE A gate PASSED. Starting STAGE B (panel, reads TEST)"
Write-Host " started $(Get-Date -Format 'HH:mm:ss')  ->  $logB"
Write-Host "=========================================================="

# Matches stage2b exactly (same grid, same argmin rule) and adds QuantileMLP, so the
# linear-vs-MLP comparison is controlled: one variable changes, nothing else.
# alpha = 0.01 only: H5's mechanism is thin-tail instability, and this keeps the run
# inside one night. alpha = 0.05 is deferred, not dropped.
python run_batch_anchored.py `
    --models SimpleQuantileNeuron,QuantileMLP `
    --hidden-size 64 --num-layers 3 `
    --weights 0,0.1,0.25,0.5,1,2 `
    --selection-rule argmin `
    --alphas 0.01 `
    --seeds 10 --val-seeds 3 `
    --outdir outputs/stageB_mlp 2>&1 | Tee-Object -FilePath $logB

$rcB = $LASTEXITCODE

Write-Host ""
Write-Host "=========================================================="
Write-Host " DONE $(Get-Date -Format 'HH:mm:ss')   stage B exit=$rcB"
Write-Host "   Stage A: outputs\pilot_capacity_wide\   ($logA)"
Write-Host "   Stage B: outputs\stageB_mlp\            ($logB)"
Write-Host "=========================================================="
