# Accuracy ablations, 24 epochs from scratch (baseline concat: AP 30.2, AP_small 11.0):
#   nwd1 : NWD term in the Hungarian matching cost (small objects)
#   wide : backbone x2 (stem 48/96/192, depths 2/3, stride-16 384 ch) -- is capacity the bottleneck?
. "$PSScriptRoot\common.ps1"
$log = Join-Path $Repo "runs\ablate2\queue.log"
New-Item -ItemType Directory -Force (Split-Path $log) | Out-Null
WaitGpu $log
$jobs = [ordered]@{
    "nwd1" = @("--nwd", "1");
    "wide" = @("--widths", "48,96,192", "--depths", "2,3", "--c16", "384")
}
foreach ($name in $jobs.Keys) {
    Log $log "$name start"
    $code = RunSarbon "train" (@("--config", "configs\train\ablation_24ep.yaml", "--tag", "_$name") + $jobs[$name]) `
        (Join-Path $Repo "runs\ablate2\$name.log")
    Log $log "$name exit $code"
}
Log $log "done"
