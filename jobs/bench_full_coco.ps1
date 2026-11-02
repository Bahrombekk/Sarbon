# Full-COCO benchmark run (research only, never product weights). Safe to start again after a
# crash or reboot: the config sets resume, so it continues from the last epoch checkpoint.
. "$PSScriptRoot\common.ps1"
$log = Join-Path $Repo "runs\bench.log"
Log $log "bench start (pid $PID)"
$code = RunSarbon "train" @("--config", "configs\train\coco_bench_s.yaml") (Join-Path $Repo "runs\sarbon\train_s_bench_distill.log")
Log $log "bench exit $code"
