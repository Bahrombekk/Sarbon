# Product detector: RailSafe classes from commercial COCO + own crossing data (partial labels).
. "$PSScriptRoot\common.ps1"
$log = Join-Path $Repo "runs\railsafe\queue.log"
New-Item -ItemType Directory -Force (Split-Path $log) | Out-Null
WaitGpu $log
Log $log "railsafe fine-tune start"
$code = RunSarbon "train" @("--config", "configs\train\railsafe_s.yaml") (Join-Path $Repo "runs\railsafe\train_s_railsafe.log")
Log $log "railsafe fine-tune exit $code"
Log $log "done"
