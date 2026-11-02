# Wide backbone, the full commercial base recipe (~9 h after the GPU frees up):
#   1. DINOv2 distillation of the wide stem + core, 20 epochs   -> runs\pretrain\s_wide_distill.pt (~1 h)
#   2. COCO surveillance classes, 72 epochs (configs\train\coco_wide.yaml) -> runs\sarbon\s_wide_distill (~8 h)
# The RailSafe fine-tune is started separately once its data recipe is chosen.
. "$PSScriptRoot\common.ps1"
$log = Join-Path $Repo "runs\jobs\wide_pipeline.queue.log"
New-Item -ItemType Directory -Force (Split-Path $log) | Out-Null
WaitGpu $log
$arch = @("--widths", "48,96,192", "--depths", "2,3", "--c16", "384")
Log $log "pretrain start"
$code = RunSarbon "pretrain" (@("--coco", "C:\data\coco", "--size", "s", "--epochs", "20", "--tag", "_wide") + $arch) `
    (Join-Path $Repo "runs\jobs\wide_pretrain.log")
Log $log "pretrain exit $code"
if ($code -ne 0) { Log $log "done"; exit $code }
Log $log "coco start"
$code = RunSarbon "train" @("--config", "configs\train\coco_wide.yaml") (Join-Path $Repo "runs\jobs\wide_coco.log")
Log $log "coco exit $code"
Log $log "done"
