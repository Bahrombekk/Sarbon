# Light network ("G", configs\train\coco_light_person.yaml), the full chain behind v0.3 (~15-18 h):
#   1. DINOv2 distillation of the light stem + core, 20 epochs -> runs\pretrain\s_light_distill.pt
#   2. general base: COCO + own people, 50 epochs (coco_light_person.yaml) -> runs\sarbon\s_light_person
#   3. pereezd like the RailSafe YOLO26n, 1088 (railsafe_light_1088.yaml) -> runs\railsafe\s_railsafe_light_1088
. "$PSScriptRoot\common.ps1"
$logs = Join-Path $Repo "runs\jobs"
$log = "$logs\light_pipeline.queue.log"
New-Item -ItemType Directory -Force $logs | Out-Null
WaitGpu $log
$steps = @(
    @("pretrain", "light_pretrain", @("--coco", "C:\data\coco", "--size", "s", "--epochs", "20", "--tag", "_light",
        "--widths", "32,64,128", "--depths", "1,1", "--c16", "256", "--hidden", "192", "--heads", "6", "--ffn", "768")),
    @("train", "light_person", @("--config", "configs\train\coco_light_person.yaml")),
    @("train", "railsafe_light_1088", @("--config", "configs\train\railsafe_light_1088.yaml"))
)
foreach ($s in $steps) {
    Log $log "$($s[1]) start"
    $code = RunSarbon $s[0] $s[2] "$logs\$($s[1]).log"
    Log $log "$($s[1]) exit $code"
    if ($code -ne 0) { break }
}
Log $log "done"
