# GPU queue after the current Sarbon-S run:
#   1) DINOv2 -> Sarbon distillation pretraining (label-free, commercial images)
#   2) Sarbon-S with --init (same recipe as the plain run -> measures the pretraining gain)
#   3) neck fusion ablation (concat / weighted / gated, 24 epochs each)
# Start detached:
#   Start-Process powershell -WindowStyle Hidden -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass','-File','D:\Sarbon\scripts\run_queue2.ps1'
param([string]$Coco = "C:\data\coco", [string]$Runs = "D:\Sarbon\runs")
$py = "D:\Sarbon\.venv\Scripts\python.exe"
$exp = "D:\Sarbon\experiments"
$log = "$Runs\queue2.log"
New-Item -ItemType Directory -Force "$Runs\pretrain", "$Runs\sarbon", "$Runs\fusion" | Out-Null
function Log($m) { Add-Content $log "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $m" }
function Busy {
    [bool](Get-CimInstance Win32_Process | Where-Object {
        ($_.CommandLine -like "*train_sarbon.py*" -and $_.CommandLine -notlike "*multiprocessing*") -or
        $_.CommandLine -like "*Collector*train.py*" })
}
Log "queue started (pid $PID); waiting for the GPU"
while (Busy) { Start-Sleep 120 }

Log "pretrain start"
& $py -u "$exp\pretrain_distill.py" --coco $Coco --size s --epochs 20 --batch 32 --workers 12 `
    --out "$Runs\pretrain" *> "$Runs\pretrain\pretrain_s.log"
Log "pretrain exit $LASTEXITCODE"

if (Test-Path "$Runs\pretrain\s_distill.pt") {
    Log "sarbon-s init start"
    & $py -u "$exp\train_sarbon.py" --coco $Coco --size s --epochs 72 --batch 16 --workers 12 `
        --no-halo-zero --init "$Runs\pretrain\s_distill.pt" --tag "_distill" --out "$Runs\sarbon" `
        *> "$Runs\sarbon\train_s_distill.log"
    Log "sarbon-s init exit $LASTEXITCODE"
}

foreach ($f in "concat", "weighted", "gated") {
    Log "fusion $f start"
    & $py -u "$exp\train_sarbon.py" --coco $Coco --size s --epochs 24 --no-aug-epochs 2 --batch 16 `
        --workers 12 --no-halo-zero --fusion $f --tag "_fusion_$f" --out "$Runs\fusion" *> "$Runs\fusion\$f.log"
    Log "fusion $f exit $LASTEXITCODE"
}
Log "done"
