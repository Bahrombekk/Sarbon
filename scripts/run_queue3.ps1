# GPU queue after the full-COCO benchmark run:
#   RailSafe fine-tune: commercial COCO (RailSafe classes) + own crossing data (partial labels),
#   starting from the commercial Sarbon-S checkpoint with its car/truck heads remapped.
# Start outside the calling app's process tree (see run_bench.ps1):
#   Invoke-CimMethod Win32_Process -MethodName Create -Arguments @{CommandLine='powershell -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File D:\Sarbon\scripts\run_queue3.ps1'}
param([string]$Coco = "C:\data\coco", [string]$Own = "C:\data\pereezd_split", [string]$Runs = "D:\Sarbon\runs")
$py = "D:\Sarbon\.venv\Scripts\python.exe"
$exp = "D:\Sarbon\experiments"
$log = "$Runs\queue3.log"
New-Item -ItemType Directory -Force "$Runs\railsafe" | Out-Null
function Log($m) { Add-Content $log "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $m" }
function Busy {
    [bool](Get-CimInstance Win32_Process | Where-Object {
        ($_.CommandLine -like "*train_sarbon.py*" -and $_.CommandLine -notlike "*multiprocessing*") -or
        $_.CommandLine -like "*Collector*train.py*" })
}
Log "queue started (pid $PID); waiting for the GPU"
while (Busy) { Start-Sleep 120 }

Log "railsafe fine-tune start"
& $py -u "$exp\train_sarbon.py" --coco $Coco --taxonomy railsafe --yolo $Own `
    --init "$Runs\sarbon\s_distill\best.pt" --init-classes surveillance `
    --size s --epochs 24 --no-aug-epochs 2 --lr 2e-4 --batch 16 --workers 12 --no-halo-zero --eval-every 2 `
    --tag "_railsafe" --out "$Runs\railsafe" --resume *>> "$Runs\railsafe\train_s_railsafe.log"
Log "railsafe fine-tune exit $LASTEXITCODE"
Log "done"
