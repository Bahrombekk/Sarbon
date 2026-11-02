# Neck fusion ablation (concat / weighted / gated), Sarbon-S, 24 epochs each.
# Waits until no other Sarbon training or the Collector training is running.
# Start detached:
#   Start-Process powershell -WindowStyle Hidden -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass','-File','D:\Sarbon\scripts\run_fusion_ablation.ps1'
param([string]$Coco = "C:\data\coco", [string]$Runs = "D:\Sarbon\runs\fusion", [int]$Epochs = 24)
$py = "D:\Sarbon\.venv\Scripts\python.exe"
New-Item -ItemType Directory -Force $Runs | Out-Null
$log = "$Runs\queue.log"
function Log($m) { Add-Content $log "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $m" }
function Busy {
    [bool](Get-CimInstance Win32_Process | Where-Object {
        ($_.CommandLine -like "*train_sarbon.py*" -and $_.CommandLine -notlike "*multiprocessing*") -or
        $_.CommandLine -like "*Collector*train.py*" })
}
Log "queue started (pid $PID); waiting for the GPU"
while (Busy) { Start-Sleep 120 }
foreach ($f in "concat", "weighted", "gated") {
    Log "fusion $f start"
    & $py -u D:\Sarbon\experiments\train_sarbon.py --coco $Coco --size s --epochs $Epochs --no-aug-epochs 2 `
        --batch 16 --workers 12 --no-halo-zero --fusion $f --tag "_fusion_$f" --out $Runs *> "$Runs\$f.log"
    Log "fusion $f exit $LASTEXITCODE"
}
Log "done"
