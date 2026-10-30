# After queue3 (RailSafe fine-tune): YOLO11s baseline on the same split, then the comparison.
# Ultralytics runs in the Collector env (AGPL, measurement only); Sarbon in its own env.
# Start outside the calling app's process tree (see run_bench.ps1):
#   Invoke-CimMethod Win32_Process -MethodName Create -Arguments @{CommandLine='powershell -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File D:\Sarbon\scripts\run_queue4.ps1'}
param([string]$Split = "C:\data\pereezd_split", [string]$Runs = "D:\Sarbon\runs",
      [string]$YoloPy = "C:\Users\User\Desktop\Collector\venv310\Scripts\python.exe")
$py = "D:\Sarbon\.venv\Scripts\python.exe"
$log = "$Runs\queue4.log"
$yolo = "$Runs\yolo"
New-Item -ItemType Directory -Force $yolo | Out-Null
function Log($m) { Add-Content $log "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $m" }
Log "queue started (pid $PID); waiting for queue3 to finish"
while (-not (Select-String -Path "$Runs\queue3.log" -Pattern " done$" -Quiet -ErrorAction SilentlyContinue)) { Start-Sleep 120 }

Set-Location $yolo  # Ultralytics downloads yolo11s.pt into the working folder
Log "yolo11s train start"
& $YoloPy -u D:\Sarbon\scripts\yolo_own.py train --split $Split --out $yolo --epochs 50 --batch 16 --workers 8 `
    *> "$yolo\train_yolo11s.log"
Log "yolo11s train exit $LASTEXITCODE"
$w = "$yolo\yolo11s_own\weights\best.pt"
& $YoloPy -u D:\Sarbon\scripts\yolo_own.py dump --weights $w --split $Split --json "$yolo\yolo11s_val.json" *> "$yolo\dump.log"
& $YoloPy -u D:\Sarbon\scripts\yolo_own.py speed --weights $w --json "$yolo\yolo11s_speed.json" *> "$yolo\speed.log"
Log "yolo dump/speed exit $LASTEXITCODE"

Set-Location D:\Sarbon
& $py -u bench\compare_own.py --split $Split --sarbon "$Runs\railsafe\s_railsafe\best.pt" `
    --yolo-json "$yolo\yolo11s_val.json" --yolo-speed "$yolo\yolo11s_speed.json" --out "$Runs\compare_own.json" `
    *> "$Runs\compare_own.log"
Log "compare exit $LASTEXITCODE"
Log "done"
