# Full-COCO benchmark run (research only, not product weights). Safe to start again after a
# crash or reboot: --resume continues from runs\sarbon\s_bench_distill\ckpt.pt.
# Start outside the calling app's process tree (a plain Start-Process child dies with it):
#   Invoke-CimMethod Win32_Process -MethodName Create -Arguments @{CommandLine='powershell -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File D:\Sarbon\scripts\run_bench.ps1'}
param([string]$Coco = "C:\data\coco", [string]$Runs = "D:\Sarbon\runs")
$py = "D:\Sarbon\.venv\Scripts\python.exe"
Add-Content "$Runs\bench.log" "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') bench start (pid $PID)"
& $py -u D:\Sarbon\experiments\train_sarbon.py --coco $Coco --benchmark --size s --epochs 72 --batch 16 `
    --workers 12 --no-halo-zero --init "$Runs\pretrain\s_distill.pt" --tag "_distill" --out "$Runs\sarbon" --resume `
    *>> "$Runs\sarbon\train_s_bench_distill.log"
Add-Content "$Runs\bench.log" "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') bench exit $LASTEXITCODE"
