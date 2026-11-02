# Accuracy ablations, 24 epochs from scratch, same recipe as the fusion ablation (baseline concat:
# AP 30.2, AP_small 11.0, runs\fusion\s_fusion_concat):
#   nwd1 : NWD term in the Hungarian matching cost (small objects)
#   wide : backbone x2 (stem 48/96/192, depths 2/3, stride-16 384 ch) -- is capacity the bottleneck?
# Start outside the calling app's process tree (see run_bench.ps1):
#   Invoke-CimMethod Win32_Process -MethodName Create -Arguments @{CommandLine='powershell -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File D:\Sarbon\scripts\run_queue5.ps1'}
param([string]$Coco = "C:\data\coco", [string]$Runs = "D:\Sarbon\runs\ablate2", [int]$Epochs = 24)
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
$jobs = [ordered]@{
    "nwd1" = @("--nwd", "1");
    "wide" = @("--widths", "48,96,192", "--depths", "2,3", "--c16", "384")
}
foreach ($name in $jobs.Keys) {
    Log "$name start"
    $extra = $jobs[$name]
    & $py -u D:\Sarbon\experiments\train_sarbon.py --coco $Coco --size s --epochs $Epochs --no-aug-epochs 2 `
        --batch 16 --workers 12 --no-halo-zero --tag "_$name" --out $Runs --resume @extra *>> "$Runs\$name.log"
    Log "$name exit $LASTEXITCODE"
}
Log "done"
