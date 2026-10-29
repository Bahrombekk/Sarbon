# Overnight GPU queue: tests -> halo-zero ablation (standard, halo) -> Sarbon-S commercial training.
# Start detached so it survives the terminal/session:
#   Start-Process powershell -WindowStyle Hidden -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass','-File','D:\Sarbon\scripts\run_overnight.ps1'
# COCO lives on the NVMe drive: on the D: HDD random JPEG reads starved the GPU (~48 img/s).
param([string]$Coco = "C:\data\coco", [string]$Runs = "D:\Sarbon\runs", [int]$Workers = 12,
      [string[]]$Variants = @("standard", "halo"), [switch]$SkipTests)
$py = "D:\Sarbon\.venv\Scripts\python.exe"
$log = "$Runs\overnight.log"
New-Item -ItemType Directory -Force "$Runs\halo_zero", "$Runs\sarbon" | Out-Null
function Log($m) { Add-Content $log "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $m" }
function Busy { [bool](Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like "*Collector*train.py*" }) }

Log "queue started (pid $PID)"
while (Busy) { Log "Collector training is running; waiting"; Start-Sleep 300 }

if (-not $SkipTests) {
    & $py D:\Sarbon\tests\test_fused_engine.py *> "$Runs\halo_zero\test_fused_engine.log"
    Log "test_fused_engine exit $LASTEXITCODE"
    & $py D:\Sarbon\tests\test_block_local.py *> "$Runs\halo_zero\test_block_local.log"
    Log "test_block_local exit $LASTEXITCODE"
}

foreach ($v in $Variants) {
    Log "ablation $v start"
    & $py -u D:\Sarbon\experiments\halo_zero_ablation.py --coco $Coco --variant $v --epochs 24 `
        --batch 32 --workers $Workers --out "$Runs\halo_zero" *> "$Runs\halo_zero\$v.log"
    Log "ablation $v exit $LASTEXITCODE"
}

Log "sarbon-s start"
& $py -u D:\Sarbon\experiments\train_sarbon.py --coco $Coco --size s --epochs 72 --batch 16 `
    --workers $Workers --out "$Runs\sarbon" *> "$Runs\sarbon\train_s.log"
Log "sarbon-s exit $LASTEXITCODE"
Log "queue done"
