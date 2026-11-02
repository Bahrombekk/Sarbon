# YOLO11s baseline on the own split, then Sarbon vs YOLO with one evaluator.
# Ultralytics (AGPL) runs only in the separate Collector env, for measurement; never in the product.
param([string]$Split = "C:\data\pereezd_split",
      [string]$YoloPy = "C:\Users\User\Desktop\Collector\venv310\Scripts\python.exe",
      [string]$Sarbon = "weights\sarbon-s-railsafe.pt")
. "$PSScriptRoot\common.ps1"
$runs = Join-Path $Repo "runs"
$yolo = "$runs\yolo"
$log = "$runs\yolo_compare.log"
New-Item -ItemType Directory -Force $yolo | Out-Null
WaitGpu $log
Push-Location $yolo  # Ultralytics downloads yolo11s.pt into the working folder
Log $log "yolo11s train start"
& $YoloPy -u "$Repo\tools\yolo_own.py" train --split $Split --out $yolo --epochs 50 --batch 16 --workers 8 *> "$yolo\train_yolo11s.log"
$w = "$yolo\yolo11s_own\weights\best.pt"
& $YoloPy -u "$Repo\tools\yolo_own.py" dump --weights $w --split $Split --json "$yolo\yolo11s_val.json" *> "$yolo\dump.log"
& $YoloPy -u "$Repo\tools\yolo_own.py" speed --weights $w --json "$yolo\yolo11s_speed.json" *> "$yolo\speed.log"
Pop-Location
Log $log "yolo done; compare"
$code = RunSarbon "compare" @("--split", $Split, "--sarbon", $Sarbon, "--yolo-json", "$yolo\yolo11s_val.json",
                              "--yolo-speed", "$yolo\yolo11s_speed.json", "--out", "$runs\compare_own.json") "$runs\compare_own.log"
Log $log "compare exit $code"
Log $log "done"
