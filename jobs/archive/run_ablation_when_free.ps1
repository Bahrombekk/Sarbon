# Wait for COCO and a free GPU, then run the GPU tests and the halo-zero ablation.
# Usage: powershell -File scripts\run_ablation_when_free.ps1
param([string]$Coco = "D:\data\coco", [string]$Out = "D:\Sarbon\runs\halo_zero",
      [int]$Epochs = 24, [int]$Batch = 32)
$py = "D:\Sarbon\.venv\Scripts\python.exe"
New-Item -ItemType Directory -Force $Out | Out-Null
function Log($m) { $s = "$(Get-Date -Format 'HH:mm:ss') $m"; Write-Output $s; Add-Content "$Out\runner.log" $s }

Log "waiting for COCO in $Coco"
while (-not (Test-Path "$Coco\train2017.zip.extracted")) { Start-Sleep 60 }

Log "waiting for the GPU (Collector training finished and utilization < 10% for 5 min)"
$quiet = 0
while ($quiet -lt 5) {
    $busy = Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like "*Collector*train.py*" }
    $util = [int](nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)
    if ($busy -or $util -ge 10) { $quiet = 0 } else { $quiet++ }
    Start-Sleep 60
}
Log "GPU free"

& $py D:\Sarbon\tests\test_fused_engine.py *> "$Out\test_fused_engine.log"
Log "test_fused_engine exit $LASTEXITCODE"
& $py D:\Sarbon\tests\test_block_local.py *> "$Out\test_block_local.log"
Log "test_block_local exit $LASTEXITCODE"

foreach ($v in "standard", "halo") {
    Log "train $v"
    & $py -u D:\Sarbon\experiments\halo_zero_ablation.py --coco $Coco --variant $v `
        --epochs $Epochs --batch $Batch --out $Out *> "$Out\$v.log"
    Log "train $v exit $LASTEXITCODE"
}
Log "done"
