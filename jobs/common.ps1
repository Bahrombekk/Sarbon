# Shared helpers for the GPU job scripts. Dot-source it: . "$PSScriptRoot\common.ps1"
$Repo = Split-Path $PSScriptRoot -Parent
$Py = Join-Path $Repo ".venv\Scripts\python.exe"
if (-not (Test-Path $Py)) { $Py = "D:\Sarbon\.venv\Scripts\python.exe" }  # worktrees share the main venv
$env:PYTHONPATH = $Repo

function Log($file, $msg) { Add-Content $file "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $msg" }

# True while another detector training (old or new entry point) or the Collector training runs
function GpuBusy {
    [bool](Get-CimInstance Win32_Process | Where-Object {
        $c = $_.CommandLine
        $c -and $c -notlike "*multiprocessing*" -and (
            $c -like "*train_sarbon.py*" -or $c -like "*sarbon.cli.train*" -or $c -like "*sarbon-train*" -or
            $c -like "*sarbon.cli.pretrain*" -or
            $c -like "*Collector*train.py*") })
}

function WaitGpu($file) {
    Log $file "waiting for the GPU (pid $PID)"
    while (GpuBusy) { Start-Sleep 120 }
}

# python -m sarbon.cli.<module> <args>, output appended to $out
function RunSarbon($module, [string[]]$arguments, $out) {
    Push-Location $Repo
    & $Py -u -m "sarbon.cli.$module" @arguments *>> $out
    $code = $LASTEXITCODE
    Pop-Location
    return $code
}
