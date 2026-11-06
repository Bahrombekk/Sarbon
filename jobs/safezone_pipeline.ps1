# SafeZone (SmartGUI shlem) modellari, ketma-ket (~20-24 soat):
#   1. yengil: G, 1024 px (configs\train\safezone_light_1024.yaml) -> runs\safezone\s_safezone_light_1024
#   2. og'ir: wide, 1280 px (configs\train\safezone_heavy_1280.yaml) -> runs\safezone\s_safezone_heavy_1280
# Yiqilsa (masalan DataLoader worker'lari) checkpoint'dan 3 martagacha davom ettiriladi (resume: true).
. "$PSScriptRoot\common.ps1"
$logs = Join-Path $Repo "runs\jobs"
$log = "$logs\safezone_pipeline.queue.log"
New-Item -ItemType Directory -Force $logs | Out-Null
WaitGpu $log
$steps = @(
    @("safezone_light_1024", "configs\train\safezone_light_1024.yaml"),
    @("safezone_heavy_1280", "configs\train\safezone_heavy_1280.yaml")
)
foreach ($s in $steps) {
    for ($try = 1; $try -le 4; $try++) {
        Log $log "$($s[0]) start (try $try)"
        $code = RunSarbon "train" @("--config", $s[1]) "$logs\$($s[0]).log"
        Log $log "$($s[0]) exit $code"
        if ($code -eq 0) { break }
        Start-Sleep 60
    }
    if ($code -ne 0) { break }
}
Log $log "done"
