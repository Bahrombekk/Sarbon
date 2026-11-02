# Generic training job: waits for a free GPU, then sarbon-train --config <Config> <Extra...>.
#   jobs\train.ps1 -Config configs\train\ablation_24ep.yaml -Name nwd1 -Extra "--nwd","1","--tag","_nwd1"
param([Parameter(Mandatory)][string]$Config, [string]$Name = "train", [string[]]$Extra = @())
. "$PSScriptRoot\common.ps1"
$logs = Join-Path $Repo "runs\jobs"
New-Item -ItemType Directory -Force $logs | Out-Null
$log = "$logs\$Name.queue.log"
WaitGpu $log
Log $log "$Name start: --config $Config $Extra"
$code = RunSarbon "train" (@("--config", $Config) + $Extra) "$logs\$Name.log"
Log $log "$Name exit $code"
Log $log "done"
