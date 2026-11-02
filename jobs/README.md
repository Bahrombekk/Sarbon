# GPU jobs (Windows)

Long runs are started **outside** the calling app's process tree, so closing or restarting an editor,
terminal or the Claude app does not kill them:

```powershell
Invoke-CimMethod Win32_Process -MethodName Create -Arguments @{CommandLine='powershell -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File D:\Sarbon\jobs\<job>.ps1'}
```

Every job waits until no other detector training (or the Collector training) uses the GPU, logs
to `runs\...`, and trains with `--resume`: if the machine reboots, start the same job again and it
continues from the last finished epoch.

| Job | What |
|---|---|
| `train.ps1 -Config <yaml> -Name <n> -Extra ...` | any training config |
| `railsafe_finetune.ps1` | product detector (`configs/train/railsafe_s.yaml`) |
| `bench_full_coco.ps1` | research-only full COCO benchmark |
| `ablate_nwd_wide.ps1` | NWD and 2x-backbone ablations, 24 epochs |
| `yolo_compare.ps1` | YOLO11s on the own split + side-by-side comparison |

`archive/` holds finished one-off queues; they reference the old file layout and are kept only
as a record of how past results were produced.
