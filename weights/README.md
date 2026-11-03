# Released weights

Model files (`*.pt`, `*.onnx`, `*.engine`) are not stored in git. This folder holds the
checkpoints the tools use by default (`sarbon/paths.py`, override with `SARBON_WEIGHTS`).

| File | What | Licence status | Source run |
|---|---|---|---|
| `sarbon-s-railsafe.pt` = `sarbon-s-railsafe-v0.1.pt` | Sarbon-S, 6 RailSafe classes (person, bicycle, motorcycle, yengil, ogir, train). Own val AP 74.1 | **Product**: commercial COCO images + own data | `runs/railsafe/s_railsafe/best.pt`, config `configs/train/railsafe_s.yaml` |
| `sarbon-s-railsafe-smart.pt` | v0.2 candidate, same classes, + people from the own smart cameras. Pereezd val AP 72.5 (-1.6), smart person AP 79.1 | **Product**: commercial COCO images + own data | `runs/railsafe/s_railsafe_smart/best.pt`, config `configs/train/railsafe_smart_s.yaml` |

Rules:
- Only weights trained on commercially usable data go here. Research-only runs (full COCO
  benchmark, anything with non-commercial images or AGPL tools) stay in `runs/`.
- When a new model replaces one, add a row with its config and metrics. Keep the old file
  under a versioned name (`sarbon-s-railsafe-v0.1.pt`) until the new one is checked on cameras.
