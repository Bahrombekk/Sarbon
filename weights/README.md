# Released weights

Model files (`*.pt`, `*.onnx`, `*.engine`) are not stored in git. This folder holds the
checkpoints the tools use by default (`sarbon/paths.py`, override with `SARBON_WEIGHTS`).

| File | What | Licence status | Source run |
|---|---|---|---|
| `sarbon-s-railsafe.pt` = `sarbon-s-railsafe-v0.1.pt` | Sarbon-S, 6 RailSafe classes (person, bicycle, motorcycle, yengil, ogir, train). Own val AP 74.1 | **Product**: commercial COCO images + own data | `runs/railsafe/s_railsafe/best.pt`, config `configs/train/railsafe_s.yaml` |
| `sarbon-s-railsafe-1088.pt` | **v0.3, in use.** Wide backbone (18.5M), same 6 classes, input 1088 (run at 640x1152 on 16:9 frames). From the general base with own people, trained like the RailSafe YOLO26n (its train/val, original labels). Its val: 91.0 AP (YOLO26n 89.4), small 71.7 (65.7); Open Images vehicles 74.0 (YOLO26n 11.4) | **Product**: commercial COCO images + own data | `runs/railsafe/s_railsafe_full_1088/best.pt`, configs `coco_wide_person.yaml` then `railsafe_full_1088.yaml` |
| `sarbon-s-railsafe-light-1088.pt` | **Fast candidate** (light network G, 8.7M). Same recipe as v0.3 from the light base. YOLO26n val: 89.1 AP (v0.3 91.0, YOLO26n 89.4), small 68.0 (65.7); Open Images 71.0 (v0.3 74.0). TensorRT fp16 1088: b1 1.7 ms, b6 3.7 ms (v0.3 2.5 / 7.9, YOLO26n 3.0 / 2.9) | **Product**: commercial COCO images + own data | `runs/railsafe/s_railsafe_light_1088/best.pt`, configs `coco_light_person.yaml` then `railsafe_light_1088.yaml` |
| `sarbon-s-railsafe-smart.pt` | v0.2 candidate, same classes, + people from the own smart cameras. Pereezd val AP 72.5 (-1.6), smart person AP 79.1 | **Product**: commercial COCO images + own data | `runs/railsafe/s_railsafe_smart/best.pt`, config `configs/train/railsafe_smart_s.yaml` |

Rules:
- Only weights trained on commercially usable data go here. Research-only runs (full COCO
  benchmark, anything with non-commercial images or AGPL tools) stay in `runs/`.
- When a new model replaces one, add a row with its config and metrics. Keep the old file
  under a versioned name (`sarbon-s-railsafe-v0.1.pt`) until the new one is checked on cameras.
