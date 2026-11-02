# O'zgarishlar tarixi

## 0.2.0 (2026-11-02): loyiha tuzilmasi

- Kutubxona (`sarbon/`) va ishga tushiriladigan qismlar ajratildi: `engine/` (o'qitish bloklari),
  `evaluation/` (COCO AP), `inference/` (video pipeline, kamera manbalari, chizish), `cli/` (buyruqlar).
- O'rnatiladigan paket (`pyproject.toml`), buyruqlar: `sarbon-train`, `sarbon-pretrain`,
  `sarbon-train-track`, `sarbon-run`, `sarbon-compare`.
- YAML konfiguratsiyalar (`configs/`), har bir CLI `--config` qabul qiladi.
- `sarbon/paths.py`: data/runs/weights joylari, muhit o'zgaruvchilari bilan almashtiriladi.
- `jobs/`: umumiy yordamchilar bilan GPU ishlari; `tools/`: data va kamera yordamchilari;
  `research/`: yopilgan tajribalar; `weights/`: chiqarilgan og'irliklar reestri.
- Fayl ko'chishlari: `experiments/train_sarbon.py` → `sarbon/cli/train.py`,
  `scripts/run_camera.py` → `sarbon/cli/run.py`, `bench/compare_own.py` → `sarbon/cli/compare.py`,
  `sarbon/pipeline.py` → `sarbon/inference/video.py`, `DESIGN.md` → `docs/DESIGN.md`,
  `scripts/*.py` → `tools/`, `scripts/*.ps1` → `jobs/`.

## 0.1.0 (2026-10-28 … 11-02): birinchi ishlaydigan versiya

- Sarbon-S detektori: blok-lokal stem, AIFI, FDR/GO-LSD, contrastive denoising, MAL, EMA.
- DINOv2 distillyatsiya init (+2.6 AP); to'liq COCO benchmark 42.6 AP (faqat tadqiqot).
- RailSafe: qisman yorliqlar bilan o'qitish; o'z val'da 74.1 AP (YOLO11s 69.3).
- Treker, temporal blok-kesh, kamera demo, kichik obyekt tahlili, qayta belgilash to'plami.
