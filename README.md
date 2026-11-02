# Sarbon

Qo'zg'almas kameralar uchun real vaqtli detektor + treker (RailSafe temir yo'l kesishmalari).
Asosiy g'oya: hisoblash hajmi piksellar soniga emas, **sahnadagi o'zgarishga** proporsional bo'lishi kerak.

- Arxitektura, tadqiqot xulosalari, o'lchangan natijalar, yo'l xaritasi: [docs/DESIGN.md](docs/DESIGN.md)
- O'zgarishlar tarixi: [CHANGELOG.md](CHANGELOG.md)

## Tuzilma

```
sarbon/                 kutubxona (import qilinadigan kod)
  models/               detektor: blok-lokal stem, dense core, DETR dekoder, loss va matcher
  data/                 datasetlar: COCO, YOLO-format o'z data, mosaic, MOT kliplari, psevdo-yorliq
  engine/               o'qitish bloklari: preprocessing, EMA, init/remap, optimizer guruhlari
  evaluation/           COCO AP (pycocotools), DETR post-processing
  track/                treker: ByteTrack asosi + appearance, re-ID, yer tekisligi (ground plane)
  sparse/               temporal blok-kesh: o'zgargan bloklarni qayta hisoblash, Triton yadrolari
  inference/            video pipeline (SarbonVideo), kamera manbalari, chizish
  cli/                  buyruqlar: sarbon-train, sarbon-run, sarbon-compare, ...
  config.py paths.py    YAML konfiguratsiya, data/runs/weights joylari
configs/                o'qitish va ishga tushirish retseptlari (YAML)
  train/                railsafe_s (mahsulot), coco_s, coco_bench_s (faqat tadqiqot), ablation_24ep
  run/                  camera.example.yaml (nusxa olib camera.yaml qiling, u git'ga tushmaydi)
jobs/                   uzoq GPU ishlari (PowerShell), ilovadan mustaqil ishga tushiriladi
tools/                  data va kamera yordamchilari: split, qayta belgilash to'plami, kamera yozish
bench/                  benchmark va tahlil skriptlari (tezlik, shovqin, recall, treking simulyatsiyasi)
research/               yopilgan tajribalar (natijasi docs/DESIGN.md da)
tests/                  testlar
weights/                chiqarilgan og'irliklar (git'da emas, ro'yxati weights/README.md da)
docs/                   dizayn va natijalar
runs/                   o'qitish natijalari (git'da emas)
```

Qoida: qayta ishlatiladigan kod faqat `sarbon/` ichida. `tools/`, `bench/`, `jobs/` undan import
qiladi, aksincha emas.

## O'rnatish

```bash
py -3.10 -m venv --system-site-packages .venv
.venv\Scripts\python.exe -m pip install -e .[gpu,dev]
```

PyTorch CUDA 12.8 build alohida o'rnatiladi (RTX 50xx uchun). `-e` bilan o'rnatilgani uchun kod
o'zgarishi darhol ishlaydi va `sarbon-*` buyruqlari `.venv\Scripts\` ichida paydo bo'ladi.

## Ishlatish

```bash
# kamerada (yoki video faylda) aniqlash + treking
.venv\Scripts\sarbon-run rtsp://USER:PASS@192.168.1.10:554/stream1
.venv\Scripts\sarbon-run --config configs\run\camera.yaml

# o'qitish (har qanday parametr buyruq qatorida konfiguratsiyani bosib o'tadi)
.venv\Scripts\sarbon-train --config configs\train\railsafe_s.yaml
.venv\Scripts\sarbon-train --config configs\train\ablation_24ep.yaml --nwd 1 --tag _nwd1

# o'z data'da Sarbon va YOLO'ni bir xil baholovchida solishtirish
.venv\Scripts\sarbon-compare --sarbon weights\sarbon-s-railsafe.pt --yolo-json runs\yolo\yolo11s_val.json
```

Data, runs va weights joylari: `SARBON_DATA` (standart `C:\data`), `SARBON_RUNS` (`runs\`),
`SARBON_WEIGHTS` (`weights\`).

## Testlar

```bash
.venv\Scripts\python.exe -m pytest
.venv\Scripts\python.exe tests\test_fused_engine.py      # GPU
.venv\Scripts\python.exe tests\test_pipeline_gpu.py      # GPU
```

## Litsenziya

Yopiq kod (proprietary), [LICENSE](LICENSE). Uchinchi tomon kodi yoki og'irliklari faqat
Apache-2.0/MIT kabi tijoriy foydalanishga ruxsat beruvchi litsenziyalar ostida ishlatiladi
(docs/DESIGN.md, 4-bo'lim). Ultralytics (AGPL) faqat alohida muhitda, o'lchov uchun ishlatiladi.
