# Sarbon

Statik kameralar uchun real vaqtli detektor + treker. Asosiy g'oya: hisoblash hajmi
piksellar soniga emas, **sahnadagi o'zgarishga** proporsional bo'lishi kerak.

- Arxitektura, tadqiqot xulosalari, o'lchangan natijalar va yo'l xaritasi: [DESIGN.md](DESIGN.md)
- Yadro: `sarbon/sparse/`. Unda temporal block cache va fused Triton tile-conv yadrosi bor.

## Muhit

```bash
py -3.10 -m venv --system-site-packages .venv
.venv\Scripts\python.exe -m pip install pycocotools timm triton-windows
```

PyTorch CUDA 12.8 build kerak (RTX 50xx uchun).

## Testlar va benchmarklar

```bash
.venv\Scripts\python.exe tests\test_fused_engine.py
.venv\Scripts\python.exe bench\bench_m1.py
.venv\Scripts\python.exe bench\bench_noise.py
```

## Litsenziya

Yopiq kod (proprietary). Uchinchi tomon kodi yoki og'irliklari faqat Apache-2.0/MIT kabi tijoriy
foydalanishga ruxsat beruvchi litsenziyalar ostida ishlatiladi. DESIGN.md'ning 4-bo'limiga qarang.
