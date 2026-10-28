# Sarbon — o'zgarishga qarab hisoblaydigan detektor + treker

> **Asosiy tezis:** statik kamerada hisoblash hajmi **piksellar soniga emas, sahnadagi o'zgarishga**
> proporsional bo'lishi kerak. Kichik obyektlar uchun yuqori rezolyutsiya esa faqat
> "qarash kerak bo'lgan" joyda ishlatiladi. Bu joylarni o'zgarish, track'lar va kichik obyekt
> ehtimoli belgilaydi.

Holat: 2026-10-28. M0 (yadro g'oya isboti) va M1 (fused Triton engine, ko'p kamera) bajarildi va
o'lchandi (5 va 5b bo'limlar). 2-tadqiqot raundidan keyin **halo-zero** arxitekturasi taklif qilindi
va CPU'da isbotlandi (5c bo'lim). Uning aniqlikka ta'siri GPU ablation'da tekshiriladi.

---

## 1. Nega bu yangi (tadqiqot xulosasi)

Uchta parallel adabiyot tadqiqoti o'tkazildi (efficient detektorlar 2023–2026, kichik obyekt +
adaptive compute, video qayta foydalanish + treking). Har bir qism alohida mavjud:

| Qism | Eng yaqin oldingi ish | Uning kamchiligi |
|---|---|---|
| (a) O'zgarish bo'yicha siyrak qayta hisoblash | CBinfer (1704.04313), DeltaCNN (2203.03996), Eventful Transformers (2308.13494) | Faqat vaqt bo'yicha, bir xil rezolyutsiya, treking yo'q. DeltaCNN **CC BY-NC** |
| (b) Track-query'lar bilan native treking | MOTR / MeMOTR / CO-MOT / MOTIP | Har kadr to'liq (dense) kodlanadi |
| (c) Kichik obyektlar uchun yuqori rezolyutsiya faqat kerakli joyda | QueryDet, ESOD, HGSQ (2609.13306), FOVEA, LZU | Har kadr qaytadan hisoblanadi, vaqt bo'yicha qayta foydalanish yo'q |

**Hech bir ish (a)+(b)+(c) ni bitta real vaqtli tizimda birlashtirmagan.** Ayniqsa
**track'lar qayerni qayta hisoblashni boshqarishi** (track-steered recompute) bo'yicha oldingi ish topilmadi.
Bu xulosa abstraktlar va README'lar asosida chiqarilgan. Maqola yozishdan oldin to'liq tekshiruv kerak.

Sarbon'ning yangi hissalari (gipotezalar, har biri tajribada isbotlanishi kerak):

1. **Lossless temporal block cache.** Lokal (conv) bosqichlar faqat receptive field'i o'zgargan
   bloklarda qayta hisoblanadi va natija to'liq hisoblash bilan **matematik jihatdan bir xil**
   (yaxlitlash darajasida). ✅ isbotlandi, pastga qarang.
2. **Qat'iy hisoblash byudjeti (bounded latency).** Har kadrda qayta hisoblanadigan bloklar soni
   oldindan belgilangan. Butun qadam (gating → gather → conv → scatter) CPU bilan
   sinxronlashsiz bitta CUDA grafida ishlaydi. ✅ prototip.
3. **Track-steered recompute.** Gate = piksel o'zgarishi ∪ track'larning keyingi kadrdagi bashorat
   qilingan joyi ∪ navbatma-navbat yangilash (drift'ni cheklash uchun).
4. **Foveal kichik-obyekt yo'li.** Asosiy tarmoq butun kadrni o'rtacha rezolyutsiyada ko'radi.
   To'liq rezolyutsiyadagi tile'lar faqat kichik obyekt heatmap'i yoki kichik track'lar bor
   bloklarda hisoblanadi. Buning uchun xuddi o'sha tile mexanizmi ishlatiladi.
5. **Track-query decoder.** Track'lar decoder'ga query sifatida kiradi: aniqlash va identifikatsiya bitta
   bosqichda bo'ladi, NMS kerak emas.
6. **Ko'p kamerali tile batching.** Serverda barcha kameralarning faol tile'lari bitta batch'ga
   yig'iladi. GPU yuklamasi kameralar soniga emas, **jami o'zgarishga** bog'liq bo'ladi.

## 2. Arxitektura

```
 kadr t ──► [Gate]  |Δpiksel| bloklar ∪ track bashorati ∪ refresh ∪ global-trigger
               │        (byudjet K; oshib ketsa → dense refresh)
               ▼
 ┌─ LOKAL YUQORI-REZ BOSQICHLAR (stride 2/4/8) ─ faqat conv/aktivatsiya/residual ─┐
 │   Temporal Block Cache: faol tile'lar gather → conv → scatter (lossless)       │
 │   + Foveal tile'lar (to'liq rez) kichik-obyekt bloklarida                       │
 └──────────────────────────────────────────────────────────────────────────────┘
               ▼   P3 (stride 8) kesh — har doim aktual
 ┌─ DENSE PAST-REZ YADRO (stride 16/32) ─ har kadr to'liq, arzon ───────────────┐
 │   HGNetv2-uslub bosqichlar + hybrid encoder (attention faqat S5'da)            │
 └──────────────────────────────────────────────────────────────────────────────┘
               ▼
 ┌─ DECODER ───────────────────────────────────────────────────────────────────┐
 │   query'lar = track query'lar (oldingi kadrdan) + top-K yangi query'lar       │
 │   deformable attention, FDR distribution regression, NMS'siz                  │
 │   chiqish: box + klass + track ID → keyingi kadr gate'i uchun bashorat        │
 └──────────────────────────────────────────────────────────────────────────────┘
```

**Asosiy dizayn qoidasi:** global operatsiyalar (attention, SE, global pooling) faqat stride ≥16
qismida bo'ladi. Yuqori rezolyutsiyali qism faqat lokal operatsiyalardan tuzilgan. Shu tufayli lossless
siyrak hisoblash mumkin bo'ladi, va hisoblashning asosiy qismi aynan shu yerda.

**O'qitish (training-only, inference xarajatsiz):**
- DEIM Dense O2O + Matchability-Aware Loss: tez konvergensiya.
- D-FINE FDR + GO-LSD: aniq lokalizatsiya.
- Kichik obyektlar: matching cost'da NWD (Wasserstein), STAL-uslub kafolatlangan pozitivlar.
- DINOv2 (Apache 2.0) teacher'dan feature distillyatsiyasi (RT-DETRv4 g'oyasi).
- Klip bo'yicha o'qitish: track-query propagatsiyasi (MOTR/CO-MOT) va **staleness simulyatsiyasi**,
  ya'ni approximate rejimda eskirgan feature'larga chidamlilik.

**Ishlash rejimlari:** `tau=0` lossless rejim. `tau>0` approximate rejim kamera shovqinini
e'tiborsiz qoldiradi. Har bir blok kamida har N kadrda bir marta yangilanadi. Yorug'lik keskin
o'zgarsa yoki kamera silkinsa, global trigger to'liq hisoblashni ishga tushiradi. H.264/H.265
dekoderidagi skip-makrobloklar "tekin" o'zgarish maskasi bo'lib xizmat qilishi mumkin (keyingi tadqiqot).

## 3. Til va texnologiya steki — qaror

| Qatlam | Til / vosita | Nega |
|---|---|---|
| Tadqiqot, o'qitish | **Python 3.10 + PyTorch 2.11 (CUDA 12.8)** | Tez iteratsiya. Barcha manbalar shu ekotizimda |
| Issiq yadrolar (gather-conv-scatter fused) | **Triton** (`triton-windows` 3.8 o'rnatilgan, 5090'da ishladi) → kerak bo'lsa **CUDA C++** | Tezlikni Python emas, yadrolar beradi |
| Deploy inference | **ONNX → TensorRT** + block gather/scatter uchun **CUDA C++ TensorRT plugin** | Dense qismlar uchun eng tez yo'l |
| Runtime (video dekodlash, kameralar, tracker holati, byudjet) | **C++17** (NVDEC, CUDA Graphs) | TensorRT/NVDEC API'lari C++. Hot path'da Python bo'lmaydi |
| Edge / CPU | Jetson'da xuddi o'sha TensorRT yo'li. CPU uchun ONNX Runtime / OpenVINO | |

Muhim halol eslatma: modelni qo'lda C++'ga ko'chirishning o'zi matematikani tezlashtirmaydi.
Tezlik ikki manbadan keladi: fused yadrolar va Python/launch overhead'ini yo'qotish. Prototipda
aynan shu ko'rindi: overhead olib tashlanganda kadr ~60 ms'dan 0.68 ms'ga tushdi. Rust yoki Mojo hozircha
kerak emas, chunki TensorRT va NVDEC bilan ishlash uchun C++ tabiiy tanlov.

## 4. Litsenziya qoidalari (tijoriy, yopiq kod)

- ✅ Ruxsat: RT-DETR, D-FINE, DEIM (v1, LICENSE'ni qayta tekshirish kerak), RF-DETR N–L, DINOv2,
  ByteTrack, OC-SORT, BoT-SORT, MOTR/MeMOTR/CO-MOT (MIT). Apache/MIT kodidan foydalanilsa,
  LICENSE va mualliflik eslatmalari saqlanadi.
- ❌ Ishlatmaymiz: Ultralytics (AGPL) va BoxMOT (AGPL), **DEIMv2 (tijoriy bo'lmagan litsenziya)**, DeltaCNN
  (CC BY-NC), RF-DETR XL/2XL (PML), StrongSORT (GPL), ToMe (NC, tekshirilmagan).
- ⚠️ Yurist tekshiruvi kerak: DINOv3 va undan distillyatsiya qilingan og'irliklar (harbiy foydalanish
  cheklovi), **Objects365'da oldindan o'qitilgan har qanday checkpoint**, UA-DETRAC kabi faqat
  tadqiqot uchun mo'ljallangan datasetlar.
- Sarbon kodi noldan yozilmoqda. Oldingi ishlardan faqat g'oyalar olinadi.

## 5. 0-bosqich natijasi (o'lchangan)

`bench/bench_block_sparse.py`: ResNet-18 uslubidagi lokal bosqichlar (stride 4 va 8), 1920×1088, RTX 5090,
FP16, CUDA graph, gating xarajati ham kiritilgan, 200 o'lchovning medianasi.

- **Lossless:** ketma-ket 3 kadrdan keyin ham maksimal nisbiy xato **2e-6** (FP32, TF32 o'chirilgan).
  Xato to'planmaydi.
- **Tezlik (block = 32 px):**

| O'zgargan maydon | Faol bloklar (oxirgi qatlam) | Byudjet | ms | Dense'ga nisbatan |
|---|---|---|---|---|
| 0.5% | 4.4% | 6.3% | 0.68 | **2.17x tez** |
| 2% | 21% | 25% | 1.07 | 1.38x |
| 5% | 31% | 50% | 1.62 | 0.91x |
| ≥20% | ≥75% | 100% | 3.3–3.7 | 0.4x (sekin) |

Dense yo'l ~1.4 ms. Testda har kadr tasodifiy to'rtburchaklar bilan o'zgartiriladi. Bu haqiqiy
kuzatuv kamerasidan og'irroq holat.

**Topilgan zaif joylar va yechimlar (1-bosqich):**
1. Gather/scatter PyTorch indekslash orqali ishlayapti va katta faollikda dense'dan sekinroq.
   Yechim: **Triton fused gather→conv→scatter** yadrosi.
2. Byudjet 2 ning darajalari bo'yicha olinyapti (31% → 50% isrof). Yechim: maydaroq byudjet pog'onalari.
3. Faollik ~30% dan oshsa, dense yo'l tezroq. Yechim: **avtomatik dense fallback**.
   Shunda eng yomon holat hech qachon dense'dan sekin bo'lmaydi.
4. Receptive field o'zgarishni kengaytiradi (0.5% → 4.4%). Yechim: high-res bosqichlarni sayozroq
   qilish va chuqur qatlamlarni past rezolyutsiyaga ko'chirish (arxitektura qaroriga ta'sir qiladi).
5. Bitta kamera uchun 5090'da dense allaqachon ~1.4 ms. Sarbon'ning haqiqiy yutug'i **ko'p kamerali
   serverda va Jetson kabi kuchsiz qurilmalarda** ko'rinadi. Navbatdagi benchmark shular uchun.

1–3-bandlar M1'da hal qilindi. 4-band M3'ga qoldi.

## 5b. M1 natijasi (o'lchangan, 2026-10-28)

**Nima qurildi:**
- `sarbon/sparse/triton_conv.py`: bitta Triton yadrosi gather → 3×3 conv (implicit GEMM, tensor
  core) → bias → (+residual) → ReLU → scatter ishlarini birga bajaradi. Keshlar NHWC tensorlar.
  Halo to'g'ridan-to'g'ri keshdan o'qiladi, nol-padding esa load niqoblari bilan
  emulyatsiya qilinadi, shuning uchun oraliq nusxa umuman yo'q. Har dastur 64 pikseldan iborat,
  bu kichik byudjetlarda ham parallellikni saqlaydi.
- `sarbon/sparse/fused_engine.py`: ko'p kamerali engine. Ish tartibi:
  1. Gate grafi: o'zgargan bloklar topiladi, har qatlam uchun dilatsiya qilinadi va sanaladi.
     Kadrga faqat bitta host sync qilinadi.
  2. Byudjet pog'onasi tanlanadi: pog'onalar ~15% oraliqda, 16 ning karralisi.
  3. Shu pog'ona uchun sparse CUDA grafi yoki dense graf ijro etiladi.
  - **Barcha kameralarning faol tile'lari bitta batch'ga yig'iladi.** Dense yo'l oddiy cuDNN, uning
    channels_last chiqishlari keshlarning o'zi bo'ladi, shuning uchun fallback qo'shimcha nusxasiz.
  - Fallback chegarasi qurilmada avtomatik kalibrlanadi.
  - Dense rejimda histerezis bor: gate faqat har 4 kadrda tekshiriladi.
  - `tau>0` approximate rejimi va `refresh` (staleness chegarasi) ham bor.
- `tests/test_fused_engine.py`: fp32 lossless testi. U 3 kamera, chegara bloklari, zanjirli kadrlar,
  dense → sparse o'tishlari va histerezisni qamraydi. **Xato 1.9e-6, ya'ni natija dense bilan bir xil.**
  fp16 xatosi 1.5e-3, bu dense fp16'ning o'z xatosi (1.2e-3) darajasida.

**Tezlik** (`bench/bench_m1.py`). Sintetik statik kamera videosi ishlatildi: teksturali fon va
harakatlanuvchi teksturali obyektlar. Har kadrda dense va Sarbon A/B navbatma-navbat o'lchandi.
Dense baseline ham CUDA graf bilan ishladi, ya'ni o'zi uchun eng qulay sharoitda. Host sync
o'lchovga kiritilgan.

| Scenariy (1920×1088, block 32) | Faol bloklar | Dense cuDNN | Sarbon | Tezlanish (median A/B) |
|---|---|---|---|---|
| 16 kamera, hammasi tinch | 3.5% | 16.7 ms | 2.58 ms | **6.8x** |
| 16 kamera aralash (10 tinch / 4 o'rta / 2 band) | 11.6% | 15.4 ms | 3.43 ms | **4.5x** |
| 16 kamera, hammasi band (eng yomon) | 47.9% | 15.7 ms | 7.49 ms | **2.1x** |
| 1 kamera tinch | 3.6% | 0.79 ms (best) | 0.55 ms (best) | 1.1x (best: 1.4x) |
| 1 kamera o'rta | 13.4% | 0.72 ms (best) | 0.56 ms (best) | 1.2x (best: 1.3x) |
| 1 kamera band → dense fallback | 44.9% | 0.74 ms (best) | 0.66 ms (best) | 0.96x (best: 1.1x) |

- **M1 mezonlari:**
  - "≤5% o'zgarishda ≥3x": ko'p kamerali serverda **bajarildi** (6.8x).
  - Bitta kamerada bajarilmadi (~1.1–1.4x). 5090'da bitta 1080p kadr uchun dense atigi ~0.7 ms va
    kadr boshiga ~0.1 ms qat'iy overhead (gate + sync) bunga ulush qo'shadi.
  - "Eng yomon holat ≥1x": 16 kamerada bajarildi (2.1x). 1 kamerada median 0.96x, eng yaxshi
    vaqtda 1.1x.
- **Kutilmagan natija:** 48% faollikda ham sparse yo'l 2x tez. Fused Triton yadrosi piksel boshiga
  cuDNN bilan deyarli bir xil samarali, shuning uchun 16 kamerada fallback chegarasi ~70%.
- **Ogohlantirish:** o'lchovlar vaqtida GPU'da boshqa o'qitish jarayoni ishlab turgan edi
  (`Collector\train.py`, ~89% yuklama). Shuning uchun A/B interleaving va "best" vaqt
  ishlatildi. 16 kamerali natijalar barqaror chiqdi, bitta kameraniki esa shovqinli. Toza GPU'da
  qayta o'lchash kerak.

**Shovqin va approximate rejim** (`bench/bench_noise.py`, sensor shovqini σ = 2/255, 1 kamera, o'rta):

| tau | Faol bloklar | Toza sahnadan chetlanish (shovqinning o'z ta'siriga nisbatan) |
|---|---|---|
| 0 (lossless) | 100% (shovqin hamma joyni o'zgartiradi) | 1.00x |
| 4/255 (≈2σ) | **12.9%** (≈ haqiqiy harakat) | **1.13x** |
| 6/255 | 12.7% | 1.50x |

Amaliy qoida: `tau ≈ 2σ`. `refresh=30` faollikni ikki baravar oshiradi, lekin deyarli foyda bermaydi.
Refresh davri uzunroq bo'lishi kerak (yuzlab kadrlar).

Bu xulosalar haqiqiy kamera shovqini va H.264 artefaktlarida tekshirilmagan. Detektor AP'iga
ta'siri M3'da o'lchanadi.

**Muhim muhandislik saboqlari:**
- Dense grafning chiqishlari keshlar sifatida tirik qoladi, shuning uchun u boshqa graflar bilan
  xotira pulini bo'lisha olmaydi. Aks holda gate grafining vaqtinchalik tensorlari keshni buzadi.
  Bu xato topildi, tuzatildi va testga qo'shildi.
- Windows'da PyTorch'ning mayda operatsiyalari CPU'ni bog'lab qo'yadi. CUDA graf va bitta sync
  bo'lmasa, sparse yo'l 40 ms'gacha sekinlashadi.

**M1'dan keyin qolgan ishlar va g'oyalar:**
- Gate'ni bitta Triton yadrosiga birlashtirish. Bu kadr boshiga ~0.05 ms tejaydi va bitta kamera
  holatini yaxshilaydi.
- Kamera bo'yicha alohida fallback: band kameralar dense, tinchlari sparse rejimda ishlaydi.
- Kodek ma'lumotidan foydalanish: H.264 P-kadrlarida skip makrobloklar piksel darajasida bir xil
  bo'ladi, ya'ni shovqinsiz "tekin" gate. I-kadrlarda esa hamma narsa o'zgaradi, shuning uchun har
  GOP'da (1–2 s) bitta dense kadr kerak bo'ladi.
- Haqiqiy video (statik kamerali ochiq datasetlar) bilan qayta o'lchash.
- Jetson'da o'lchash. U yerda dense qimmat, shuning uchun foyda kattaroq bo'lishi kutiladi.

## 5c. 2-tadqiqot raundi va yangi arxitektura qarori (2026-10-28, GPU'siz)

### Asosiy kashfiyot: halo-zero yuqori rezolyutsiya bosqichlari

**Muammo.** M0/M1 engine'ida o'zgarish receptive radius (55 px) bo'yicha qo'shni bloklarga
tarqaladi. Natijada kadrning 0.24%i o'zgarsa ham GPU ishining 2.6%i bajariladi.

**Yechim.** Stride ≤ 8 bosqichlarida har bir 32 px blok **mustaqil** kodlanadi: bloklar batch
o'lchamiga aylantiriladi va conv'lar blok chegarasida nol bilan padding qiladi. Bloklararo kontekst
faqat dense yadroda (stride 16/32) va top-down fusion orqali beriladi. Bu ViT'ning o'zaro
kesishmaydigan patchify qadamining CNN analogi; RF-DETR va LW-DETR shu tamoyilda ishlaydi.

**Afzalliklari:**
- Sparse yangilash aniq bo'lib qoladi va soddalashadi: faqat o'zgargan bloklar qayta kodlanadi,
  hech qanday dilatsiya kerak emas.
- Maxsus yadro kerak emas, chunki bloklar batch qilinib oddiy cuDNN'dan o'tadi.
- Halo qismini takror hisoblash isrofi ham yo'qoladi.

**Isbotlar:**
- `sarbon/models/block_local.py` + `tests/test_block_local.py` (CPU): sparse yangilash natijasi
  **bitma-bit** bir xil (xato 0.0), kodlangan bloklar soni = o'zgargan bloklar soni.
- `bench/analyze_gating.py` (CPU, FLOP bilan og'irlangan GPU ishi, 1080p):

| Sahna | O'zgargan piksel | Hozirgi (block-level) | Cell-level 8 px (aniq) | **Halo-zero** |
|---|---|---|---|---|
| tinch | 0.24% | 2.6% | 1.9% (−29%) | **0.6% (4.3x kam)** |
| o'rta | 1.32% | 9.8% | 7.4% (−25%) | **2.6% (3.8x kam)** |
| band | 16.1% | 38.2% | 33.2% (−13%) | **21.6% (1.8x kam)** |

**Ochiq savol:** blok chegaralari aniqlikka qancha zarar qiladi? Buni faqat o'qitish ko'rsatadi.
- Tajriba tayyor: `experiments/halo_zero_ablation.py`. Bitta CenterNet-uslub detektor ikki xil stem bilan
  o'qitiladi, qolgan hamma narsa bir xil.
- **Qaror qoidasi:** AP pasayishi ≤ 1.0 va AP_small pasayishi ≤ 1.5 bo'lsa, halo-zero qabul qilinadi.
  Aks holda cell-level gating bilan halo engine'da qolinadi (u ham aniq, ham 25–29% tejaydi).

### Ma'lumotlar (litsenziya, tijoriy foydalanish)

- **Pretraining:**
  - Open Images V7: annotatsiyalar CC BY 4.0, rasmlar CC BY 2.0. Har rasmning attributsiyasi saqlanadi.
  - COCO: faqat CC BY, CC BY-SA, "no known copyright" va US Gov litsenziyali rasmlar.
    `sarbon/data/coco.py` buni avtomatik filtrlaydi.
- **Statik kamera fine-tune/eval:**
  1. NVIDIA PhysicalAI-SmartSpaces: CC BY 4.0, sintetik, ~1800 statik kamera, track ID'lari bor.
  2. MEVA: CC BY 4.0, haqiqiy statik kuzatuv kameralari, EO va IR, kichik obyektlar.
  3. VIRAT: tijoriy foydalanish ruxsat etilgan, lekin Kitware kelishuvni bekor qilish huquqini saqlaydi.
     Undan foydalanishdan oldin yozma tasdiq olish kerak.
  4. O'zingizning kameralaringiz.
- **Ishlatmaymiz:** MOT17/20, CrowdHuman, Objects365, SA-1B, DanceTrack, SportsMOT, SODA, AI City
  (haqiqiy qismi), MOTSynth/GTA, UA-DETRAC, WILDTRACK.
- **Sintetik data** yaxshi yo'l: statik kamera eng oson holat.
  - Vosita: Blender (chiqish sizniki) yoki Isaac Sim.
  - Asset'lar: CC0 yoki sotib olingan bo'lishi kerak.
  - Unreal/Epic asset'larini AI o'qitishda ishlatmaymiz (EULA cheklovlari).

### Kodek orqali "tekin" gate (H.264/H.265)

- "Skip blok = o'zgarmagan" qoidasi **faqat shartlar bajarilganda** bitma-bit to'g'ri:
  - MV aynan (0,0) bo'lishi kerak. H.264 P_Skip'da MV bashorat qilinadi va doim ham nol emas.
  - Reference oldingi kadr bo'lishi kerak.
  - Residual bo'lmasligi kerak.
  - Weighted prediction o'chiq bo'lishi kerak.
  - HEVC'da shu CTB uchun SAO o'chiq bo'lishi kerak.
  - Deblocking qo'shni bloklar chegarasidan ~3 px gacha ta'sir qiladi, shuning uchun maskani 4 px
    dilatsiya qilish kerak.
- **Hikvision H.265+ tuzog'i:** skip "fon kadriga teng" degani bo'lishi mumkin (long-term reference).
  Shuning uchun reference index'ni tekshirish shart.
- **NVDEC MV yoki MB turlarini bermaydi.** FFmpeg software dekoder MV'larni faqat H.264 uchun eksport
  qiladi va unda skip yoki cbp bayrog'i yo'q. Demak, FFmpeg'ga kichik patch kerak bo'ladi.
- **Xulosa:** NVDEC'dan keyin GPU'dagi piksel farqi deyarli tekin, ya'ni o'tkazuvchanlik muammo emas.
  Kodekning asosiy foydasi chegarasiz aniqlik.
  - Birinchi qadam: kameralaringizda kunduzi va kechasi skip ulushini o'lchash.
  - Ulush kechasi past bo'lsa (shovqin), foyda kichik bo'ladi.
  - Ustuvorligi past, M6'dan keyin.

### Yangi, sinaladigan g'oyalar (prior art bo'yicha tekshirilgan)

Eng yaqin oldingi ishlar: CaTDet (1810.00434, tracker ROI tanlaydi) va FOVEA (2108.12102). Ular
albatta iqtibos qilinishi kerak. Exact delta-sparse conv + track-query DETR + track feedback
kombinatsiyasi topilmadi.

1. **Halo-zero stem** (yuqorida). *Test:* ablation, mezon AP ≤ −1.0.
2. **Staleness-age embedding.** Har blok uchun "oxirgi qayta hisoblangandan beri o'tgan kadrlar" soni
   stride-16 tokenlarga va cross-attention'ga beriladi (StreamPETR'ning motion-aware LN uslubida).
   *Test:* tau > 0 holatida dense va stale o'rtasidagi AP farqining ≥50%ini yopishi kerak.
3. **O'rganiladigan staleness-kritik.** Kichik MLP blokdagi o'zgarish, blok yoshi, eng yaqin track'gacha
   masofa va track kovariansiyasi asosida detection loss qancha oshishini bashorat qiladi. Bloklar
   FLOP byudjetigacha ochko'z (greedy) tanlanadi.
   *Test:* teng byudjetda oddiy threshold'dan HOTA va AP_s yuqori bo'lishi kerak.
4. **Track-query decoder early-exit.** Track'ning box hududida qayta hisoblangan blok bo'lmasa va
   harakat dispersiyasi past bo'lsa, track query faqat oxirgi decoder qatlamidan o'tadi.
   *Test:* decoder FLOP'lari 50% kam bo'lganda HOTA ≤ 0.5 pasayishi kerak.
5. **Track-driven NWD + zichlik kvotasi.** Hungarian matching cost'iga NWD qo'shiladi. Kichik track'lar
   yonida stride-4/8 tokenlar uchun query slotlari zaxiralanadi. DETR matching'da NWD ishlatilgan ish
   topilmadi.
   *Test:* AI-TOD-v2 va VisDrone-MOT'da AP_vt va AP_t.

Approximate rejim uchun o'qitish retsepti:
- Reference'ga nisbatan gating qilinadi (oldingi kadrga emas, Eventful uslubida), shunda drift to'planmaydi.
- O'qitishda staleness aniq simulyatsiya qilinadi: t−k va t kadrlar feature'lari deploy qoidasi bilan
  splice qilinadi, `tau` har batch'da tasodifiy tanlanadi.
- Stride 16'da distillyatsiya loss'i qo'shiladi.

Treking uchun:
- MOTIP (Apache-2.0, ID classification) asos sifatida eng mos.
- CO-MOT'ning COLA usuli va MATR'ning motion update'i arzon yutuq beradi.
- Track'ning tug'ilishi va o'limi uchun histerezis bilan ikki chegara ishlatiladi.

### Sarbon detektori: birinchi to'liq versiya (CPU'da tekshirilgan)

`sarbon/models/sarbon_det.py` modeli quyidagi qismlardan iborat:
- **Stem:** halo-zero `BlockLocalStem`.
- **Dense yadro:** stride 16/32 bosqichlari, AIFI attention (faqat stride 32'da), top-down va
  bottom-up fusion. Fusion yengil: 1×1 conv, keyin 3×3 depthwise.
- **Query tanlash:** RT-DETR uslubida, encoder top-K bo'yicha.
- **Decoder:** box-relative multi-scale deformable attention (`deform_attn.py`, sof PyTorch) va
  iterativ box refinement. NMS ishlatilmaydi.
- **Track query'lar:** oldingi kadrning `embed`/`boxes` chiqishlari keyingi kadrga query bo'lib qaytadi.

`criterion.py`:
- Hungarian matcher: cost = focal + L1 + GIoU, NWD opsional.
- Track query'lar Hungarian'ga kirmaydi, o'z ID'siga qat'iy biriktiriladi (MOTR uslubi).
- Loss: focal + L1 + GIoU. Har decoder qatlamida va encoder proposal'larida hisoblanadi.

| Model | Parametr | Decoder | Query |
|---|---|---|---|
| Sarbon-N | 3.9M | 3 qatlam, d=128 | 150 |
| Sarbon-S | 12.8M | 4 qatlam, d=256 | 300 |
| Sarbon-M | 24.9M | 6 qatlam, d=256 | 300 |

`tests/test_sarbon_det.py` (CPU) natijalari:
- Forward va backward ishlaydi.
- Kichik batch yodlab olindi: loss 23 → 0.3, har obyekt IoU 0.95–0.98.
- Track query qadami: ID'lar saqlanadi.

Hali yo'q, keyingi navbatda: D-FINE FDR regressiya, kontrastiv denoising (CDN), DEIM Dense O2O,
foveal yo'l, klip bo'yicha video o'qitish va staleness simulyatsiyasi.

### Kechki GPU sessiyasi rejasi (GPU bo'shagach)

1. `tests/test_fused_engine.py` va `tests/test_block_local.py` ishga tushiriladi.
2. Halo-zero ablation o'tkaziladi. Dataset kerak: COCO 2017 (~19 GB, foydalanuvchi ruxsati bilan
   yuklanadi). 6 klass, faqat tijoriy litsenziyali rasmlar, 640 px, 24 epoxa, ikkala variant uchun.
3. Halo-zero cache'ni GPU'da o'lchash: `nonzero_static` + CUDA graph, 16 kamera, M1 bilan solishtirish.
4. Cell-level gating fused engine'ga qo'shiladi (halo-zero qabul qilinmasa).

## 6. Yo'l xaritasi (har bosqichda o'lchanadigan mezon)

| Bosqich | Ish | O'tish mezoni |
|---|---|---|
| M0 ✅ | Lossless temporal block cache + bounded-budget CUDA graph | xato < 1e-5, kam o'zgarishda > 1x |
| M1 ✅ | Triton fused yadro, mayda byudjet, dense fallback, 16 kamerali batching benchmarki | ≤5% o'zgarishda ≥3x, eng yomon holat ≥1x. 16 kamerada bajarildi (6.8x / 2.1x). 1 kamerada qisman (1.1–1.4x / 0.96x), 5b bo'limiga qarang |
| M2 | Baseline: D-FINE/DEIM-S (faqat COCO og'irliklari) + ByteTrack sizning datasetingizda | AP, HOTA, FPS raqamlari |
| M3 | Sarbon backbone (lokal sparse + dense yadro), staleness bilan o'qitish | Dense versiya bilan AP farqi < 0.3 |
| M4 | Track-query decoder + track-steered gating | Treking sifati (HOTA) baseline'dan past emas, yana tezroq |
| M5 | Foveal kichik-obyekt yo'li | Kichik obyektlarda AP_s +3, xarajat < 20% |
| M6 | C++/TensorRT runtime, Jetson sinovi | Real kamerada end-to-end kechikish |

## 7. Ochiq savollar (foydalanuvchidan)

- Deploy qurilmasi: server GPU (nechta kamera?), Jetson yoki CPU?
- Qaysi obyektlar va klasslar aniqlanadi? Kamera rezolyutsiyasi va FPS qanday?
- O'zingizning datasetingiz bormi? Bo'lmasa, statik kamerali ochiq datasetlar ko'rib chiqiladi
  (MOT17/20 ning statik qismlari, VIRAT, AI City). Har birining litsenziyasi tekshiriladi.
