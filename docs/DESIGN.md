# Sarbon — o'zgarishga qarab hisoblaydigan detektor + treker

> Fayl yo'llari 0.2.0 da o'zgardi (CHANGELOG.md). Quyidagi tarixiy yozuvlardagi eski yo'llar:
> `experiments/train_sarbon.py` → `sarbon/cli/train.py` (`sarbon-train`), `scripts/run_camera.py` →
> `sarbon/cli/run.py` (`sarbon-run`), `bench/compare_own.py` → `sarbon/cli/compare.py`,
> `sarbon/pipeline.py` → `sarbon/inference/video.py`, `scripts/*.py` → `tools/`, `scripts/*.ps1` → `jobs/`.

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

**Ablation natijasi (2026-10-29, RTX 5090):**
- Sinov detektori: CenterNet uslubi.
- Data: COCO, 6 klass, faqat tijoriy litsenziyali 18 143 rasm. Baholash val2017'ning 2 968 rasmida.
- O'qitish: 24 epoxa, 640 px, noldan.

| Epoxa | Oddiy AP | Halo-zero AP |
|---|---|---|
| 8 | 5.62 | 4.45 |
| 16 | 11.00 | 8.64 |
| 24 | **11.90** (AP_s 3.38) | **9.01** (AP_s 2.46) |

**Qaror: halo-zero sof ko'rinishda RAD ETILDI.** AP farqi −2.89, mezon esa ≤ −1.0 edi. Farq o'qitish
oxirigacha ham yopilmadi: 8-epoxada −1.2, 24-epoxada −2.9. Demak blok chegaralarida yo'qotiladigan
kontekst haqiqiy zarar keltiradi va bu shunchaki sekin konvergensiya emas.

Shunga ko'ra:
- Sarbon oddiy stem bilan o'qitiladi (`--no-halo-zero`).
- Temporal sparsity M1 halo engine'i orqali beriladi (aniq), cell-level gating esa yana 25–29% tejaydi.

Halo-zero'ni qutqarish g'oyalari (keyinroq sinash mumkin):
- (a) blokka faqat 4–8 px ustma-ust tushish (overlap) qo'shish;
- (b) blok chegaralarini kadrlar orasida siljitish (shifted blocks, Swin uslubi);
- (c) halo-zero'ni faqat stride 2/4 da qo'llab, stride 8'ni oddiy qoldirish.

### Treking: Ultralytics'dan o'tish (2026-10-29)

**Baseline (Ultralytics `main` manbasidan, faqat faktlar uchun o'qildi):**
- ByteTrack sozlamalari: high 0.25, low 0.1, new 0.25, buffer 30 kadr, match 0.8.
- BoT-SORT'da ReID standart holatda o'chiq. Yoqilganda ham faqat IoU ≥ 0.5 bo'lgan joyda ishlaydi.
- Bufer 30 kadr, ya'ni taxminan 1 soniya.

Demak uzoq yashiringan odam doim yangi ID oladi. Statik kamerada bu eng ko'p uchraydigan xato: ustun,
daraxt yoki mashina orqasidan o'tish.

**`sarbon/track/tracker.py`:** o'z kodimiz, g'oyalar MIT litsenziyali maqolalardan. ByteTrack mantig'i
asosida, har bir yaxshilanish alohida yoqiladi: NSA Kalman, OC-SORT ORU/OCR/OCM, embedding va
**uzoq muddatli xotira re-ID**. Re-ID: yo'qolgan track 300 kadrgacha saqlanadi va embedding, harakat
ehtimoli (|v|·vaqt) hamda o'lcham bo'yicha qayta topiladi. IoU talab qilinmaydi.

**`bench/mot_sim.py`:** sintetik statik kamera MOT benchmarki.
- Sahna: odamlar kadrni kesib o'tadi, bir-birini to'sadi, 120–220 px kenglikdagi to'siqlar orqasida
  yashirinadi.
- Shovqin: yolg'on aniqlashlar, bir-biriga o'xshash kiyimdagi odamlar guruhlari.
- Metrikalar: MOTA, IDF1, IDsw. 3 ta sahna, har biri 600 kadr.

| Treker | MOTA | IDF1 | IDsw | ID (GT 26) |
|---|---|---|---|---|
| ByteTrack (Ultralytics standarti) | 63.9 | 62.9 | 21 | 43 |
| + BoT-SORT uslubidagi embedding | 63.9 | 62.9 | 21 | 43 |
| + OC-SORT ORU / OCR / NSA | ≈ teng | ≈ teng | — | — |
| **+ uzoq xotira re-ID (Sarbon)** | **64.1** | **70.9** | **15** | **35** |
| re-ID + ORU + NSA | 64.0 | 67.5 | 20 | 39 |

Bu natija sintetik sahnada olingan. Haqiqiy isbot uchun MOT20 (statik kameralar), MEVA va
TrackEval HOTA kerak. Ultralytics bilan ikki holatda solishtiriladi: standart sozlamalari bilan
**va** sozlangan buferi bilan.

**Yer tekisligi, chuqurlik va rang (2026-10-29):**
- `sarbon/track/ground.py`: **o'z-o'zini kalibrlaydigan yer tekisligi.** To'silmagan, ishonchli
  track'lardan `h = a·(y − y0)` bog'liqligi robust (Huber) usulda o'rganiladi. Natijada gorizont y0
  va kamera balandligi Hc = 1.7 / a topiladi. Undan har bir odamning yerdagi joyi (X metr) va
  kameradan masofasi Z (**chuqurlik**) chiqadi, chuqurlik tarmog'i kerak emas. Harakat metrlarda,
  shovqin kamera orqali propagatsiya qilinadi (UCMCTrack uslubi). Re-ID ekstrapolyatsiyasi ham yerda.
- `sarbon/track/appearance.py`: **rang deskriptori.** Gavdaning yuqori va pastki yarmining HSV
  gistogrammasi, Bhattacharyya masofasi bilan solishtiriladi.
- Simulyator endi haqiqiy kamera geometriyasiga ega: balandligi 4 m, fokus 1500 px, odamlar 7–45 m
  masofada 1.0–1.6 m/s tezlikda yuradi. Har kimning kiyim rangi bor, 8 rangli palitra bo'lgani
  uchun ranglar takrorlanadi.

| Treker (sintetik, 3 sahna, GT 26 kishi) | MOTA | IDF1 | IDsw | ID'lar |
|---|---|---|---|---|
| ByteTrack (Ultralytics standarti) | 57.0 | 42.8 | 70 | 64 |
| re-ID: embedding | 57.4 | 54.9 | 52 | 44 |
| re-ID: faqat rang | 58.1 | 67.6 | 33 | 30 |
| re-ID: embedding + yer tekisligi | 57.2 | 57.5 | 51 | 45 |
| **re-ID: rang + yer tekisligi** | **58.1** | **70.6** | **28** | **30** |

Kalibrlash: treker gorizontni **129 px** deb topdi (haqiqiysi 150), kamera balandligini esa
**4.26 m** (haqiqiysi 4.00).

Saboq: yer tekisligi IoU o'rniga asosiy xarajat sifatida ishlatilganda **zarar qildi**
(FP 1 → 144). Uzoq masofada chuqurlik noaniqligi bir necha metrga yetadi va soxta aniqlashlar ham
"mumkin" bo'lib qoladi. U darvoza va jarima sifatida ishlatilganda foyda beradi.

Cheklov: bu simulyatorda embedding ataylab shovqinli qilingan (hali ID loss bilan o'qitilmagan),
rang esa informativ. Haqiqiy sahnada yorug'lik rangni o'zgartiradi. Og'irliklar MOT20/MEVA'da
qayta sozlanadi.

**To'liq video pipeline (`sarbon/pipeline.py`, 2026-10-29):**
- O'qitilgan stem BN'i yig'ilib `LocalBackbone`'ga aylantiriladi (`sparse/convert.py`, xato 1.5e-6).
  U fused sparse engine'da ishlaydi va detektor tayyor p8 xaritasini qabul qiladi.
- Har kamera uchun alohida `SarbonTracker` ishlaydi: embedding, rang va yer tekisligi bilan.
- **Track-steered gate** (`engine.set_extra`): `tau>0` bo'lganda track'larning bashorat qilingan
  bloklari har kadrda majburan qayta hisoblanadi. Fon keshda qoladi, obyektlar yangi bo'ladi.
- GPU testi: sparse pipeline 12 kadr davomida dense stem bilan bir xil feature beradi
  (fp16'da 1.6e-3). Kechikish hali o'lchanmagan: GPU o'qitish bilan band.

**Re-ID o'qitish (`experiments/train_track.py`):**
- Detektorda `id_head` bor: normallashtirilgan 128 o'lchamli embedding.
- Klip bo'yicha SupCon loss (`criterion.id_contrastive`) ishlatiladi.
- MOTChallenge formatidagi klip loader (`data/mot.py`) bor.
- Sintetik smoke test: ID loss 3.6 → 0.01, ko'rilmagan videolarda retrieval top-1 0.63 → 0.79.
  Keyin pasaydi: 48 ta ID'ni yodlab olish. Haqiqiy natija uchun MEVA yoki SmartSpaces kerak.

**Keyingi treking qadamlari (tadqiqot bo'yicha tartiblangan):**
1. Decoder embedding'ini ID loss bilan o'qitish (QDTrack yoki MOTIP uslubi). EMA'da confidence og'irligi
   ishlatiladi (Deep OC-SORT).
2. **Yer tekisligida treking** (UCMCTrack, MIT): MOT17 val'da IDF1 77.1 → 82.2. Kamera harakati
   kompensatsiyasi o'chiriladi, chunki u statik kamerada zarar qiladi.
3. **Yangi g'oya: o'z-o'zini kalibrlaydigan yer tekisligi.** Ishonchli track'lardan odam bo'yi va oyoq
   satri bog'liqligi o'rganiladi, undan gorizont va homografiya chiqariladi. Bu UCMC'ni qo'lda
   kalibrlashsiz ishlatish imkonini beradi.
4. DETR uchun past ishonchli dublikatlarni stage 2 dan oldin bostirish. Faol track'ga yopishgan
   tug'ilishlarni bloklash (TrackTrack TAI). Track'ni 2–3 hit'dan keyin tasdiqlash.
5. Har bir hujayra uchun sahna prior'larini online o'rganish: tug'ilish/o'lim zonalari, tezlik
   va o'lcham prior'i, takrorlanuvchi yolg'on aniqlashlar banki.

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
| Sarbon-N | 4.0M | 3 qatlam, d=128 | 150 |
| Sarbon-S | 13.1M (`dw` blok bilan 10.2M) | 4 qatlam, d=256 | 300 |
| Sarbon-M | 25.2M | 6 qatlam, d=256 | 300 |

`tests/test_sarbon_det.py` (CPU) natijalari:
- Forward va backward ishlaydi.
- Kichik batch yodlab olindi: loss 23 → 0.3, har obyekt IoU 0.95–0.98.
- Track query qadami: ID'lar saqlanadi.

**v1 (shu kuni qo'shildi, barchasi o'z kodimiz, g'oyalar Apache litsenziyali maqolalardan):**
- **FDR (D-FINE):** har box chegarasi 33 bin'li distribution sifatida beriladi. Bin qiymatlari 0 atrofida
  zich, chetlarda siyrak. Har decoder qatlami oldingi qatlam logit'lariga qoldiq qo'shadi.
  Bin bo'yicha kodlash va dekodlash aylanasi aniq tekshirildi.
- **GO-LSD:** oxirgi qatlam distribution'lari oldingi qatlamlarga KL distillyatsiya qilinadi (DDF loss).
- **MAL (DEIM):** musbat target = IoU^1.5. Boxi yaxshi query'ning ishonchi ham yuqori bo'ladi.
- **Kontrastiv denoising (CDN):** GT'dan shovqinli musbat va manfiy query'lar yasaladi. Attention
  niqobi qo'yiladi: guruhlar bir-biridan ajratilgan, matching query'lar denoising query'larni ko'rmaydi.
  Tartib: `[denoising | track | yangi]`.
- **Mosaic augmentatsiya** (DEIM Dense O2O): bitta qadamda ko'proq musbat. Oxirgi epoxalarda o'chiriladi.
- **`experiments/train_sarbon.py`:** EMA, bf16, cosine LR, DETR top-K postprocess, COCO eval.
  Ikki rejimi bor: tijoriy (6 klass, tijoriy litsenziyali rasmlar) va `--benchmark` (COCO-80,
  faqat tadqiqot uchun solishtirish).
- **Arxitektura variantlari** (config orqali):
  - `block_type`: `res`, `rep` yoki `dw`.
    - `rep` = RepVGG: o'qitishda 3 tarmoq, inference'da bitta 3×3 conv. Fusion aniqligi 5.5e-6
      ichida, halo-zero xossasi saqlanadi.
    - `dw` = inverted bottleneck: Sarbon-S 10.2M parametr (`res` da 13.1M).
  - `act`: `relu`, `silu` yoki `gelu`.

Testlar (CPU): `test_sarbon_det.py` (overfit, IoU 0.90–0.98, denoising bilan) va `test_arch_variants.py`.

### "YOLO'dan kuchli" ekanini qanday isbotlaymiz (protokol)

Da'vo faqat bir xil sharoitdagi o'lchov bilan qilinadi:
1. **Aniqlik:** `train_sarbon.py --benchmark` bilan COCO-80 val2017, 640 px. Natija YOLO11/YOLO26
   (s, m) va D-FINE/DEIM'ning e'lon qilingan AP raqamlari bilan solishtiriladi.
2. **Tezlik:** bir xil GPU (5090), TensorRT FP16, batch 1 va batch 16, end-to-end vaqt (postprocess
   bilan). YOLO'ni o'zimiz o'lchaymiz: Ultralytics'ni faqat lokal benchmark uchun ishga tushirish AGPL
   bo'yicha mumkin, chunki kod tarqatilmaydi va bizning kodimizga qo'shilmaydi.
3. **Sarbon'ning asosiy maydoni:** statik kamera videosi (MEVA/SmartSpaces). Ko'p kamerali serverda
   "kamera/GPU" va "kadr/s" o'lchanadi. Bu yerda temporal sparsity ustunlik berishi kerak.
4. **Kichik obyektlar:** AP_small va AI-TOD uslubidagi tiny-obyekt metrikalari.

**Halol cheklov:** YOLO, D-FINE va DEIM backbone'lari ImageNet'da yoki Objects365'da oldindan
o'qitilgan. Bizda hozircha pretraining yo'q. ImageNet esa tijoriy foydalanish uchun ruxsat bermaydi.

Reja: DINOv2 (Apache-2.0) feature'laridan stem va dense yadroga distillyatsiya qilish. Bu tijoriy
litsenziyali rasmlarda (Open Images, filtrlangan COCO) yorliqsiz amalga oshiriladi. Shu qilinmaguncha
qisqa jadvalda (72 epoxa) natija pastroq bo'lishi kutiladi.

**Pretraining (`experiments/pretrain_distill.py`):**
- Teacher: DINOv2 (Apache-2.0), muzlatilgan holda.
- Student: Sarbon stem va dense yadro.
- Student'ning stride-16 feature'lari teacher patch tokenlariga cosine loss bilan yaqinlashtiriladi.
- Yorliq kerak emas: barcha tijoriy litsenziyali COCO rasmlari, keyinchalik Open Images ishlatiladi.
- Natija `train_sarbon.py --init` orqali yuklanadi.
- Soxta teacher bilan CPU smoke test o'tdi. Haqiqiy DINOv2 og'irliklarini (ViT-S/14, ~90 MB)
  yuklash uchun ruxsat kerak.

**O'z kameralarimizdan o'z-o'zini o'qitish (tashqi og'irliksiz, `data/pseudo.py`):**

Statik kamera fonni soatlab ko'radi, shuning uchun harakatlanayotgan narsa yorliqsiz ham ajralib turadi.
1. MOG2 fon modeli (OpenCV) fonni o'rganadi va foreground blob'larni ajratadi; soyalar chiqarib
   tashlanadi.
2. Blob'larga o'z trekerimiz ID beradi.
3. Faqat uzoq yashagan **va** siljigan track'lar saqlanadi: shamolda tebranayotgan daraxt va
   miltillash o'tib ketmaydi.
4. Qisqa bo'shliqlar interpolatsiya qilinadi.
5. **Yorliqsiz qolgan harakatlanuvchi blob'lar fon tasviri bilan bo'yaladi.** Shunda model haqiqiy
   obyektni "fon" deb o'rganmaydi.
6. Natija MOTChallenge formatida yoziladi (`scripts/make_pseudo_mot.py`), so'ng `train_track.py`
   (obyektlik va re-ID) ishga tushadi, oxirida `train_sarbon.py --init` bilan COCO fine-tune qilinadi.

Sifat `bench/pseudo_label_eval.py` bilan o'lchandi. Sintetik video: harakatlanuvchi obyektlar,
shovqin, yorug'lik o'zgarishi va tebranuvchi "daraxt". Faqat o'qitishga yoziladigan kadrlar
hisobga olindi.

| Variant | Precision | Ko'rinadigan recall | ID tozaligi | Ko'rinadigan, yorliqsiz |
|---|---|---|---|---|
| Faqat track'lar | 0.864 | 0.529 | 0.963 | 47% |
| + interpolatsiya | 0.863 | 0.533 | 0.963 | 47% |
| **+ fon bilan bo'yash** | **0.863** | **0.627** | **0.963** | **32%** |

Qolgan yorliqsiz obyektlarning asosiy sababi: ustma-ust tushgan obyektlar bitta blob'ga
birlashadi. Haqiqiy kameradagi ko'rsatkichlar boshqacha bo'ladi va ular foydalanuvchi
yozuvlarida o'lchanadi.

Butun zanjir sinaldi: video fayl → psevdo-dataset → o'z-o'zini o'qitish → `--init`.

**Pretraining strategiyasi, uchta variant o'lchov bilan solishtiriladi:**
1. Scratch.
2. DINOv2 distillyatsiya (navbatda).
3. O'z kameralarimizdan o'z-o'zini o'qitish (foydalanuvchi yozuvlari kerak).

### Arxitektura qidiruvi (GPU'da, qisqa jadval, bir xil byudjet)

| Savol | Variantlar | Mezon |
|---|---|---|
| Halo-zero aniqlikka zarar qiladimi? | standard / halo | AP ≤ −1.0 (navbatda) |
| Qaysi blok? | res / rep / dw | AP va TensorRT kechikishi (Pareto) |
| Qaysi aktivatsiya? | relu / silu / gelu | AP; SiLU/GELU Triton yadrosiga epilog sifatida qo'shiladi |
| Kichik obyekt matching | NWD 0 / 1 / 2 | AP_small |
| Decoder chuqurligi va query soni | 3/4/6 qatlam, 150/300 | AP va kechikish |

Hali yo'q, keyingi navbatda: DINOv2 distillyatsiya pretraining, foveal yo'l, klip bo'yicha video
o'qitish va staleness simulyatsiyasi.

### Sarbon-S natijalari va fusion ablation (2026-10-29/30, o'lchangan)

COCO val2017, 6 kuzatuv klassi, 640 px; train: faqat tijoriy litsenziyali ~18k rasm, 72 epoxa.

| Model | Eng yaxshi AP (epoxa) | AP50 | AP_small | AP_large | 72-epoxa AP |
|---|---|---|---|---|---|
| Sarbon-S scratch | 36.4 (60) | 55.5 | 16.1 | 50.9 | 36.2 |
| Sarbon-S + DINOv2 distill init | **39.0 (52)** | **59.6** | 16.3 | **56.3** | 38.0 |

- Distillyatsiya **+2.6 AP** beradi va bir xil AP'ga ~2.5x kam epoxada yetadi.
- Distill run 52-epoxadan keyin pasaydi (39.0 → 38.0): 18k rasmda overfitting. Ma'lumot
  ko'paymaguncha ~50 epoxa yetarli. Asosiy cheklov endi arxitektura emas, ma'lumot.
- AP_small deyarli o'zgarmadi (16.1 → 16.3): DINOv2 stride-16 xususiyatlari kichik obyektga yordam bermaydi.

Neck fusion ablation (scratch, 24 epoxa, bitta seed):

| Fusion | AP | AP50 | AP_small | AP_large |
|---|---|---|---|---|
| concat | **30.2** | **47.6** | 11.0 | **43.3** |
| weighted (BiFPN) | 29.6 | 46.1 | **12.9** | 42.3 |
| gated | 29.8 | 46.7 | 11.2 | 42.6 |

- Umumiy AP farqi (≤0.6) bitta seed shovqini ichida: concat default qoladi.
- weighted AP_small'da +1.9 beradi, bu kuzatuv kameralari uchun muhim. Ko'p ma'lumotli
  o'qitishda (multi-dataset) qayta tekshiriladi.

### To'liq COCO benchmark (faqat ilmiy o'lchov, 2026-10-30 dan)

YOLO bilan teng sharoitda solishtirish uchun Sarbon-S to'liq COCO train2017'da (117k rasm, 80 klass)
o'qitiladi: DINOv2 distill init, 72 epoxa, `runs/sarbon/s_bench_distill`. To'liq COCO'da NC litsenziyali
rasmlar bor, shuning uchun **bu og'irliklar mahsulotga kirmaydi**, ular faqat arxitekturani o'lchaydi.
Natija jadvalda init va epoxa bilan birga beriladi (YOLO11s: scratch, 600 ep; D-FINE-S: ImageNet, 120+ ep).
Yakuniy raqam rasmiy pycocotools eval'da (asl koordinatalar, crowd = ignore) qayta o'lchanadi.

**Natija (2026-11-01, ichki eval, letterbox koordinatalarda):** eng yaxshi AP **42.6** (ep52), oxirgi
(ep72) 42.6; AP50 59.7, AP75 45.7, APs 20.4, APm 39.4, APl 57.1. Ep48 dan keyin plato, overfit yo'q.
YOLO11s 47.0 / D-FINE-S 48.5 dan **~4.5–6 AP orqada**, asosiy farq kichik obyektlarda. Rasmiy eval hali
qilinmagan.

### RailSafe o'z ma'lumoti: qisman yorliqlar bilan o'qitish (2026-10-30)

Foydalanuvchining `pereezd_dataset` (YOLO format, 8 413 kadr 2560x1440, 47 video, 2 klass:
yengil 42.5k / og'ir 2.8k box) ko'rib chiqildi:
- Asl val qismidagi 41 ketma-ketlikning hammasi train'da ham bor edi (kadrlar 0.2 s farq): val
  sun'iy baland chiqardi. `scripts/make_yolo_split.py` videolar bo'yicha qayta bo'ladi:
  train 6 722 (38 video), val 1 691 (9 video, og'ir box'larning 36%i). Asl ma'lumotga tegilmaydi,
  NVMe nusxa: `C:\data\pereezd_dataset`, ro'yxatlar: `C:\data\pereezd_split`.
- Unda faqat transport belgilangan: piyoda va poyezd bor, lekin ramkasiz. Oddiy o'qitish ularni
  "fon" deb o'rgatardi. Yechim, **qisman yorliq (partial label)**: har bir rasm qaysi klasslarni
  belgilaganini bildiruvchi `valid` maska (`WithClassMask`), MAL/focal loss belgilanmagan klassga
  na musbat, na manfiy signal beradi (Federated loss / UniDet g'oyasi). Mosaic bitta manba ichida.
- RailSafe taksonomiyasi: person, bicycle, motorcycle, **yengil** (COCO car), **og'ir** (COCO bus +
  truck), **train**. Fine-tune tijoriy `s_distill/best.pt` dan; klass head qatorlari nom bo'yicha
  ko'chiriladi (car → yengil, truck → og'ir), train yangidan.
- Baholash ikki xil: COCO val (6 klass) va o'z val'imiz (faqat yengil/og'ir); eng yaxshi checkpoint
  o'z val'imiz bo'yicha tanlanadi. Navbat: `scripts/run_queue3.ps1` (benchmark tugagach, 24 epoxa).

**Natija (2026-11-01).** O'z val (1 691 kadr, yengil + og'ir), bir xil evaluator (`bench/compare_own.py`):

| Model | AP | AP50 | AP75 | APs | APm | APl |
|---|---|---|---|---|---|---|
| Sarbon-S (COCO 6 klass + o'z data, 24 ep, best ep6) | **74.1** | **88.1** | **81.7** | 44.5 | **80.4** | **86.1** |
| YOLO11s (COCO pretrained, faqat o'z data, 50 ep) | 69.3 | 83.0 | 76.2 | **57.2** | 75.9 | 78.2 |

- Sarbon +4.8 AP, lekin **kichik obyektlarda −12.7 AP**: bu eng katta zaif nuqta (COCO'dagi kabi).
- O'z val'da eng yaxshi natija ep6 da, keyin biroz pasaydi (73.1). COCO 6-klass AP 38.96 → 40.8
  (unutish yo'q).
- Tezlik **hali o'lchanmagan deb hisoblanadi**: eager PyTorch, batch 1 da RTX 5090 bir xil model uchun
  11–78 ms orasida sakraydi (CPU kernel-launch va GPU P-state'ga bog'liq). Adolatli o'lchov: ikkala model
  ONNX → TensorRT fp16, bir xil sharoitda.

### Kichik obyektdagi −12.7 qayerdan keladi? (2026-11-02, o'lchangan)

`bench/small_recall.py`: kichik GT box'larning klassga bog'liq bo'lmagan recall'i, o'z val, 640 px.
Qatorlar: `enc` = 300 ta encoder taklifi, `final` = barcha 300 dekoder box'i, `@0.3` = ball ≥ 0.3.

| IoU | Bosqich | 0–16 px | 16–32 px | 32–96 px | >96 px |
|---|---|---|---|---|---|
| 0.5 | Sarbon @0.3 | **88.2%** | **95.4%** | **98.3%** | **94.3%** |
| 0.5 | YOLO11s @0.3 | 71.9% | 92.6% | 96.5% | 86.3% |
| 0.75 | Sarbon @0.3 | **69.4%** | **85.1%** | **96.6%** | **92.7%** |
| 0.75 | YOLO11s @0.3 | 60.7% | 83.8% | 94.8% | 84.6% |
| 0.9 | Sarbon final / YOLO any | 18.6 / 16.6% | 57.2 / 54.4% | 89.9 / 88.2% | 90.8 / 80.8% |

- **Sarbon kichik obyektlarni YOLO'dan ko'proq va aniqroq topadi** (har IoU'da). Encoder tanlovi ham
  ularni yo'qotmaydi (IoU 0.5 da 99%). Demak −12.7 AP_small recall yoki lokalizatsiya muammosi emas.
- Ball ≥ 0.3 bo'lgan kichik detection'lar: Sarbon 74.0% GT bilan mos, **24.3% mos emas**; YOLO 82.9 / 15.5%.
- Mos kelmagan 48 ta Sarbon box'i qo'lda ko'rildi (`runs/small_fp_sheet.jpg`): taxminan **3/4 qismi
  haqiqiy, lekin belgilanmagan mashinalar** (devor ortida turgan, uzoqdagi, avtobus old qismi). Qolgani
  haqiqiy xato: vagondagi po'lat rulonlar "yengil", semafor ustunlari.
- Xulosa: o'z val'dagi AP_small asosan **yorliqlash siyosatini** o'lchaydi. YOLO faqat shu data'da
  o'qigani uchun "uzoq/turgan mashinani belgilama" qoidasini o'rgangan, Sarbon COCO'dan o'rgangani uchun
  ularni topadi va FP deb jazolanadi. Kerakli ish: (1) to'liq qayta belgilangan kichik test to'plami
  (barcha transport, piyoda, poyezd); (2) rulon/vagon uchun hard-negative yoki `train` yorliqlari.
- Arxitektura bo'yicha haqiqiy o'lchov COCO'da qoladi (to'liq yorliqli): Sarbon 42.6 va YOLO11s 47.0.
  Navbat `scripts/run_queue5.ps1` (24 ep, scratch, baseline 30.2 / APs 11.0): NWD matching va 2x backbone
  (18.5M, backbone 5.3M → 10.6M).

Yuqori kirish o'lchami o'qitishsiz yordam bermaydi: 640 da 74.1 AP, 960 da 61.7, 1152 da 38.5
(model 640 da o'qigan). Yuqori aniqlik faqat shu o'lchamda o'qitish bilan sinaladi.

**Sparse stem real videoda.** Pereezd yozuvlarida haqiqatan o'zgargan bloklar (tau 0.02–0.03, 1152x640):
busy kamerada 16–24%, tinchida 7–8%. Lekin oddiy (halo'siz) stem'ning receptive field'i 55 px, shuning
uchun engine o'zgarishni 2 blok radiusga yoyadi va ish 85–95% ga chiqadi, sparse foyda deyarli yo'q.
Bundan tashqari:
- `tau` normallashtirilgan birlikda solishtirilardi (amalda ~4x past). Endi `SarbonVideo` tau'ni rasm
  birligida (0..1) qabul qiladi.
- `calibrate()` chaqirilmagani uchun dense'ga qaytish ishlamasdi. Endi birinchi kadrda chaqiriladi.
- Stem umumiy vaqtning kichik qismi. Tezlik uchun asosiy ish dense core + dekoder (TensorRT) va
  treker (yo'qolgan treklar 300 kadr saqlanadi, vaqt 15 → 60 ms o'sadi).

### Tezlik: CUDA graph va TensorRT (2026-11-02, GPU o'qitish bilan band paytda)

Sarbon-S, 640x640, fp16: eager 31.6 ms → CUDA graph 2.1 ms → TensorRT 1.44 ms (1 kadr); 4 kadrda
TensorRT 0.51 ms/kadr. Wide backbone (18.5M) TensorRT'da 1 kadrda +3%, 4 kadrda +25% (0.63 ms/kadr).
TensorRT detektsiyalari PyTorch fp16 bilan 338/343 mos (IoU > 0.9). Demak katta GPU'da neyron tarmoq
endi to'siq emas: pipeline vaqti asosan CPU tomonda (resize, rang gistogrammasi, treker).

### RailSafe YOLO26n bilan solishtirish (2026-11-03)

RailSafe `models/pereezd_yolo26n.pt` (Ultralytics, 1088, 100 ep) `pereezd_dataset/train` ning hammasida
o'qigan. Bizning val'ning 1544/1691 kadri shu train'da, qolganlarining 0.2 s qo'shnilari ham. Shuning
uchun bizning val'da u 89.7 AP (Sarbon v0.1 74.1): bu yodlash, adolatli emas. Adolatli o'lchov oldingi
YOLO11s (bir xil split, 640): Sarbon 74.1, YOLO11s 69.3.

**Begona domen testi: Open Images V7 val** (`tools/openimages/`, 1000 rasm, faqat CC BY, rasmiy CVDF
bucket, `C:\data\openimages_val`). Car/Taxi/Limousine = yengil, Bus/Truck = og'ir. Van, Ambulance,
guruh va tasvir (depiction) box'li rasmlar chiqarilgan. Open Images qoidasi: sinf faqat u tasdiqlangan
rasmlarda baholanadi (yengil 994, og'ir 200 rasm). Rasmlar ~1024x683.

| Model | yengil AP | og'ir AP | o'rtacha |
|---|---|---|---|
| Sarbon v0.1 (640) | 66.1 | 79.9 | 73.0 |
| Sarbon smart (640) | 66.8 | 79.1 | 72.9 |
| Sarbon wide (640) | **67.8** | 79.3 | **73.6** |
| RailSafe YOLO26n @1088 | 15.1 | 7.6 | 11.4 |
| RailSafe YOLO26n @640 | 44.7 | 36.5 | 40.6 |

YOLO26n faqat o'z pereezd kameralarini biladi: yaqindan olingan va boshqa burchakdagi mashinalarda
yiqiladi. Cheklov: Open Images (Flickr) COCO'ga o'xshash domen, Sarbon esa COCO bilan o'qigan. Demak bu
"yangi pereezd kamerasida Sarbon yaxshi" degani emas; bu uchun mustaqil yorliqlangan pereezd kadrlari
kerak (foydalanuvchi yig'adi).

**YOLO26n sharoitida o'qitilgan Sarbon (2026-11-04).** Avval umumiy asos `coco_wide_person.yaml`
(wide + o'z kameralardagi odamlar, 50 ep). COCO AP 40.9 → 39.6 (-1.3), lekin o'z kameralardagi odam AP:
person val 43.3 → **71.8**, smart val 59.6 → **82.5**. Undan `railsafe_full_1088.yaml`: YOLO26n'ning
o'z train/val papkalari, asl yorliqlar, 1088, batch 8, 30 ep (8 soat). Eng yaxshisi oxirgi epoxada.

| Model | Pereezd val (YOLO26n'niki, 765) | yengil | og'ir | kichik | Open Images o'rtacha |
|---|---|---|---|---|---|
| **Sarbon full_1088 @1088** | **91.0** | **90.9** | **91.1** | **71.7** | **74.0** |
| RailSafe YOLO26n @1088 | 89.4 | 89.8 | 89.0 | 65.7 | 11.4 (640 da 40.6) |

Bir xil ma'lumot va o'lchamda Sarbon YOLO26n'dan pereezdda +1.6 AP (kichik obyektlarda +6.0) va begona
domenda +62.6 AP ustun. Model o'z o'lchamida ishlatilishi kerak: 1088 model 640 da Open Images'da 52.5,
640 da o'qigan wide 1088 da 35.8.

Tezlik (RTX 5090, 1088x1088, fp16, faqat tarmoq, GPU bo'sh; YOLO TensorRT 10.15, Sarbon 11.4):

| | Parametr | TRT b1 | TRT b6 (6 kamera) | PyTorch b1 | Sarbon CUDA graph b1 / b6 |
|---|---|---|---|---|---|
| Sarbon v0.3 | 18.5M | **2.57 ms** | 6.89 ms (1.15/kadr) | 30.7 ms | 4.0 / 18.9 ms |
| YOLO26n | 2.4M | 3.02 ms | **2.94 ms** (0.49/kadr) | 18.2 ms | — |

Bitta kadrda Sarbon tezroq (ikkalasi ham launch overhead chegarasida), 6 kamera batch'ida YOLO26n
~2.3x tezroq (7.8x kam parametr). Ikkalasi ham byudjetdan ancha past: 6 x 25 fps = 150 kadr/s, Sarbon
TRT ~870 kadr/s ko'taradi. sarbon-run hozir TRT emas, CUDA graph ishlatadi (b6 18.9 ms).

**Nega Sarbon og'ir (2026-11-04).** 1088 da Sarbon v0.3 238.7 GFLOPs/rasm, YOLO26n 16.7 (14x). TensorRT
qatlam profili (b6): stem s2–s8 **39%** (s4 ikki va s8 uch Res blok, 96/192 kanal, 272² va 136² xaritada:
sparse stem tezisi uchun yuqori rezolyutsiyada og'ir qilingan, u esa real videoda foyda bermadi),
Myelin "glue" (memory concat, anchor niqobi, LayerNorm 24k token ustida, top-k, deformable grid) 18%,
s16/s32 13%, qolgani neck, decoder va headlar. O'qitilmagan variantlar, TRT fp16, 1088, b6 (±0.5 ms shovqin):

| Variant | Param | GFLOPs | b6 ms |
|---|---|---|---|
| v0.3: widths 48/96/192, depths 2,3, c16 384, hidden 256, 300 q, 4 dec | 18.5M | 238.7 | 7.1–8.0 |
| A depths 1,1 | 17.0M | 165.0 | 6.2 |
| B + widths 32/64/128 | 16.2M | 128.0 | 5.2–5.4 |
| C = B + dw bloklar | 11.7M | 79.5 | 4.5–5.3 |
| D = B + c16 256 | 12.8M | 98.4 | 4.4–4.7 |
| F = D + 150 query, 3 dec | 11.6M | 93.3 | 4.2 |
| **G = F + hidden 192, ffn 768** | 8.7M | 77.6 | **3.5** |
| H = F + hidden 128 | 6.7M | 66.7 | 3.5 |
| I = H + dw | 4.1M | 36.3 | 3.6 |
| YOLO26n | 2.4M | 16.7 | 2.9 |

~3.4 ms dan pastda FLOPs kamaytirish yordam bermaydi: qolgan vaqt glue va kichik kernellar (decoder,
deformable attention, 3 darajali memory). YOLO26n darajasiga borish uchun ular ham qayta yozilishi kerak
(proposal faqat P16/P32 da yoki arzon head, doimiy anchorlar, kamroq daraja). Aniqlik o'qitishsiz
noma'lum: G to'liq zanjir bilan o'qitilishi kerak (distill ~1 soat, COCO base ~9 soat, pereezd 1088 ~8 soat).

G o'qitilmoqda (2026-11-04, `jobs\light_pipeline.ps1`): distill → `coco_light_person.yaml` →
`railsafe_light_1088.yaml`. Arxitektura CLI/YAML'da (`--hidden --heads --ffn --dec-layers --queries`)
va checkpoint shakllaridan qayta tiklanadi (`arch_from_state_dict`; query soni `n_queries` buferida).
Qabul qoidasi: v0.3 bilan bir xil val'da ≤ 1 AP pasayish va Open Images'da sezilarli yo'qotish yo'q.

G umumiy asosi (`s_light_person`, 640): best = 32-epoxa (o'z ma'lumotlari bo'yicha). COCO AP 34.0
(wide 39.6), person 68.1 (71.8), smart 80.2 (82.5), AP_small 12.2 (16.3). 43-epoxada elektr o'chib,
`ckpt.pt` yarim yozilgan holda qoldi; o'z AP'i 32-epoxadan keyin o'smagani uchun run qayta boshlanmadi.
Shundan keyin `train.py` checkpoint'ni fsync qilib yozadi va oldingisini `ckpt.prev.pt` sifatida saqlaydi.

**G natijasi (2026-11-05).** Pereezd 1088 (`railsafe_light_1088.yaml`, best = 28-epoxa), bir xil baholovchi:

| Model | Param. | YOLO26n val AP | small | yengil / og'ir | Open Images | TRT b1 | TRT b6 |
|---|---|---|---|---|---|---|---|
| Sarbon v0.3 | 18.5M | **91.0** | **71.7** | 90.9 / 91.1 | **74.0** | 2.5 ms | 7.9 ms |
| Sarbon G | 8.7M | 89.1 | 68.0 | 89.6 / 88.6 | 71.0 | **1.7 ms** | 3.7 ms |
| RailSafe YOLO26n | 2.4M | 89.4 | 65.7 | 89.8 / 89.0 | 11.4 | 3.0 ms | **2.9 ms** |

TensorRT fp16, 1088x1088, RTX 5090; v0.3 va G bitta jarayonda navbatma-navbat o'lchandi, YOLO26n
RailSafe env'ida (TRT 10.15) oldinroq. G v0.3 dan 2.1x tez (b6), lekin -1.9 AP: qabul qoidasi
(≤ 1 AP) bajarilmadi, shuning uchun v0.3 asosiy model bo'lib qoladi, G esa tezkor nomzod
(`weights/sarbon-s-railsafe-light-1088.pt`). YOLO26n bilan teng (-0.3 AP, kichik obyektlarda +2.3,
domen tashqarisida 71 ga qarshi 11), b1 da tezroq, b6 da 25% sekinroq.

**Pipeline CPU qismi (2026-11-05).** G bilan 6 kamerada qadamning ~80% i CPU'da edi: rang
gistogrammasi ~16 ms, treker ~17 ms, natijalarni yig'ish ~8 ms, neyron tarmoq esa ~5 ms. O'zgarishlar:
- rang faqat treker ishlatadigan box'lar uchun hisoblanadi (score >= `high`); 2-bosqich (past score,
  faqat IoU) trek rangini yangilamaydi, chunki bu box'lar ko'pincha to'silgan bo'ladi;
- katta box'lar gistogrammadan oldin ~64 px gacha siyraklashtiriladi;
- kadrlar qayta ishlatiladigan pinned buferga bitta nusxa bilan yoziladi (`np.stack` + `pin_memory` o'rniga);
- `inv2` slice'lardan `np.stack` qilmasdan to'ldiriladi; har kadrda CUDA tensor yaratilmaydi;
  label qidirish vektorlashtirildi.

A/B (`bench_step.py`, pereezd yozuvlari, G, CUDA graph, fonda o'qitish ishlab turgan holda):

| Kamera | Oldin | Keyin | Treklar/kadr (oldin / keyin) |
|---|---|---|---|
| 2 | 21.1 ms | **13.3 ms** | 21.28 / 21.25 |
| 3 | 29.6 ms | **17.0 ms** | 20.27 / 20.25 (noyob ID 192 / 192) |
| 6 | 52.2 ms | **31.2 ms** | 20.27 / 20.25 |

Keyingi qadamlar: GPU va CPU'ni ustma-ust ishlatish (bir qadam kechikish evaziga GPU kutishini
yashirish), trekerlarni alohida jarayonlarga chiqarish.

**G distillatsiyasi (o'qitilmoqda).** v0.3 o'qituvchi, psevdo-belgilar `tools/pseudo_label.py` bilan
(1088, score >= 0.5): pereezd to'plamiga odam, velosiped va mototsikl qo'shildi (`C:/data/pereezd_plus`:
10016 odam), yozuvlardan har 2 s da kadr olindi (`C:/data/pereezd_pseudo`: 2705 + 301 kadr). Poyezd sinfi
chiqarildi: o'qituvchi o'tayotgan poyezdning faqat ayrim vagonlarini topadi. Config
`railsafe_light_distill.yaml`, G'ning pereezd modelidan boshlanadi.

### Hisoblash scheduler'i (Perception Debt v0): o'lchangan, salbiy natija (2026-11-02)

Savol: kamera kadrini qachon hisoblash, qachon trekerning bashoratiga ishonish kerak? Sinov: 5 ta
pereezd yozuvi (`C:\data\recordings`, har biri 10 daqiqa = 12000 kadr), har kadr bir marta to'liq
hisoblangan (`tools/dense_reference.py`), siyosatlar oflayn simulyatsiya qilingan
(`bench/scheduler_eval.py`). O'lchov — har kadrni hisoblaydigan pipeline'ga sodiqlik (recall,
yangi obyekt kechikishi). Siyosatlar `sarbon/schedule/policies.py` da.

Bitta kamera, o'z chegarasi bilan (o'rtacha 5 kamera):

| Siyosat | Hisoblash | Recall | Odam recall | Kechikish p95 |
|---|---|---|---|---|
| har 2-kadr | 0.50 | 0.917 | 0.895 | 0.53 s |
| har 3-kadr | 0.33 | 0.841 | 0.824 | 1.08 s |
| harakat > 1 | 0.64 | 0.904 | 0.866 | 0.96 s |
| debt x1.5 | 0.65 | 0.929 | 0.907 | 0.49 s |
| debt x3 | 0.40 | 0.854 | 0.825 | 1.05 s |

5 kamera umumiy byudjetda (har taktda K kamera hisoblanadi):

| K (hisoblash) | Round robin | Debt v0 | Oracle (kelajakni biladi) |
|---|---|---|---|
| 1 (0.2) | 0.802 | 0.764 | 0.796 |
| 2 (0.4) | 0.869 | 0.855 | 0.902 (p95 kechikish 0.85 → 0.50 s) |
| 3 (0.6) | 0.926 | 0.914 | 0.945 (0.42 → 0.22 s) |
| 4 (0.8) | 0.962 | 0.961 | 0.969 |

Xulosalar:
- Qo'lda yozilgan debt oddiy stride'lar aralashmasidan yaxshi emas; umumiy byudjetda round robin'dan
  biroz yomon. Harakatga qarab gating ham yutmaydi.
- Oracle ham round robin'dan atigi +2..3 recall ustun (kechikish ikki baravar kam). Demak bu
  chorrahalarda (kadriga 20–50 obyekt, tinch davr yo'q) "qachon hisoblash" bo'yicha zaxira kichik;
  o'rganilgan debt eng yaxshi holda shu +3 ni oladi.
- Asosiy yo'qotish o'tkazib yuborilgan kadrning o'zida: 50% hisoblashda recall 0.92 ga tushadi,
  yangi paydo bo'lgan treklarda yo'qotish 17%, barqaror treklarda 2.7%. Keyingi dastak: o'tkazib
  yuborish o'rniga arzon qisman hisoblash (past o'lcham yoki faqat o'zgargan hudud) va siyrak
  kadrlarga moslashgan treker; scheduler tinch kameralar ko'p bo'lgan tarmoqlarda qayta sinaladi.

**Arzon qisman hisoblash (past o'lcham) — bu ham salbiy** (`bench/partial_eval.py`, o'sha 5 x 12000 kadr).
Narx 8 kamera batch'ida o'lchangan: 448x800 = 0.58, 320x576 = 0.41 (1 kamerada 0.79 / 0.67, decoder
doimiy xarajati tufayli).

| Reja | Narx | Recall | Aniqlik | Odam recall |
|---|---|---|---|---|
| har 2-kadr to'liq, qolgani skip | 0.50 | **0.917** | 0.939 | **0.895** |
| faqat 448 | 0.58 | 0.732 | 0.684 | 0.591 |
| har 2-kadr to'liq + 448 | 0.79 | 0.742 | 0.900 | 0.605 |
| faqat 320 | 0.41 | 0.573 | 0.701 | 0.400 |

Past o'lchamdagi detektsiyalar treker holatini buzadi: aralashtirilganda ham recall skip'dan 17 punkt
past. Model 640 da o'qigan, kichik kirishda kichik obyektlar yo'qoladi va boxlar siljiydi; Kalman
bashorati (coast) bir necha kadr uchun ancha aniqroq. Xulosa: hozirgi model va bu kameralarda
hisoblashni boshqarish (qachon ham, qanday arzon ham) yutuq bermaydi; TensorRT bilan detektor
4 kamerada 0.5 ms/kadr, ya'ni tejaladigan GPU vaqti ham kichik. Simulyator va reference'lar
saqlanadi: tinch kameralar ko'p tarmoq yoki Jetson kabi kuchsiz qurilmada qayta sinash uchun.

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
