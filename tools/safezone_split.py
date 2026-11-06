"""SafeZone (SmartGUI shlem) ma'lumotini Sarbon o'qitishi uchun NVMe'ga nusxalash va ro'yxatlar.

Manba (o'zgartirilmaydi): D:/Data Sentir/smart/smart_v4_1024 (smart_v4_build.py natijasi):
  train: asl kadr + "_aug" (rang/geometriya augment) + "_mos" (3x3 uzoqlashtirish mozaikasi),
  val / test: asl kadrlar, 25 kadrlik bloklar bilan bo'lingan.
Sarbon o'zi augment qiladi (mozaika, masshtab, rang), shuning uchun "_aug" nusxalar olinmaydi;
"_mos" qoldiriladi: u kesishma kamerasidagi kichik odamlarni beradi, Sarbon mozaikasi esa 2x2.

    python tools/safezone_split.py --out C:/data/safezone
Natija: <out>/images/{train,val,test}, <out>/labels/..., train.txt, val.txt, test.txt, build.json.
Sinflar: 0 person, 1 helmet, 2 no_helmet (Sarbon SAFEZONE bilan bir xil, map "0:0,1:1,2:2").
"""
import argparse
import json
import shutil
from pathlib import Path

SRC = Path(r"D:/Data Sentir/smart/smart_v4_1024")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(SRC))
    ap.add_argument("--out", default="C:/data/safezone")
    ap.add_argument("--no-mosaic", action="store_true", help="train'ga _mos rasmlarini qo'shmaslik")
    args = ap.parse_args()
    src, out = Path(args.src), Path(args.out)
    info = {"source": str(src), "train_mosaic": not args.no_mosaic, "images": {}, "boxes": {}}
    for part in ("train", "val", "test"):
        (out / "images" / part).mkdir(parents=True, exist_ok=True)
        (out / "labels" / part).mkdir(parents=True, exist_ok=True)
        paths, counts = [], [0, 0, 0]
        for img in sorted((src / "images" / part).glob("*.jpg")):
            if img.stem.endswith("_aug") or (args.no_mosaic and img.stem.endswith("_mos")):
                continue
            lab = src / "labels" / part / (img.stem + ".txt")
            if not lab.exists():
                continue
            dst = out / "images" / part / img.name
            if not dst.exists():
                shutil.copy2(img, dst)
            shutil.copy2(lab, out / "labels" / part / lab.name)
            for line in lab.read_text().splitlines():
                if line.strip():
                    counts[int(line.split()[0])] += 1
            paths.append(str(dst))
        (out / f"{part}.txt").write_text("\n".join(paths) + "\n", encoding="utf-8")
        info["images"][part] = len(paths)
        info["boxes"][part] = dict(zip(("person", "helmet", "no_helmet"), counts))
        print(part, len(paths), info["boxes"][part], flush=True)
    (out / "build.json").write_text(json.dumps(info, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
