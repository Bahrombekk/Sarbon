"""Download the selected Open Images val images (official CVDF bucket) and write YOLO labels + list."""
import json, os, urllib.request
from concurrent.futures import ThreadPoolExecutor
R = r"C:\data\openimages_val"
sel = json.load(open(os.path.join(R, "selection.json"), encoding="utf8"))
os.makedirs(os.path.join(R, "images"), exist_ok=True); os.makedirs(os.path.join(R, "labels"), exist_ok=True)

def get(i):
    p = os.path.join(R, "images", f"{i}.jpg")
    if not os.path.exists(p):
        try:
            urllib.request.urlretrieve(f"https://open-images-dataset.s3.amazonaws.com/validation/{i}.jpg", p + ".part")
            os.replace(p + ".part", p)
        except Exception as e:
            return i, str(e)
    with open(os.path.join(R, "labels", f"{i}.txt"), "w") as f:
        for c, x0, y0, x1, y1 in sel[i]["boxes"]:
            f.write(f"{c} {(x0 + x1) / 2:.6f} {(y0 + y1) / 2:.6f} {x1 - x0:.6f} {y1 - y0:.6f}\n")
    return i, None

with ThreadPoolExecutor(16) as ex:
    res = list(ex.map(get, sel))
fails = [r for r in res if r[1]]
okids = [i for i, e in res if not e]
open(os.path.join(R, "val.txt"), "w").write("\n".join(os.path.join(R, "images", f"{i}.jpg") for i in okids) + "\n")
size = sum(os.path.getsize(os.path.join(R, "images", f"{i}.jpg")) for i in okids)
print("ok", len(okids), "failed", len(fails), fails[:3], "MB %.0f" % (size / 1e6))
