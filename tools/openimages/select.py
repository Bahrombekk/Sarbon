"""(tools/openimages) Pick Open Images V7 validation images for a yengil/og'ir vehicle test (read-only on the CSVs)."""
import csv, json, random
from collections import defaultdict
M = r"C:\data\openimages_val\meta"
CAR, BUS, TRUCK = "/m/0k4j", "/m/01bjv", "/m/07r04"
LIGHT = {CAR, "/m/0pg52", "/m/01lcw4"}  # car, taxi, limousine -> yengil
HEAVY = {BUS, TRUCK}                     # -> og'ir
AMBIG = {"/m/0h2r6", "/m/012n7d", "/m/01prls", "/m/07yv9"}  # van, ambulance, land vehicle, vehicle
need = [CAR, BUS, TRUCK]
verified = defaultdict(dict)
for r in csv.DictReader(open(f"{M}/validation-annotations-human-imagelabels-boxable.csv", encoding="utf8")):
    verified[r["ImageID"]][r["LabelName"]] = r["Confidence"] == "1"
meta = {r["ImageID"]: r for r in csv.DictReader(open(f"{M}/validation-images-with-rotation.csv", encoding="utf8"))}
boxes, bad = defaultdict(list), set()
for r in csv.DictReader(open(f"{M}/validation-annotations-bbox.csv", encoding="utf8")):
    l, i = r["LabelName"], r["ImageID"]
    if l in AMBIG or ((l in LIGHT or l in HEAVY) and (r["IsGroupOf"] == "1" or r["IsDepiction"] == "1")):
        bad.add(i)
    elif l in LIGHT or l in HEAVY:
        boxes[i].append([0 if l in LIGHT else 1, *(float(r[k]) for k in ("XMin", "YMin", "XMax", "YMax"))])
ok = []
for i, v in verified.items():
    if i in bad or not any(v.get(k) for k in need):
        continue  # at least one vehicle class verified present
    if any(v.get(k) and not any(bb[0] == (0 if k == CAR else 1) for bb in boxes[i]) for k in need):
        continue  # a positive label without boxes: annotation incomplete
    m = meta.get(i, {})
    if m.get("Rotation") not in ("", "0", "0.0", None) or "/by/" not in m.get("License", ""):
        continue  # rotated, or not CC BY
    ok.append(i)
def heavy(v):
    """og'ir verified present if bus or truck is; verified absent only if both are."""
    if v.get(BUS) or v.get(TRUCK):
        return True
    return False if (BUS in v and TRUCK in v) else None


random.Random(0).shuffle(ok)
hv = [i for i in ok if heavy(verified[i])]
pick = hv + [i for i in ok if i not in set(hv)][:1000 - len(hv)]
n_l = sum(b[0] == 0 for i in pick for b in boxes[i]); n_h = sum(b[0] == 1 for i in pick for b in boxes[i])
print("eligible", len(ok), "picked", len(pick), "boxes yengil", n_l, "og'ir", n_h)
json.dump({i: {"boxes": boxes[i], "verified": {"yengil": verified[i].get(CAR), "ogir": heavy(verified[i])}, "license": meta[i]["License"], "author": meta[i]["Author"],
               "url": meta[i]["OriginalLandingURL"]} for i in pick},
          open(r"C:\data\openimages_val\selection.json", "w", encoding="utf8"))
