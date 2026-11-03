"""Out-of-domain vehicle test on Open Images V7 val: Sarbon models vs RailSafe YOLO26n.

Open Images protocol: a class is scored only on the images where it was verified (present or
absent), so unverified objects are neither misses nor false positives."""
import json, sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
sys.path.insert(0, "D:/Sarbon")
from sarbon.cli.compare import yolo_dets
from sarbon.data.coco import collate
from sarbon.data.yolo import YoloDetection
from sarbon.evaluation import coco_metrics
from sarbon.evaluation.coco_eval import prep, postprocess
from sarbon.models.sarbon_det import load_detector

R = Path(r"C:\data\openimages_val")
MODELS = {"Sarbon v0.1": "D:/Sarbon/weights/sarbon-s-railsafe-v0.1.pt",
          "Sarbon smart": "D:/Sarbon/weights/sarbon-s-railsafe-smart.pt",
          "Sarbon wide": "D:/Sarbon/runs/railsafe/s_railsafe_hn_wide/best.pt"}
YOLO = {"RailSafe YOLO26n @1088": R / "yolo26n_1088.json", "RailSafe YOLO26n @640": R / "yolo26n_640.json"}


def main():
    extra = dict(a.split("=", 1) for a in sys.argv[1:])  # name=weights.pt adds a Sarbon model
    sel = json.load(open(R / "selection.json", encoding="utf8"))
    ds = YoloDetection(R / "val.txt", 640, {0: 3, 1: 4}, train=False)
    gt = ds.gt_letterboxed()
    ver = {im["id"]: sel[Path(im["file_name"]).stem]["verified"] for im in ds.images}
    subset = {3: {i for i, v in ver.items() if v["yengil"] is not None},
              4: {i for i, v in ver.items() if v["ogir"] is not None}}
    dets = {}
    for name, w in {**MODELS, **extra}.items():
        m = load_detector(w, "s", 6, halo_zero=False).cuda().eval()
        out = []
        for imgs, _, _, meta in DataLoader(ds, 16, num_workers=4, collate_fn=collate):
            with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
                o = m(prep(imgs, torch.device("cuda")))
            for (bx, sc, lb), mm in zip(postprocess(o, 640), meta):
                out += [{"image_id": mm[0], "category_id": l, "bbox": [b[0], b[1], b[2] - b[0], b[3] - b[1]], "score": s}
                        for b, s, l in zip(bx.float().tolist(), sc.float().tolist(), lb.tolist())]
        dets[name] = out
    for name, p in YOLO.items():
        if p.exists():
            dets[name] = yolo_dets(ds, p, {0: 3, 1: 4})
    res = {}
    for name, d in dets.items():
        r = {}
        for c, cname in ((3, "yengil"), (4, "ogir")):
            ids = subset[c]
            m = coco_metrics([x for x in gt if x[0] in ids], [x for x in d if x["image_id"] in ids and x["category_id"] == c], 6, [c])
            r.update({f"{cname}_{k}": round(v, 1) for k, v in m.items() if k in ("AP", "AP50", "AP_small", "AP_medium", "AP_large")})
        r["mean_AP"] = round((r["yengil_AP"] + r["ogir_AP"]) / 2, 1)
        res[name] = r
    (R / "result.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print("images", len(ds), "yengil images", len(subset[3]), "og'ir images", len(subset[4]))
    for k, v in res.items():
        print(f"{k:26s}", v)


if __name__ == "__main__":
    main()
