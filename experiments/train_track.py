"""Fine-tune Sarbon for re-identification on video clips (detection loss + ID contrastive loss).

Every frame of a clip gets the usual detection loss; the identity embeddings of the
queries matched to ground truth, gathered over all frames of the batch, get a supervised
contrastive loss (same person in other frames = positive). Evaluated by cross-frame
retrieval: does each object's nearest neighbour in the other frame have its identity?

  .venv\\Scripts\\python.exe experiments\\train_track.py --mot D:\\data\\mot_like --ckpt runs\\sarbon\\s\\best.pt
CPU smoke test:
  .venv\\Scripts\\python.exe experiments\\train_track.py --synthetic --device cpu --size n --epochs 3
"""
import argparse
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sarbon.data.mot import MOTClips, SyntheticClips, collate_clips  # noqa: E402
from sarbon.models.criterion import SetCriterion, id_contrastive  # noqa: E402
from sarbon.models.sarbon_det import build, load_weights  # noqa: E402
from train_sarbon import param_groups, prep, to_targets  # noqa: E402


def clip_step(model, crit, clip, size, device):
    det_loss, embs, ids, frame_of = 0.0, [], [], []
    for t, (imgs, boxes, labels, idl) in enumerate(clip):
        targets = to_targets(boxes, labels, size, device)
        with torch.autocast(device.type, torch.bfloat16, enabled=device.type == "cuda"):
            out = model(prep(imgs, device), targets=targets)
        loss, _, matches = crit(out, targets)
        det_loss = det_loss + loss
        for b, (q, g) in enumerate(matches):
            embs.append(out["id_embed"][b, q.to(device)])
            ids.append(idl[b][g].to(device))
            frame_of.append(torch.full((len(q),), t, device=device))
    return det_loss / len(clip), torch.cat(embs), torch.cat(ids), torch.cat(frame_of)


@torch.no_grad()
def retrieval(embs, ids, frame_of):
    """Top-1 accuracy of matching each frame-0 object to the same identity in frame 1."""
    a, b = frame_of == 0, frame_of == 1
    if not a.any() or not b.any():
        return float("nan")
    sim = embs[a] @ embs[b].T
    nn_ids = ids[b][sim.argmax(1)]
    common = torch.isin(ids[a], ids[b])
    return (nn_ids[common] == ids[a][common]).float().mean().item() if common.any() else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mot")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--ckpt")
    ap.add_argument("--size", default="s")
    ap.add_argument("--classes", type=int, default=6)
    ap.add_argument("--img", type=int, default=640)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--w-id", type=float, default=1.0)
    ap.add_argument("--freeze-stem", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="runs/track")
    args = ap.parse_args()
    torch.manual_seed(0)
    device = torch.device(args.device)
    if args.synthetic:
        size, train_ds, val_ds, args.workers = 256, SyntheticClips(8, 256), SyntheticClips(4, 256, seed=99), 0
    else:
        size = args.img
        train_ds = MOTClips(args.mot, size)
        val_ds = MOTClips(args.mot, size, samples=200)
    dl = DataLoader(train_ds, args.batch, shuffle=True, num_workers=args.workers, collate_fn=collate_clips,
                    drop_last=True, persistent_workers=args.workers > 0)
    vdl = DataLoader(val_ds, args.batch, num_workers=min(4, args.workers), collate_fn=collate_clips)
    model = build(args.size, args.classes, halo_zero=False).to(device)
    if args.ckpt:
        load_weights(model, args.ckpt)
    if args.freeze_stem:
        for p in model.stem.parameters():
            p.requires_grad_(False)
    crit = SetCriterion(args.classes, project=model.project)
    opt = torch.optim.AdamW(param_groups(model, 1e-4), args.lr)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    def evaluate():
        model.eval()
        accs = []
        with torch.no_grad():
            for clip in vdl:
                _, e, i, f = clip_step(model, crit, clip, size, device)
                accs.append(retrieval(e.float(), i, f))
        model.train()
        accs = [a for a in accs if a == a]
        return sum(accs) / max(1, len(accs))

    print(f"before: cross-frame ID retrieval top-1 = {evaluate():.3f}")
    t0 = time.time()
    for ep in range(args.epochs):
        for it, clip in enumerate(dl):
            det_loss, e, i, _ = clip_step(model, crit, clip, size, device)
            id_loss = id_contrastive(e.float(), i)
            loss = det_loss + args.w_id * id_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            opt.step()
            if it % 50 == 0:
                print(f"ep {ep} it {it}/{len(dl)} det {det_loss.item():.3f} id {id_loss.item():.3f} "
                      f"{time.time() - t0:.0f}s", flush=True)
        print(f"ep {ep}: cross-frame ID retrieval top-1 = {evaluate():.3f}", flush=True)
        torch.save(model.state_dict(), out_dir / f"{args.size}_reid.pt")


if __name__ == "__main__":
    main()
