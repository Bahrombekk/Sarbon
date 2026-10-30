"""Partial-label training: class masks in the loss, YOLO data, class-head remapping."""
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from sarbon.data.yolo import WithClassMask, YoloDetection, label_path, read_labels  # noqa: E402
from sarbon.models.criterion import SetCriterion, mal_loss, sigmoid_focal  # noqa: E402
from sarbon.models.sarbon_det import build  # noqa: E402


def test_masked_class_gets_no_gradient():
    logits = torch.randn(2, 5, 3, requires_grad=True)
    pos = torch.zeros(2, 5, 3, dtype=torch.bool)
    pos[0, 1, 0] = True
    q = pos.float() * 0.8
    valid = torch.tensor([[True, True, True], [False, True, False]])[:, None]
    for loss in (mal_loss(logits, q, pos, valid=valid), sigmoid_focal(logits, pos.float(), valid=valid)):
        g, = torch.autograd.grad(loss, logits)
        assert (g[1, :, 0] == 0).all() and (g[1, :, 2] == 0).all()
        assert (g[1, :, 1] != 0).all() and (g[0] != 0).all()


def test_criterion_valid_matches_unmasked_when_all_true():
    torch.manual_seed(0)
    m = build("n", 3).eval()
    img = torch.randn(2, 3, 128, 128)
    targets = [{"labels": torch.tensor([0]), "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.3]])},
               {"labels": torch.tensor([2]), "boxes": torch.tensor([[0.3, 0.4, 0.1, 0.2]])}]
    out = m(img)
    crit = SetCriterion(3, project=m.project)
    a, _, _ = crit(out, targets)
    full = [dict(t, valid=torch.ones(3, dtype=torch.bool)) for t in targets]
    b, _, _ = crit(out, full)
    assert torch.allclose(a, b)
    part = [dict(targets[0]), dict(targets[1], valid=torch.tensor([False, False, True]))]
    c, _, _ = crit(out, part)
    assert c < a  # background loss of the masked classes is gone


def _yolo_set(tmp):
    root = Path(tmp) / "ds"
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir()
    paths = []
    for i in range(3):
        p = root / "images" / f"seq_{i:04d}.jpg"
        cv2.imencode(".jpg", np.full((720, 1280, 3), 100, np.uint8))[1].tofile(str(p))
        # one valid box, one out-of-range class id (dropped)
        label_path(p).write_text("0 0.5 0.5 0.25 0.5\n7 0.1 0.1 0.1 0.1\n")
        paths.append(str(p))
    (root / "list.txt").write_text("\n".join(paths))
    return root


def test_yolo_dataset_letterbox_and_mask(tmp_path):
    root = _yolo_set(tmp_path)
    rows, bad = read_labels(root / "labels" / "seq_0000.txt", 2)
    assert len(rows) == 1 and len(bad) == 1
    ds = YoloDetection(root / "list.txt", size=640, class_map={0: 3, 1: 4}, train=False)
    img, b, l, meta = ds[0]
    assert img.shape == (3, 640, 640) and l.tolist() == [3]
    # 1280x720 -> scale 0.5, pad_y 140: box centre (640, 360) -> (320, 320), size 160 x 180
    assert np.allclose(b.numpy(), [[240, 230, 400, 410]], atol=1)
    gid, gb, gl = ds.gt_letterboxed()[0]
    assert np.allclose(gb, b.numpy(), atol=1) and gl == [3]
    w = WithClassMask(ds, 6, [3, 4])
    assert w[0][3][-1].tolist() == [False, False, False, True, True, False]


def test_init_remaps_class_rows(tmp_path):
    from train_sarbon import load_init
    src = build("n", 6)
    path = tmp_path / "src.pt"
    torch.save(src.state_dict(), path)
    dst = build("n", 6)
    names_src = ["person", "bicycle", "car", "motorcycle", "bus", "truck"]
    names_dst = ["person", "bicycle", "motorcycle", "yengil", "ogir", "train"]
    before = dst.enc_cls.weight.detach().clone()
    load_init(dst, path, names_src, names_dst, alias={"yengil": "car", "ogir": "truck"})
    w_s, w_d = src.enc_cls.weight, dst.enc_cls.weight
    assert torch.equal(w_d[3], w_s[2]) and torch.equal(w_d[4], w_s[5]) and torch.equal(w_d[2], w_s[3])
    assert torch.equal(w_d[5], before[5])  # train: never seen, fresh init kept
    assert torch.equal(dst.dn_label.weight[-1], src.dn_label.weight[-1])
