"""Synthetic fixed-camera video: textured static background + moving textured objects.

Changed area per frame is the union of each object's old and new footprint, which
is what a real static camera produces (minus sensor noise, added optionally).
"""
import torch
import torch.nn.functional as F

# activity presets: (objects, min size px, max size px, max speed px/frame)
PRESETS = {
    "quiet": (2, 16, 64, 3),       # parking lot at night, corridor
    "moderate": (6, 24, 128, 6),   # entrance, street with a few people
    "busy": (18, 32, 220, 10),     # crowded square, highway
}


def _texture(h, w, gen, device, scale=8):
    t = torch.rand(1, 3, h // scale + 1, w // scale + 1, generator=gen, device=device)
    t = F.interpolate(t, size=(h, w), mode="bilinear", align_corners=False)[0]
    return t.permute(1, 2, 0).contiguous()  # HWC


class Camera:
    def __init__(self, h, w, preset, seed, device="cuda", noise=0.0):
        self.h, self.w, self.noise, self.dev = h, w, noise, device
        self.gen = torch.Generator(device=device).manual_seed(seed)
        n, lo, hi, v = PRESETS[preset]
        g = self.gen
        self.bg = _texture(h, w, g, device)
        self.size = torch.randint(lo, hi, (n, 2), generator=g, device=device)
        self.pos = torch.rand(n, 2, generator=g, device=device) * torch.tensor(
            [h, w], device=device) * 0.8
        self.vel = (torch.rand(n, 2, generator=g, device=device) * 2 - 1) * v
        self.tex = [_texture(int(s[0]), int(s[1]), g, device, 4) for s in self.size.tolist()]

    def next(self):
        lim = torch.tensor([self.h, self.w], device=self.dev) - self.size
        self.pos += self.vel
        bounce = (self.pos < 0) | (self.pos > lim)
        self.vel[bounce] *= -1
        self.pos.clamp_(min=torch.zeros_like(lim), max=lim)
        f = self.bg.clone()
        for (y, x), (sh, sw), t in zip(self.pos.int().tolist(), self.size.tolist(), self.tex):
            f[y:y + sh, x:x + sw] = t
        if self.noise:
            f += torch.randn(f.shape, generator=self.gen, device=self.dev) * self.noise
        return f


def make_sequence(cams, n_frames, dtype):
    """[n_frames, n_cams, H, W, 3] on the GPU."""
    return torch.stack([torch.stack([c.next().to(dtype) for c in cams]) for _ in range(n_frames)])
