"""Shape + parameter-budget check for ADMSB.

Run from the repo root:
    python -m model_v2.test_admsb
"""

import torch
import torch.nn as nn

from .admsb import ADMSB, C2f_ADMSB
from .backbone_cnn import CSPBackbone
from .layers import Bottleneck, C2f


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def main():
    torch.manual_seed(0)

    # ---- 1. shape check on a P3-sized feature map -----------------------
    x = torch.randn(2, 256, 80, 80)
    blk = ADMSB(256, 256).eval()
    with torch.no_grad():
        y = blk(x)
    print("input :", tuple(x.shape))
    print("output:", tuple(y.shape))
    assert y.shape == (2, 256, 80, 80), y.shape
    assert torch.isfinite(y).all()

    # zero-init offsets => at step 0 the deformable branch is a plain 3x3
    off = blk.branch_a.offset(blk.reduce(x))
    assert torch.allclose(off, torch.zeros_like(off)), "offsets not zero-init"

    # ---- 2. per-block parameter comparison ------------------------------
    print("\nper-block params (C in -> C out)")
    print(f"{'C':>6} {'Bottleneck':>12} {'ADMSB':>12} {'delta':>12}")
    for c in (128, 256, 512):
        pb = n_params(Bottleneck(c, c, True, e=1.0))
        pa = n_params(ADMSB(c, c))
        print(f"{c:>6} {pb:>12,} {pa:>12,} {pa - pb:>+12,}")

    # ---- 3. realistic budget: P3 + P4 C2f, first 2 blocks swapped -------
    # YOLOv26-style backbone: P3 = C2f(256, 256, n=6), P4 = C2f(512, 512, n=6).
    # Inside C2f the bottlenecks run on c2 * e channels, not on c2.
    total_base, total_new = 0, 0
    print("\nstage-level budget (first 2 of 6 blocks replaced)")
    for name, ch, n in (("P3", 256, 6), ("P4", 512, 6)):
        base = n_params(C2f(ch, ch, n=n, shortcut=True))
        new = n_params(C2f_ADMSB(ch, ch, n=n, shortcut=True, n_admsb=2))
        total_base += base
        total_new += new
        print(f"  {name}: {base:,} -> {new:,}  ({new - base:+,})")
    delta = (total_new - total_base) / 1e6
    print(f"  total delta: {delta:+.4f} M  (budget: increase <= 0.4 M)")
    # the constraint is on the *increase*; ADMSB is in fact lighter than
    # two stock bottlenecks because it works at C/2 internally.
    assert delta <= 0.4, f"parameter budget exceeded: {delta:+.4f} M"

    # ---- 4. C2f_ADMSB forward + backward --------------------------------
    m = C2f_ADMSB(256, 256, n=6, shortcut=True, n_admsb=2)
    xin = torch.randn(2, 256, 80, 80, requires_grad=True)
    out = m(xin)
    print("\nC2f_ADMSB:", tuple(xin.shape), "->", tuple(out.shape))
    assert out.shape == (2, 256, 80, 80)
    out.mean().backward()
    grads = [p.grad for p in m.parameters() if p.requires_grad]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    # offsets get gradient even though they start at zero
    g_off = m.m[0].branch_a.offset.weight.grad
    print("offset-conv grad norm:", float(g_off.norm()))

    # ---- 5. end-to-end backbone with the real config --------------------
    bb = CSPBackbone(width=0.5, depth=0.34).eval()
    img = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        feats = bb(img)
    print("\nCSPBackbone(width=0.5, depth=0.34) on 640x640:")
    for name, f, st in zip(("C2", "C3", "C4", "C5"), feats, bb.out_strides):
        print(f"  {name}: {tuple(f.shape)}  stride {st}")
    assert [f.shape[1] for f in feats] == bb.out_channels
    assert [640 // f.shape[-1] for f in feats] == list(bb.out_strides)

    # same backbone with the stock C2f everywhere, for the parameter delta
    stock = CSPBackbone(width=0.5, depth=0.34)
    for stage, ch, rep in ((stock.stage2, 256, 6), (stock.stage3, 512, 6)):
        cc = max(int(round(ch * 0.5)), 16)
        nn_ = max(int(round(rep * 0.34)), 1)
        stage[1] = C2f(cc, cc, nn_, shortcut=True)
    d = (n_params(bb) - n_params(stock)) / 1e6
    print(f"\nbackbone params: {n_params(stock):,} -> {n_params(bb):,} "
          f"({d:+.4f} M, budget: increase <= 0.4 M)")
    assert d <= 0.4, f"parameter budget exceeded: {d:+.4f} M"

    print("\nall checks passed")


if __name__ == "__main__":
    main()
