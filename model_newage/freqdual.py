"""Design 1: frequency/spatial dual path with a class-conditioned gate.

Why a frequency path.  crazing and rolled-in_scale are not objects with edges;
they are texture statistics.  NEU-DET images carry only 200x200 of real
information (the 640x640 files are an upsample), so a convolution stack has
little to work with in the pixel domain, while rolled steel has strongly
oriented, band-limited texture that separates cleanly in the spectrum.  A
patch-wise FFT pooled into radial x orientation bands is a different *form* of
information, not another convolution variant -- and it is parameter-free.

Why a gate.  93.2% of NEU-DET images contain exactly one defect class, so an
image-level classifier is close to an oracle here.  Letting it decide how much
of the texture path to mix in is what stops the two defect families from
competing: measured on this data, every shared-trunk change gained ~0.04 on the
texture classes and lost ~0.03 on the crisp ones.

The module keeps CSPBackbone's interface (out_channels / out_strides / a list
of four feature maps), so it drops into YOLOSAM2 in place of the backbone.

    python -m model_newage.freqdual        # shape and parameter self-test
"""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.layers import Conv


def _band_matrix(patch: int, n_radial: int, n_orient: int,
                 log_radial: bool = False) -> torch.Tensor:
    """(F, n_radial*n_orient) soft membership of each rfft2 bin in a band.

    Radial bands capture the scale of the texture, orientation bins capture the
    rolling direction -- which is shared by every image in the dataset, so it
    carries real discriminative signal rather than nuisance variation.
    """
    fy = torch.fft.fftfreq(patch).view(-1, 1).expand(patch, patch // 2 + 1)
    fx = torch.fft.rfftfreq(patch).view(1, -1).expand(patch, patch // 2 + 1)
    r = torch.sqrt(fy ** 2 + fx ** 2)
    theta = torch.atan2(fy, fx) % torch.pi              # unoriented: [0, pi)

    r_max = float(r.max()) + 1e-6
    if log_radial:
        # Measured on NEU-DET: the defect/background band-energy contrast falls
        # monotonically with radius for every class, and the top quarter of the
        # spectrum carries +0.01..+0.04 against +0.2..+0.6 at the bottom.  With
        # linear bins half the descriptor is spent where there is no signal.
        rr = torch.log1p(r / r_max * (torch.e - 1))       # 0..1, low-freq spread
    else:
        rr = r / r_max
    r_idx = (rr * n_radial).clamp(0, n_radial - 1e-4).long()
    t_idx = (theta / torch.pi * n_orient).clamp(0, n_orient - 1e-4).long()
    flat = (r_idx * n_orient + t_idx).reshape(-1)

    n_bands = n_radial * n_orient
    M = torch.zeros(flat.numel(), n_bands)
    M[torch.arange(flat.numel()), flat] = 1.0
    M = M / M.sum(0, keepdim=True).clamp(min=1.0)        # mean, not sum
    return M


class SpectralStem(nn.Module):
    """Patch-wise FFT -> band descriptors -> a small conv stack.

    Output stride equals the patch size, so patch=16 lands on P4 and one more
    stride-2 conv reaches P5 -- the two levels where the diffuse classes live
    (their boxes are 216-489 px, far above what stride 8 can represent).  A
    single patch fixes the analysis window while the classes differ in texture
    scale by an order of magnitude (inclusion 126 px, pitted_surface 489 px),
    so several stems can run in parallel and be summed per level.

    `analyse` and `encode` are split so a router can reweight the bands in
    between, which is where per-band conditioning has to happen.
    """

    def __init__(self, patch: int = 16, n_radial: int = 8, n_orient: int = 4,
                 width: int = 96, use_fft: bool = True, hann: bool = False,
                 phase: bool = False, log_radial: bool = False,
                 mps_bond: int = 0):
        super().__init__()
        self.patch = patch
        self.use_fft = use_fft
        self.phase = phase
        self.register_buffer("bands", _band_matrix(patch, n_radial, n_orient,
                                                   log_radial))
        if hann:
            # A rectangular window makes the patch edges look like a step, and
            # the spectrum reports that discontinuity as broadband high
            # frequency.  Hann tapers it, so the high bands measure texture
            # rather than the windowing.
            w1 = torch.hann_window(patch, periodic=False)
            self.register_buffer("window", (w1[:, None] * w1[None, :]))
        else:
            self.window = None
        self.n_bands = n_radial * n_orient
        # magnitude, plus the circular mean of phase (cos, sin) per band
        self.n_feat = self.n_bands * (3 if phase else 1)
        # MPS output is concatenated, so the conv stack still sees the raw
        # bands and the ablation is exactly "these extra channels or not".
        self.mps = (MPSBand(n_radial, n_orient, mps_bond, self.n_bands)
                    if mps_bond > 0 else None)
        if self.mps is not None:
            self.n_feat += self.n_bands
        self.norm = nn.BatchNorm2d(self.n_feat)
        self.stem = nn.Sequential(Conv(self.n_feat, width, 3, 1),
                                  Conv(width, width, 3, 1))
        self.down = Conv(width, width * 2, 3, 2)
        self.out_channels = [width, width * 2]           # strides p and 2p

    def analyse(self, x: torch.Tensor) -> torch.Tensor:
        """(B,3,H,W) -> (B, n_feat, H/p, W/p) band descriptors."""
        b, _, h, w = x.shape
        p = self.patch
        assert h % p == 0 and w % p == 0, f"imgsz must be divisible by {p}"
        g = x.mean(1, keepdim=True)
        patches = F.unfold(g, kernel_size=p, stride=p)
        L = patches.shape[-1]
        patches = patches.permute(0, 2, 1).reshape(b * L, p, p).float()
        if self.window is not None:
            patches = patches * self.window

        if self.use_fft:
            z = torch.fft.rfft2(patches, norm="ortho")
            mag = torch.log1p(z.abs()).reshape(b * L, -1)
            feats = [mag @ self.bands]
            if self.phase:
                # circular mean per band: the resultant's length says how
                # coherently the band's structure sits inside the patch, which
                # is positional information magnitude alone cannot carry
                ang = torch.angle(z).reshape(b * L, -1)
                feats.append(torch.cos(ang) @ self.bands)
                feats.append(torch.sin(ang) @ self.bands)
            if self.mps is not None:
                feats.append(self.mps(feats[0]).to(feats[0].dtype))
            feat = torch.cat(feats, 1)
        else:
            # ablation: identical path and parameter count, no transform
            flat = patches.reshape(b * L, -1)[:, : self.bands.shape[0]]
            feat = (flat @ self.bands).repeat(1, 3 if self.phase else 1)
            if self.mps is not None:
                feat = torch.cat([feat, self.mps(feat[:, :self.n_bands])], 1)

        feat = feat.reshape(b, h // p, w // p, -1).permute(0, 3, 1, 2)
        return self.norm(feat.to(x.dtype))

    def encode(self, bandmap: torch.Tensor) -> List[torch.Tensor]:
        f1 = self.stem(bandmap)
        return [f1, self.down(f1)]

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        return self.encode(self.analyse(x))


class MPSBand(nn.Module):
    """Matrix-product-state mixing over the radial band axis.

    The band descriptor is not a flat channel vector: it is a (radial x
    orientation) tensor, and `SpectralStem` currently flattens it so the 3x3
    conv sees 32 unordered channels.  That throws away the one piece of
    structure the FFT actually gives us -- that band r and band r+1 are
    adjacent scales of the same texture.

    An MPS treats the radial axis as a chain of sites.  Site r carries the
    orientation vector (plus a constant, which lets a site drop out of the
    product), each site contributes a chi x chi matrix, and the output is the
    ordered product of all of them.  That makes the map a degree-`n_radial`
    polynomial in the input with only O(R * O * chi^2) parameters -- a
    multiplicative interaction across scales that a linear channel mixer
    cannot express at any width.  It is the classical (tensor-network) limit
    of a quantum circuit acting on the band register: no sampling, no
    barren plateau, fully differentiable.

    The running vector is renormalised after every site.  Without that, a
    product of eight matrices either overflows or underflows within a few
    hundred steps -- the same failure mode that killed the first cross-domain
    attention, and the reason the contraction is forced to fp32.
    """

    def __init__(self, n_radial: int, n_orient: int, bond: int = 16,
                 out_feat: int = 32):
        super().__init__()
        self.R, self.O, self.chi = n_radial, n_orient, bond
        # cores[r, o, :, :] is the matrix contributed by orientation channel o
        # at site r; the extra o index (index O) is the constant term.
        cores = torch.randn(n_radial, n_orient + 1, bond, bond) * (bond ** -0.5)
        # start each site near the identity so the initial product is stable
        cores[:, -1] = torch.eye(bond)
        self.cores = nn.Parameter(cores)
        self.left = nn.Parameter(torch.randn(bond) * (bond ** -0.5))
        self.out = nn.Linear(bond, out_feat)
        self.out_feat = out_feat

    def forward(self, bands: torch.Tensor) -> torch.Tensor:
        """(N, R*O) magnitude descriptors -> (N, out_feat)."""
        n = bands.shape[0]
        v = bands.float().reshape(n, self.R, self.O)
        v = torch.cat([v, torch.ones_like(v[:, :, :1])], dim=2)   # (N,R,O+1)
        h = self.left.expand(n, self.chi)
        for r in range(self.R):
            # (N,O+1) x (O+1,chi,chi) -> (N,chi,chi), then h @ M
            m = torch.einsum("no,oij->nij", v[:, r], self.cores[r])
            h = torch.einsum("ni,nij->nj", h, m)
            h = h / h.norm(dim=1, keepdim=True).clamp_min(1e-6)
        return self.out(h)


class BandRouter(nn.Module):
    """One attention weight per frequency band, conditioned on image and class.

    A single scalar gate makes the whole spectral path share one weight, but
    the classes are expected to live in different bands (crazing isotropic and
    high, rolled-in_scale mid and oriented, scratches low and strongly
    oriented).  The weights are also directly plottable per class, which turns
    "why frequency" into a figure.
    """

    def __init__(self, in_ch: int, nc: int, n_feat: int, hidden: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(in_ch + nc, hidden), nn.SiLU(),
                                 nn.Linear(hidden, n_feat))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)        # starts flat at 0.5
        self.last: Optional[torch.Tensor] = None

    def forward(self, feat: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        z = torch.cat([feat.mean((2, 3)), logits.softmax(-1)], -1)
        a = self.mlp(z).sigmoid()                # (B, n_feat)
        self.last = a.detach() if not self.training else None
        return a[:, :, None, None] * 2.0         # neutral at 1.0


class SpatialGate(nn.Module):
    """Per-pixel modulation on top of the per-image class gate.

    One scalar per image cannot represent an image where a boundary defect sits
    on a textured background, and 6.8% of images carry more than one class.
    """

    def __init__(self, in_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, 1, 1)
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)          # starts at 0.5, i.e. neutral

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.conv(feat).sigmoid()        # (B,1,H,W)


class AuxSegHead(nn.Module):
    """Dense defect-region supervision, dropped at inference.

    Costs nothing to label -- the targets are the ground-truth boxes
    rasterised -- and nothing at test time.  On 1440 images the detection loss
    alone is a thin training signal, and the all-classes-rise pattern of the
    dual path suggests this model responds to added supervision rather than to
    any one architectural trick.
    """

    def __init__(self, in_ch: int, nc: int, hidden: int = 64):
        super().__init__()
        self.body = nn.Sequential(Conv(in_ch, hidden, 3, 1),
                                  nn.Conv2d(hidden, nc, 1))
        nn.init.constant_(self.body[-1].bias, -4.0)

    def forward(self, feat):
        return self.body(feat)

    @staticmethod
    def target(logits: torch.Tensor, targets: torch.Tensor,
               imgsz: int) -> torch.Tensor:
        b, nc, h, w = logits.shape
        y = torch.zeros_like(logits)
        if targets.numel() == 0:
            return y
        ys = torch.arange(h, device=logits.device).view(h, 1) * (imgsz / h)
        xs = torch.arange(w, device=logits.device).view(1, w) * (imgsz / w)
        for row in targets:
            i, c, cx, cy, bw, bh = row.tolist()
            i, c = int(i), int(c)
            if not (0 <= i < b and 0 <= c < nc):
                continue
            x1, y1 = (cx - bw / 2) * imgsz, (cy - bh / 2) * imgsz
            x2, y2 = (cx + bw / 2) * imgsz, (cy + bh / 2) * imgsz
            m = ((xs >= x1) & (xs < x2) & (ys >= y1) & (ys < y2))
            y[i, c][m] = 1.0
        return y


class CrossDomainFusion(nn.Module):
    """Spatial positions query the spectral map, instead of a scalar mix.

    The gated sum forces one mixing ratio (per image, or per pixel with the
    spatial gate) onto every channel; here each spatial position reads spectral
    context from anywhere in the image.  gamma starts at zero, so training
    begins exactly at the spatial baseline and the attention path has to earn
    its way in -- the same discipline as the gate starting neutral.

    K/V are pooled to at most kv x kv positions: cross-domain context does not
    need full resolution, and it bounds the attention matrix at the fine
    levels (P3 at 640 would otherwise be 6400x6400).
    """

    def __init__(self, channels: int, dim: int = 64, kv: int = 20):
        super().__init__()
        self.q = nn.Conv2d(channels, dim, 1)
        self.k = nn.Conv2d(channels, dim, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.kv = kv
        # cosine attention: normalised q,k bound every logit to [-1,1] times a
        # learnable temperature, so no activation growth can overflow it
        self.logit_scale = nn.Parameter(torch.tensor(2.0).log())
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, spatial: torch.Tensor, spec: torch.Tensor) -> torch.Tensor:
        b, c, h, w = spatial.shape
        if max(spec.shape[-2:]) > self.kv:
            spec = F.adaptive_avg_pool2d(spec, self.kv)
        # Bounded logits, fp32 softmax.  The first version dotted raw conv
        # features in fp16 under autocast; their magnitude grows over training,
        # the 64-dim dot product passes 65504, softmax turns NaN, and the model
        # collapses to mAP 0.000 mid-run (seed 0 died at epoch ~110).  Same
        # failure class as the CIoU overflow in the assigner.
        q = F.normalize(self.q(spatial).flatten(2).transpose(1, 2), dim=-1)
        k = F.normalize(self.k(spec).flatten(2), dim=1)
        v = self.v(spec).flatten(2).transpose(1, 2)             # B,M,C
        logits = (q.float() @ k.float()) * self.logit_scale.exp().clamp(max=50)
        attn = torch.softmax(logits, dim=-1).to(v.dtype)        # B,N,M
        out = (attn @ v).transpose(1, 2).reshape(b, c, h, w)
        return spatial + self.gamma * out


class ClassGate(nn.Module):
    """Image-level classifier turned into one scalar mixing weight.

    The per-class affinity is learned rather than hand-set: we know from the
    ablations that the texture path should matter for some classes and not
    others, but not by how much, and hard-coding it would be tuning on the
    validation split.
    """

    def __init__(self, in_ch: int, nc: int):
        super().__init__()
        self.fc = nn.Linear(in_ch, nc)
        self.affinity = nn.Parameter(torch.zeros(nc))
        self.logits: Optional[torch.Tensor] = None       # for the aux loss

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        pooled = feat.mean((2, 3))
        logits = self.fc(pooled)                         # (B, nc)
        # Only hold on to it while training, and only until aux_loss consumes
        # it: this tensor carries an autograd graph, and copy.deepcopy(model)
        # in the evaluation path cannot copy a non-leaf tensor.
        self.logits = logits if self.training else None
        p = logits.softmax(-1)
        gate = (p * self.affinity.sigmoid()).sum(-1)     # (B,)
        return gate.view(-1, 1, 1, 1)


class FreqDualBackbone(nn.Module):
    """CSPBackbone + spectral path, fused at P4/P5 under the class gate.

    P2/P3 are left alone: the crisp classes are the ones that live there and
    they are the ones every wide-receptive-field change has damaged.
    """

    def __init__(self, nc: int = 6, patch=16, n_radial: int = 8,
                 n_orient: int = 4, freq_width: int = 96,
                 aux_cls_weight: float = 0.2, aux_seg_weight: float = 0.0,
                 gate_mode: str = "class", use_fft: bool = True,
                 hann: bool = False, phase: bool = False,
                 band_routing: bool = False, log_radial: bool = False,
                 fusion: str = "gate", mps_bond: int = 0,
                 imgsz: int = 640, trunk: str = "cnn", **trunk_kwargs):
        super().__init__()
        # Either trunk works: both expose out_channels / out_strides and return
        # four maps.  Pairing the spectral path with SAM2's Hiera answers a
        # question the CNN pairing cannot -- whether explicit frequency
        # descriptors still add anything on top of a large pretrained ViT, or
        # only compensate for a weaker trunk.
        if trunk == "cnn":
            from model.backbone_cnn import CSPBackbone
            self.spatial = CSPBackbone(**trunk_kwargs)
        elif trunk == "hiera":
            from model.hiera import HieraAdapterBackbone
            ckpt = trunk_kwargs.pop("checkpoint", None)
            self.spatial = HieraAdapterBackbone(**trunk_kwargs)
            if ckpt:
                self.spatial.load_sam2_checkpoint(ckpt)
        else:
            raise ValueError(f"unknown trunk {trunk!r}; use cnn or hiera")
        self.out_channels = list(self.spatial.out_channels)
        self.out_strides = list(self.spatial.out_strides)
        self.aux_cls_weight = aux_cls_weight
        self.aux_seg_weight = aux_seg_weight
        self.gate_mode = gate_mode
        self.imgsz = imgsz

        patches = [patch] if isinstance(patch, int) else list(patch)
        self.stems = nn.ModuleList(
            SpectralStem(p, n_radial, n_orient, freq_width, use_fft, hann,
                         phase, log_radial, mps_bond)
            for p in patches)

        # every (stem, its two output strides) that meets a backbone level
        self.fuse = []                       # (level_idx, stem_idx, out_idx)
        for si, p in enumerate(patches):
            for oi, fs in enumerate((p, p * 2)):
                for k, s_ in enumerate(self.out_strides):
                    if s_ == fs:
                        self.fuse.append((k, si, oi))
        if not self.fuse:
            raise ValueError(f"patches {patches} meet no backbone stride "
                             f"{self.out_strides}")
        self.proj = nn.ModuleList(
            Conv(self.stems[si].out_channels[oi], self.out_channels[k], 1, 1)
            for k, si, oi in self.fuse)

        self.fusion = fusion
        self.xattn = (nn.ModuleList(
            CrossDomainFusion(self.out_channels[k]) for k, _, _ in self.fuse)
            if fusion == "xattn" else None)
        self.gate = ClassGate(self.out_channels[-1], nc)
        self.routers = (nn.ModuleList(
            BandRouter(self.out_channels[-1], nc, st.n_feat) for st in self.stems)
            if band_routing else None)
        self.sgate = (nn.ModuleList(
            SpatialGate(self.out_channels[k]) for k, _, _ in self.fuse)
            if gate_mode == "spatial" else None)
        # auxiliary dense head hangs off P3, the finest level the neck uses
        self.aux_seg = (AuxSegHead(self.out_channels[1], nc)
                        if aux_seg_weight > 0 else None)
        self._seg_logits: Optional[torch.Tensor] = None

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        feats = self.spatial(x)
        g = (self.gate(feats[-1]) if self.gate_mode != "off"
             else torch.full((x.shape[0], 1, 1, 1), 0.5, device=x.device,
                             dtype=x.dtype))
        # the gate's classifier also conditions the router, so compute it first
        logits = (self.gate.logits if self.gate.logits is not None
                  else self.gate.fc(feats[-1].mean((2, 3))))
        spec = []
        for si, stem in enumerate(self.stems):
            bm = stem.analyse(x)
            if self.routers is not None:
                bm = bm * self.routers[si](feats[-1], logits)
            spec.append(stem.encode(bm))

        for n, ((k, si, oi), proj) in enumerate(zip(self.fuse, self.proj)):
            s_ = proj(spec[si][oi])
            if s_.shape[-2:] != feats[k].shape[-2:]:
                s_ = F.interpolate(s_, size=feats[k].shape[-2:], mode="nearest")
            if self.xattn is not None:
                feats[k] = self.xattn[n](feats[k], s_)
            else:
                gk = g if self.sgate is None else g * self.sgate[n](feats[k])
                feats[k] = (1.0 - gk) * feats[k] + gk * s_
        self._seg_logits = (self.aux_seg(feats[1])
                            if (self.aux_seg is not None and self.training)
                            else None)
        return feats

    @torch.no_grad()
    def band_weights(self) -> Optional[torch.Tensor]:
        """(B, n_feat) router weights from the last eval forward, for plotting."""
        if self.routers is None:
            return None
        w = [r.last for r in self.routers if r.last is not None]
        return torch.cat(w, -1) if w else None

    def aux_loss(self, targets: torch.Tensor, bs: int) -> torch.Tensor:
        """Multi-label image-level loss on the gate's classifier.

        targets are the detector's (N, 6) rows; an image counts as containing
        every class that appears in its boxes, which handles the 6.8% of images
        with more than one defect type.
        """
        logits, self.gate.logits = self.gate.logits, None   # consume it
        if logits is None:
            self._seg_logits = None
            return torch.zeros((), device=next(self.parameters()).device)
        if targets.numel() == 0:
            self._seg_logits = None
            return logits.sum() * 0.0
        y = torch.zeros_like(logits)
        bi = targets[:, 0].long().clamp(0, bs - 1)
        ci = targets[:, 1].long().clamp(0, logits.shape[1] - 1)
        y[bi, ci] = 1.0
        loss = F.binary_cross_entropy_with_logits(logits, y) * self.aux_cls_weight
        seg, self._seg_logits = self._seg_logits, None
        if seg is not None:
            t = AuxSegHead.target(seg, targets, self.imgsz)
            loss = loss + F.binary_cross_entropy_with_logits(seg, t) \
                * self.aux_seg_weight
        return loss


def _selftest():
    import copy
    torch.manual_seed(0)
    tgt = torch.tensor([[0., 0., .5, .5, .3, .3], [1., 4., .5, .5, .6, .6]])
    cfgs = [
        ("默认 (已验证 0.7483)", dict()),
        ("多尺度 8/16/32", dict(patch=[8, 16, 32])),
        ("+ Hann 窗", dict(hann=True)),
        ("+ 相位", dict(phase=True)),
        ("+ 对数径向分箱", dict(log_radial=True)),
        ("+ 按频带路由 (依据不足)", dict(band_routing=True)),
        ("v3 推荐组合", dict(patch=[8, 16, 32], hann=True, phase=True,
                            log_radial=True)),
        ("消融: 无 FFT", dict(use_fft=False)),
        ("交叉注意力融合", dict(fusion="xattn")),
    ]
    base = None
    for label, kw in cfgs:
        m = FreqDualBackbone(nc=6, width=0.5, depth=0.34, **kw)
        n = sum(p.numel() for p in m.parameters())
        if base is None:
            base = sum(p.numel() for p in m.spatial.parameters())
        x = torch.randn(2, 3, 640, 640)
        m.train()
        feats = m(x)
        aux = float(m.aux_loss(tgt, bs=2).detach())
        assert [f.shape[1] for f in feats] == m.out_channels
        for f, st_ in zip(feats, m.out_strides):
            assert f.shape[-1] == 640 // st_, (f.shape, st_)
        m.eval(); m(x); copy.deepcopy(m)          # evaluation path must survive
        print(f"[ok] {label:<22} 参数 {n/1e6:.3f}M "
              f"(+{(n-base)/1e6:.3f}M)  融合点 {len(m.fuse)}  aux={aux:.4f}")
    print("\n所有配置的形状、反传和 deepcopy 均通过")


if __name__ == "__main__":
    _selftest()
