"""Numerically diff our TAL assigner and DFL decode against ultralytics.

Both implementations get byte-identical inputs; anything that differs in the
outputs is a bug in ours.  This replaces "the code looks equivalent" -- which
has been wrong five times in a row on this project -- with an elementwise
comparison against a reference that reaches 0.71 mAP on the same data.

    python -m model.crosscheck
"""

import argparse

import torch

from .head import bbox2dist, dist2bbox, make_anchors
from .loss import TaskAlignedAssigner


def build_inputs(bs=2, nc=6, imgsz=640, strides=(8, 16, 32), reg_max=16, seed=0):
    """Realistic predictions plus GT boxes spanning the NEU-DET size range."""
    torch.manual_seed(seed)
    feats = [torch.randn(bs, 8, imgsz // s, imgsz // s) for s in strides]
    n_anchors = sum((imgsz // s) ** 2 for s in strides)

    pred_scores = torch.randn(bs, n_anchors, nc) * 0.5 - 4.0     # logits
    pred_distri = torch.randn(bs, n_anchors, 4 * reg_max) * 0.5

    # one small, one medium and one very large box (pitted_surface sized)
    gt = torch.tensor([
        [[0.0,  40.,  60., 130., 210.],      # small, class 0
         [1.0, 200., 180., 390., 400.],      # medium, class 1
         [3.0,  60.,  10., 495., 618.]],     # huge, class 3
    ]).repeat(bs, 1, 1)
    gt_labels, gt_bboxes = gt[..., :1], gt[..., 1:]
    mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
    return feats, pred_scores, pred_distri, gt_labels, gt_bboxes, mask_gt


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--reg-max", type=int, default=16)
    ap.add_argument("--nc", type=int, default=6)
    args = ap.parse_args(argv)

    try:
        from ultralytics.utils.tal import TaskAlignedAssigner as RefAssigner
        from ultralytics.utils.tal import bbox2dist as ref_bbox2dist
        from ultralytics.utils.tal import dist2bbox as ref_dist2bbox
        from ultralytics.utils.tal import make_anchors as ref_make_anchors
    except ImportError as e:
        raise SystemExit(f"需要 ultralytics 作为参照: {e}")

    strides = (8, 16, 32)
    feats, pred_scores, pred_distri, gt_labels, gt_bboxes, mask_gt = build_inputs(
        nc=args.nc, imgsz=args.imgsz, reg_max=args.reg_max)
    rm = args.reg_max

    # ---- 1. anchors ------------------------------------------------------ #
    ref_pts, ref_str = ref_make_anchors(feats, torch.tensor(strides, dtype=torch.float))
    my_pts, my_str = make_anchors(feats, strides)
    d_anchor = (ref_pts * ref_str - my_pts).abs().max()
    d_stride = (ref_str - my_str).abs().max()
    print(f"[anchors] 像素坐标最大差 {d_anchor:.3e}   stride 最大差 {d_stride:.3e}")

    # ---- 2. DFL decode --------------------------------------------------- #
    proj = torch.arange(rm, dtype=torch.float)
    b, a, _ = pred_distri.shape
    ref_dist = pred_distri.view(b, a, 4, rm).softmax(3).matmul(proj)
    ref_boxes = ref_dist2bbox(ref_dist, ref_pts, xywh=False)         # grid units

    my_reg = pred_distri.permute(0, 2, 1)                            # (B, 4*rm, A)
    d = my_reg.view(b, 4, rm, a).permute(0, 3, 1, 2).softmax(-1).matmul(proj)
    my_boxes = dist2bbox(d, (my_pts / my_str).unsqueeze(0))
    print(f"[decode ] 框坐标最大差 {(ref_boxes - my_boxes).abs().max():.3e}")

    # ---- 3. bbox2dist ---------------------------------------------------- #
    tgt = gt_bboxes[:, :1].expand(-1, a, -1) / my_str
    ref_ltrb = ref_bbox2dist(ref_pts, tgt, rm - 1)
    my_ltrb = bbox2dist(tgt, (my_pts / my_str).unsqueeze(0), rm)
    print(f"[bbox2dist] 最大差 {(ref_ltrb - my_ltrb).abs().max():.3e}")

    # ---- 4. assigner ----------------------------------------------------- #
    anchors_px = my_pts
    pd_scores = pred_scores.sigmoid()
    pd_boxes_px = my_boxes * my_str

    ref = RefAssigner(topk=10, num_classes=args.nc, alpha=0.5, beta=6.0)
    r_labels, r_boxes, r_scores, r_fg, _ = ref(
        pd_scores, pd_boxes_px, anchors_px, gt_labels, gt_bboxes, mask_gt)

    mine = TaskAlignedAssigner(topk=10, num_classes=args.nc)
    m_labels, m_boxes, m_scores, m_fg = mine(
        pd_scores, pd_boxes_px, anchors_px, gt_labels, gt_bboxes, mask_gt)

    print(f"\n[assigner] 正样本数  参考 {int(r_fg.sum())}   我们 {int(m_fg.sum())}")
    print(f"[assigner] fg_mask 不一致的 anchor 数 "
          f"{int((r_fg.bool() ^ m_fg.bool()).sum())}")
    print(f"[assigner] target_scores  和: 参考 {float(r_scores.sum()):.4f} "
          f"我们 {float(m_scores.sum()):.4f}   最大差 "
          f"{float((r_scores - m_scores).abs().max()):.3e}")
    both = r_fg.bool() & m_fg.bool()
    if both.any():
        print(f"[assigner] 共同正样本的 target_bbox 最大差 "
              f"{float((r_boxes[both] - m_boxes[both]).abs().max()):.3e}")

    # positives per pyramid level -- the collapse we keep observing
    start = 0
    print(f"\n{'层级':<10}{'参考正样本':>12}{'我们正样本':>12}")
    for s in strides:
        n = (args.imgsz // s) ** 2
        print(f"stride {s:<4}{int(r_fg[:, start:start + n].sum()):>12}"
              f"{int(m_fg[:, start:start + n].sum()):>12}")
        start += n
    print("\n任何一行不一致 -> 差异就在那一处；全部一致 -> 分配器没问题，去查 neck/head/训练循环")


if __name__ == "__main__":
    main()
