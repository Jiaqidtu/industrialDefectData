"""Layer-by-layer diff of the inference path against ultralytics.

Same weights, same images.  ultralytics scores runs/ultra_nomosaic at 0.6725
while our pipeline scores it at 0.36-0.41, so something between the network
output and the final metric differs.  This walks the three stages in order --
decode, NMS, metric -- and reports where the two sides stop agreeing.

    python -m model.predcheck --weights runs/ultra_nomosaic/weights/best.pt
"""

import argparse

import torch

from .dataset import build_dataloader
from .head import non_max_suppression as my_nms
from .metrics import DetMetrics
from .refarch import RefArchModel, RefHeadShim
from .val import xywhn_to_xyxy


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="runs/ultra_nomosaic/weights/best.pt")
    ap.add_argument("--data", default="neu-det-yolo")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--nc", type=int, default=6)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)

    from ultralytics.utils import ops
    try:
        from ultralytics.utils.nms import non_max_suppression as ref_nms
    except ImportError:                      # older layout
        from ultralytics.utils.ops import non_max_suppression as ref_nms

    ck = torch.load(args.weights, map_location="cpu", weights_only=False)
    src = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    sd = {k: v.float() for k, v in src.state_dict().items()}
    model = RefArchModel({"nc": args.nc, "imgsz": args.imgsz}).to(args.device)
    model.det.load_state_dict(sd, strict=False)
    model.eval()

    loader = build_dataloader(args.data, args.split, imgsz=args.imgsz,
                              batch_size=args.batch_size, augment=False,
                              shuffle=False, workers=2, normalize=False)

    m_ours, m_theirs = DetMetrics(model.names), DetMetrics(model.names)
    d_box = d_score = 0.0
    n_ours = n_theirs = 0

    with torch.no_grad():
        for imgs, targets, _ in loader:
            imgs = imgs.to(args.device)
            y, preds = model.det(imgs)             # y: their decoded (B, 4+nc, A)

            # ---- stage 1: decode ----------------------------------------- #
            ours = model.head.decode(preds)        # (B, A, 4+nc) xyxy + sigmoid
            theirs = y.permute(0, 2, 1)            # (B, A, 4+nc) xywh + sigmoid
            t_box = ops.xywh2xyxy(theirs[..., :4])
            d_box = max(d_box, float((ours[..., :4] - t_box).abs().max()))
            d_score = max(d_score, float((ours[..., 4:] - theirs[..., 4:]).abs().max()))

            # ---- stage 2: NMS -------------------------------------------- #
            det_ours = my_nms(ours, args.conf, args.iou, 300, multi_label=True)
            # nc=0 is what DetectionValidator passes for the detect task
            det_theirs = ref_nms(y, args.conf, args.iou, nc=0,
                                 multi_label=True, max_det=300)

            # ---- stage 3: metric ----------------------------------------- #
            for i in range(imgs.shape[0]):
                lab = xywhn_to_xyxy(targets[targets[:, 0] == i][:, 1:].to(args.device),
                                    args.imgsz)
                m_ours.update(det_ours[i], lab)
                m_theirs.update(det_theirs[i], lab)
                n_ours += det_ours[i].shape[0]
                n_theirs += det_theirs[i].shape[0]

    print(f"\n[1 解码] 框最大差 {d_box:.4e}   分数最大差 {d_score:.4e}")
    print(f"[2 NMS ] 保留框总数  我们 {n_ours}   他们 {n_theirs}")
    r_ours, r_theirs = m_ours.compute(), m_theirs.compute()
    print(f"[3 指标] 同一套指标下  我们的NMS {r_ours['mAP50']:.4f}   "
          f"他们的NMS {r_theirs['mAP50']:.4f}")
    print("\n判读：解码有差 -> decode 的问题；解码一致但两个 mAP 有差 -> NMS 的问题；"
          "两个 mAP 都远低于 0.6725 -> 指标或真值的问题")


if __name__ == "__main__":
    main()
