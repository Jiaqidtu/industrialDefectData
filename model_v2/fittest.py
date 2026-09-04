"""Minimal reproduction: can the detector fit ONE box of a given size?

Trains on a single synthetic sample until convergence and reports the box it
predicts.  Run for several ground-truth sizes: if the small ones converge and
the large ones saturate, the defect is in the head/loss, not in the data, the
backbone or the training recipe.

    python3 -m model.fittest
    python3 -m model.fittest --steps 400 --reg-max 32
"""

import argparse

import torch

from .head import non_max_suppression
from .yolo_sam2 import build_model


def run_one(frac_w, frac_h, steps, imgsz, reg_max, use_obj, device, lr=2e-3,
            seed=0):
    torch.manual_seed(seed)
    model = build_model(
        imgsz=imgsz,
        backbone={"type": "cnn", "width": 0.25, "depth": 0.34},
        neck={"width": 64}, head={"reg_max": reg_max, "use_obj": use_obj},
    ).to(device)
    model.train()

    # a fixed random image and one centred box of the requested size
    x = torch.randn(1, 3, imgsz, imgsz, device=device) * 0.5
    x[:, :, int(imgsz * (0.5 - frac_h / 2)):int(imgsz * (0.5 + frac_h / 2)),
      int(imgsz * (0.5 - frac_w / 2)):int(imgsz * (0.5 + frac_w / 2))] += 2.0
    targets = torch.tensor([[0, 0, 0.5, 0.5, frac_w, frac_h]], device=device)

    opt = torch.optim.AdamW(model.trainable_parameters(), lr=lr)
    for i in range(steps):
        loss, items = model(x, targets)
        opt.zero_grad()
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        det = non_max_suppression(model(x), conf_thres=0.05, iou_thres=0.65)[0]
    gt = torch.tensor([[imgsz * (0.5 - frac_w / 2), imgsz * (0.5 - frac_h / 2),
                        imgsz * (0.5 + frac_w / 2), imgsz * (0.5 + frac_h / 2)]],
                      device=device)
    if det.shape[0] == 0:
        return None, 0.0, items, gt[0]
    from .head import box_iou
    iou = box_iou(gt, det[:, :4])[0]
    k = int(iou.argmax())
    return det[k], float(iou[k]), items, gt[0]


def main(argv=None):
    ap = argparse.ArgumentParser("single-box fitting test")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--reg-max", type=int, default=16)
    ap.add_argument("--no-obj", dest="use_obj", action="store_false", default=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)

    sizes = [(0.10, 0.10), (0.30, 0.30), (0.50, 0.50),
             (0.68, 0.95), (0.95, 0.95)]
    print(f"reg_max={args.reg_max} use_obj={args.use_obj} steps={args.steps} "
          f"imgsz={args.imgsz}")
    print(f"\n{'GT w x h (px)':>16}{'pred w x h (px)':>18}{'IoU':>7}"
          f"{'box':>8}{'cls':>8}{'dfl':>8}")
    for fw, fh in sizes:
        det, iou, items, gt = run_one(fw, fh, args.steps, args.imgsz,
                                      args.reg_max, args.use_obj, args.device)
        gw, gh = float(gt[2] - gt[0]), float(gt[3] - gt[1])
        if det is None:
            print(f"{gw:>7.0f} x{gh:>6.0f}{'(无预测)':>18}{0.0:>7.3f}"
                  f"{items['box']:>8.3f}{items['cls']:>8.3f}{items['dfl']:>8.3f}")
            continue
        pw, ph = float(det[2] - det[0]), float(det[3] - det[1])
        print(f"{gw:>7.0f} x{gh:>6.0f}{pw:>9.0f} x{ph:>6.0f}{iou:>7.3f}"
              f"{items['box']:>8.3f}{items['cls']:>8.3f}{items['dfl']:>8.3f}")
    print("\n小框 IoU 高、大框 IoU 低 => 缺陷在检测头/损失对大目标的处理")
    print("全部都低 => 缺陷更基础，在分配器或解码")


if __name__ == "__main__":
    main()
