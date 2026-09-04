"""Reference baseline: train YOLOv8 on NEU-DET with ultralytics.

This exists to answer one question -- is the ~0.31-0.37 mAP we get from our
own implementation a property of the data, or a defect in our code?  A widely
used implementation on the identical split settles it, and the number is a
baseline the paper needs anyway.

    python3 ref_yolov8.py                    # yolov8n, 150 epochs
    python3 ref_yolov8.py --model yolov8s.pt --epochs 200
"""

import argparse
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "neu-det-abs.yaml"))
    ap.add_argument("--model", default="yolov8n.pt",
                    help="a .pt (COCO-pretrained) or a .yaml (from scratch)")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--name", default="ultra_ref")
    ap.add_argument("--project", default=os.path.join(HERE, "runs"))
    ap.add_argument("--device", default=0)
    ap.add_argument("--seed", type=int, default=0,
                    help="ultralytics is deterministic at a fixed seed, so "
                         "without this every repeat returns the same number "
                         "and the standard deviation is a fabricated zero")
    # augmentation knobs, so the reference can be matched to our pipeline
    ap.add_argument("--mosaic", type=float, default=1.0)
    ap.add_argument("--hsv", type=float, default=None,
                    help="0 disables all three HSV gains")
    ap.add_argument("--degrees", type=float, default=0.0)
    ap.add_argument("--no-amp", dest="amp", action="store_false", default=True,
                    help="fp32 training.  RT-DETR lost its EMA to a NaN at "
                         "epoch 107 under AMP: train losses stayed healthy "
                         "while every val metric read 0 from there on, since "
                         "ultralytics validates the EMA copy.")
    ap.add_argument("--nbs", type=int, default=64,
                    help="nominal batch size for gradient accumulation")
    args = ap.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError as e:
        # never swallow the real cause: ultralytics pulls in torchvision and a
        # dozen other packages, and any one of them failing surfaces here
        import traceback
        traceback.print_exc()
        raise SystemExit(
            f"\nultralytics 导入失败，真实原因见上方 traceback: {e}\n"
            f"python: {os.sys.executable} {os.sys.version.split()[0]}")

    # RT-DETR is the DETR-family entry point in ultralytics and needs its own
    # class; YOLO() silently mis-parses the checkpoint otherwise.  This is the
    # only published transformer detector we can run under our own protocol,
    # which is what makes it the right stand-in for DSAT (no code released).
    if os.path.basename(str(args.model)).startswith("rtdetr"):
        from ultralytics import RTDETR
        model = RTDETR(args.model)
    else:
        model = YOLO(args.model)
    extra = {}
    if args.hsv is not None:
        extra.update(hsv_h=args.hsv, hsv_s=args.hsv, hsv_v=args.hsv)
    model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz,
                mosaic=args.mosaic, degrees=args.degrees, nbs=args.nbs,
                seed=args.seed, amp=args.amp, **extra,
                batch=args.batch, workers=args.workers, device=args.device,
                project=args.project, name=args.name, exist_ok=True,
                pretrained=str(args.model).endswith(".pt"), val=True, plots=False)

    metrics = model.val(data=args.data, imgsz=args.imgsz, batch=args.batch,
                        workers=args.workers, device=args.device,
                        project=args.project, name=args.name + "_val",
                        exist_ok=True)

    names = model.names
    print(f"\n{'class':<18}{'AP50':>9}{'AP50-95':>10}")
    for i, ap50 in enumerate(metrics.box.ap50):
        c = metrics.box.ap_class_index[i]
        print(f"{names[int(c)]:<18}{ap50:>9.4f}{metrics.box.ap[i]:>10.4f}")
    print(f"{'all':<18}{metrics.box.map50:>9.4f}{metrics.box.map:>10.4f}")
    print("\n对照我们自己的实现：i_cnn_fixed mAP50 = 0.3678, "
          "e_mosaic(SAM2) = 0.3311, g_unfrozen(SAM2) = 0.3805")
    print("如果这里明显更高 -> 我们的实现有 bug；如果也在 0.3-0.4 -> 问题在数据")


if __name__ == "__main__":
    main()
