"""Re-run ultralytics' own validator on a checkpoint, right now.

predcheck proved our decode, NMS and detections match ultralytics exactly on
these weights, yet our metric says 0.359 where the training run reported
0.6725.  Every component being identical means one of the premises is wrong,
and the untested premise is that 0.6725 belongs to this checkpoint on this
split.  This measures it directly.

    python refval.py --weights runs/ultra_nomosaic/weights/best.pt
"""

import argparse
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="runs/ultra_nomosaic/weights/best.pt")
    ap.add_argument("--data", default=os.path.join(HERE, "neu-det-abs.yaml"))
    ap.add_argument("--split", default="val")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.7)
    args = ap.parse_args()

    from ultralytics import YOLO

    m = YOLO(args.weights)
    r = m.val(data=args.data, split=args.split, imgsz=args.imgsz,
              batch=args.batch, conf=args.conf, iou=args.iou, plots=False)

    print(f"\nmAP50 {r.box.map50:.4f}   mAP50-95 {r.box.map:.4f}")


if __name__ == "__main__":
    main()
