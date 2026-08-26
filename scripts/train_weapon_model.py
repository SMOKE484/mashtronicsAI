"""Fine-tune a YOLO11 nano checkpoint for weapon detection.

Standalone utility, not part of the `driveway_guard` package — meant to be
run once (on Kaggle GPU) to produce a checkpoint for `--weapon-model`, not
imported at pipeline runtime. Class-agnostic: works with any YOLO-format
dataset regardless of class list, since WeaponDetector.detect() only cares
about the highest-confidence box in a gated crop, not which class fired.

Expects a dataset already exported in YOLO format (a `data.yaml` plus
images/labels dirs). See README.md's "Weapon detection model" section for
the recommended dataset (Roboflow's CCTV-focused multi-class weapon set)
and the full Kaggle workflow.

Usage:
    python scripts/train_weapon_model.py --data /path/to/data.yaml --out /kaggle/working/weapon_model --device cuda:0
"""

import argparse
import json
import logging
import shutil
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune YOLO11n for firearm detection")
    parser.add_argument("--data", required=True, type=Path, help="Path to data.yaml")
    parser.add_argument("--out", required=True, type=Path, help="Output directory")
    parser.add_argument("--base-model", default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=-1, help="-1 = auto batch size")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--patience", type=int, default=15, help="Early-stopping patience")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s: %(message)s")
    logger = logging.getLogger(__name__)

    args.out.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.base_model)
    results = model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        patience=args.patience,
        project=str(args.out),
        name="train",
    )

    run_dir = Path(results.save_dir)
    best_weights = run_dir / "weights" / "best.pt"
    if not best_weights.exists():
        raise FileNotFoundError(f"Expected trained weights at {best_weights}, not found")

    dest_weights = args.out / "weapon_model.pt"
    shutil.copy2(best_weights, dest_weights)

    metrics = model.val(data=str(args.data), device=args.device)
    per_class = {}
    for i, cls_idx in enumerate(metrics.box.ap_class_index):
        p, r, ap50, ap = metrics.box.class_result(i)
        per_class[metrics.names[int(cls_idx)]] = {
            "precision": float(p),
            "recall": float(r),
            "mAP50": float(ap50),
            "mAP50-95": float(ap),
        }
    metrics_summary = {
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "mAP50": float(metrics.box.map50),
        "mAP50-95": float(metrics.box.map),
        "per_class": per_class,
        "epochs_trained": args.epochs,
        "base_model": args.base_model,
        "data": str(args.data),
    }
    (args.out / "metrics.json").write_text(json.dumps(metrics_summary, indent=2))

    logger.info("Weights: %s", dest_weights)
    logger.info("Metrics: %s", json.dumps(metrics_summary, indent=2))
    logger.info(
        "Do not trust this checkpoint off these numbers alone — run it against "
        "real driveway/CCTV clips via --weapon-model and spot-check false positives "
        "before wiring it into anything that scores risk."
    )


if __name__ == "__main__":
    main()
