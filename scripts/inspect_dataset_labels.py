"""Inspect ground-truth label boxes for one class in a YOLO-format dataset,
to sanity-check *what the model is actually being taught*, not just how it
performs after training.

Written to investigate why the Part-1 retrained weapon checkpoint keys off
dark/reflective surfaces (car roof, side mirror) instead of resolved gun
shapes -- see HANDOVER.md "Validation against real footage -- outcome, then
retracted". The suspicion is a box-shape/quality problem in the added
`dietest/gun-cctv-detection` source (renamed `gun_cctv_*` by
merge_yolo_datasets.py) relative to the original `cctv_v3_*` source: if that
source's `Handgun`/`Short_rifle` boxes are oversized, low-contrast, or
elongated, the model may have learned "large dark blob near a person" as a
proxy for "weapon" rather than an actual gun shape.

Since merge_yolo_datasets.py prefixes every copied filename with its source
name (`{source}_{original_name}`), this script can split the *merged*
dataset back out by source just from filenames -- no need to keep the
original per-source exports around. Pass --group-by-prefix to enable this
(splits each label's source group on the first `_`).

For each group, prints box-shape stats (width/height as a fraction of image
size, aspect ratio, area fraction) and saves:
- a sample of --sample random boxes (crop + full frame, box drawn), for
  eyeballing whether the ground truth looks like a real gun or not.
- the --extreme most elongated boxes by aspect ratio, since an elongated
  dark region is the specific shape a car roofline / mirror strip would
  falsely match.

Usage:
    python scripts/inspect_dataset_labels.py --data merged_weapon_dataset/data.yaml \
        --class weapon --split train --out out/label_inspection --group-by-prefix
"""

import argparse
import logging
import random
import statistics
from pathlib import Path

import cv2
import yaml

logger = logging.getLogger(__name__)

_BOX_COLOR = (0, 0, 255)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect ground-truth label boxes for one class")
    parser.add_argument("--data", required=True, type=Path, help="Path to data.yaml")
    parser.add_argument("--class", dest="class_name", required=True, help="Class name to inspect")
    parser.add_argument("--split", default="train", choices=["train", "valid", "test"])
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--sample", type=int, default=24, help="Random boxes to export per group")
    parser.add_argument("--extreme", type=int, default=8, help="Most-elongated boxes to export per group")
    parser.add_argument("--crop-pad-ratio", type=float, default=0.6)
    parser.add_argument(
        "--group-by-prefix",
        action="store_true",
        help="Split boxes by the source prefix merge_yolo_datasets.py stamped on each filename",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _load_class_index(data_yaml: Path, class_name: str) -> int:
    data = yaml.safe_load(data_yaml.read_text())
    names = data["names"]
    if isinstance(names, dict):
        names = [names[i] for i in sorted(names, key=int)]
    if class_name not in names:
        raise ValueError(f"class {class_name!r} not in {names}")
    return names.index(class_name)


def _group_for(image_name: str, group_by_prefix: bool) -> str:
    if not group_by_prefix or "_" not in image_name:
        return "all"
    return image_name.split("_", 1)[0]


class Box:
    __slots__ = ("image_path", "group", "xyxy", "w_norm", "h_norm")

    def __init__(self, image_path: Path, group: str, xyxy: tuple[float, float, float, float], w_norm: float, h_norm: float):
        self.image_path = image_path
        self.group = group
        self.xyxy = xyxy
        self.w_norm = w_norm
        self.h_norm = h_norm

    @property
    def aspect_ratio(self) -> float:
        """>1 means wider than tall; always expressed as the longer side over the shorter side."""
        return max(self.w_norm, self.h_norm) / max(min(self.w_norm, self.h_norm), 1e-9)

    @property
    def area_norm(self) -> float:
        return self.w_norm * self.h_norm


def _collect_boxes(data_yaml: Path, split: str, class_idx: int, group_by_prefix: bool) -> list[Box]:
    source_root = data_yaml.parent
    images_dir = source_root / split / "images"
    labels_dir = source_root / split / "labels"
    boxes: list[Box] = []
    for label_path in sorted(labels_dir.glob("*.txt")):
        image_path = None
        for ext in (".jpg", ".jpeg", ".png"):
            candidate = images_dir / f"{label_path.stem}{ext}"
            if candidate.exists():
                image_path = candidate
                break
        if image_path is None:
            logger.warning("no image found for label %s, skipping", label_path.name)
            continue

        img = cv2.imread(str(image_path))
        if img is None:
            logger.warning("failed to read %s, skipping", image_path.name)
            continue
        h, w = img.shape[:2]
        group = _group_for(label_path.stem, group_by_prefix)

        for line in label_path.read_text().splitlines():
            parts = line.split()
            if not parts or int(parts[0]) != class_idx:
                continue
            cx, cy, bw, bh = (float(v) for v in parts[1:5])
            x1 = (cx - bw / 2) * w
            y1 = (cy - bh / 2) * h
            x2 = (cx + bw / 2) * w
            y2 = (cy + bh / 2) * h
            boxes.append(Box(image_path, group, (x1, y1, x2, y2), bw, bh))
    return boxes


def _print_stats(group: str, boxes: list[Box]) -> None:
    if not boxes:
        logger.info("group=%s: no boxes", group)
        return
    widths = [b.w_norm for b in boxes]
    heights = [b.h_norm for b in boxes]
    aspects = [b.aspect_ratio for b in boxes]
    areas = [b.area_norm for b in boxes]
    logger.info(
        "group=%s n=%d | width_norm median=%.3f mean=%.3f | height_norm median=%.3f mean=%.3f | "
        "aspect_ratio median=%.2f mean=%.2f max=%.2f | area_norm median=%.4f mean=%.4f",
        group,
        len(boxes),
        statistics.median(widths),
        statistics.mean(widths),
        statistics.median(heights),
        statistics.mean(heights),
        statistics.median(aspects),
        statistics.mean(aspects),
        max(aspects),
        statistics.median(areas),
        statistics.mean(areas),
    )


def _export_box(box: Box, out_dir: Path, stem: str, crop_pad_ratio: float) -> None:
    from driveway_guard.imaging import crop_with_padding

    img = cv2.imread(str(box.image_path))
    h, w = img.shape[:2]
    annotated = img.copy()
    x1, y1, x2, y2 = (int(v) for v in box.xyxy)
    cv2.rectangle(annotated, (x1, y1), (x2, y2), _BOX_COLOR, 2)
    cv2.imwrite(str(out_dir / f"{stem}_full.png"), annotated)
    crop, _ox, _oy = crop_with_padding(annotated, box.xyxy, crop_pad_ratio, w, h)
    cv2.imwrite(str(out_dir / f"{stem}_crop.png"), crop)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level="INFO", format="%(message)s")
    random.seed(args.seed)

    class_idx = _load_class_index(args.data, args.class_name)
    boxes = _collect_boxes(args.data, args.split, class_idx, args.group_by_prefix)
    if not boxes:
        logger.info("No boxes found for class %r in split %r", args.class_name, args.split)
        return

    groups: dict[str, list[Box]] = {}
    for box in boxes:
        groups.setdefault(box.group, []).append(box)

    logger.info("Found %d %r box(es) across %d group(s)", len(boxes), args.class_name, len(groups))
    for group in sorted(groups):
        _print_stats(group, groups[group])

    args.out.mkdir(parents=True, exist_ok=True)
    for group, group_boxes in groups.items():
        group_dir = args.out / group
        (group_dir / "random_sample").mkdir(parents=True, exist_ok=True)
        (group_dir / "most_elongated").mkdir(parents=True, exist_ok=True)

        sample = random.sample(group_boxes, min(args.sample, len(group_boxes)))
        for i, box in enumerate(sample):
            _export_box(box, group_dir / "random_sample", f"{i:03d}_{box.image_path.stem}", args.crop_pad_ratio)

        most_elongated = sorted(group_boxes, key=lambda b: b.aspect_ratio, reverse=True)[: args.extreme]
        for i, box in enumerate(most_elongated):
            _export_box(
                box,
                group_dir / "most_elongated",
                f"{i:03d}_ar{box.aspect_ratio:.1f}_{box.image_path.stem}",
                args.crop_pad_ratio,
            )

        logger.info(
            "group=%s: exported %d random sample(s) and %d most-elongated box(es) to %s",
            group,
            len(sample),
            len(most_elongated),
            group_dir,
        )


if __name__ == "__main__":
    main()
