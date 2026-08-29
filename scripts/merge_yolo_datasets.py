"""Merge multiple Roboflow-exported YOLO detection datasets into one, with
per-source class-name remapping onto a single unified class list.

Built for combining the existing weapon-detection-cctv-v3-dataset (classes
person/weapon) with a second, weapon-only dataset (e.g. a single `Guns`
class) without losing the `person` class the first dataset provides -- see
HANDOVER.md "Part 1" for why the `person` class matters (WeaponDetector's
non-threat-class denylist depends on it existing in the checkpoint).

Each source dataset must already be a standard Roboflow YOLO export: a
`data.yaml` at the root, with `train/images/`, `train/labels/`, and
`valid/images/`, `valid/labels/` (and optionally `test/...`) alongside it.

Usage:
    python scripts/merge_yolo_datasets.py --config merge_config.json

Where merge_config.json looks like:
{
  "target_classes": ["person", "weapon"],
  "output": "/kaggle/working/merged_weapon_dataset",
  "sources": [
    {
      "name": "cctv_v3",
      "data_yaml": "/kaggle/working/weapon-detection-cctv-v3-dataset-1/data.yaml",
      "class_map": {"person": "person", "weapon": "weapon"}
    },
    {
      "name": "gun_cctv",
      "data_yaml": "/kaggle/working/gun-cctv-detection-1/data.yaml",
      "class_map": {"Guns": "weapon"}
    }
  ]
}

`class_map` maps each of that source's own class names (as spelled in its
own data.yaml) to a name in `target_classes`. Every class name the source
actually has must appear in its `class_map` -- this fails loudly rather than
silently dropping an unmapped class. A `class_map` value of `null` means
"drop boxes of this class" (the image is still copied, as a background/
negative example, if nothing else remains in its label file) -- use this to
exclude a source class you don't want folded into any target class, e.g.
{"Handgun": "weapon", "Knife": null, "Short_rifle": null}.

An optional per-source `max_area_norm` dict caps how large (as a fraction of
image area, width_norm * height_norm) a box may be for a given *target*
class before it's dropped -- added to filter out a source whose boxes are
scaled very differently from the rest (e.g. close-up product/stock photos
mixed into an otherwise-CCTV dataset, where a "weapon" box can cover 20%+ of
the frame vs. under 1% for an actual distant gun sighting), without having
to drop that source's contribution to the class entirely:
{"max_area_norm": {"weapon": 0.02}}
"""

import argparse
import json
import logging
import shutil
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

_SPLITS = ("train", "valid", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge YOLO datasets with class remapping")
    parser.add_argument("--config", required=True, type=Path)
    return parser.parse_args()


def _load_source_names(data_yaml_path: Path) -> list[str]:
    """Returns class names in local-index order, handling both the list and
    dict `names:` forms different Ultralytics/Roboflow export versions use."""
    data = yaml.safe_load(data_yaml_path.read_text())
    names = data["names"]
    if isinstance(names, dict):
        return [names[i] for i in sorted(names, key=int)]
    return list(names)


def build_class_remap(
    source_names: list[str], class_map: dict[str, str | None], target_classes: list[str]
) -> dict[int, int | None]:
    """Maps each source-local class index to a target-class index, per
    class_map. A class_map value of None means "drop boxes of this class"
    rather than remap them. Raises if any source class name has no entry in
    class_map, or maps to a name not present in target_classes -- no silent
    drops of a class missing from class_map entirely."""
    remap: dict[int, int | None] = {}
    for local_id, name in enumerate(source_names):
        if name not in class_map:
            raise ValueError(f"source class {name!r} has no entry in class_map")
        target_name = class_map[name]
        if target_name is None:
            remap[local_id] = None
            continue
        if target_name not in target_classes:
            raise ValueError(f"class_map target {target_name!r} not in target_classes")
        remap[local_id] = target_classes.index(target_name)
    return remap


def remap_label_line(
    line: str, remap: dict[int, int | None], max_area_norm: dict[int, float] | None = None
) -> str | None:
    """Rewrites a YOLO label line's leading class-index per remap. Returns
    None to drop the line -- either class_map explicitly maps this class to
    None, the box's area exceeds max_area_norm for its target class, or
    (shouldn't happen if build_class_remap covered every source class) the
    id isn't in remap at all."""
    parts = line.split()
    if not parts:
        return None
    local_id = int(parts[0])
    if local_id not in remap:
        logger.warning("label line references unmapped class id %d, skipping line", local_id)
        return None
    target_id = remap[local_id]
    if target_id is None:
        return None
    if max_area_norm and target_id in max_area_norm:
        width_norm, height_norm = float(parts[3]), float(parts[4])
        if width_norm * height_norm > max_area_norm[target_id]:
            return None
    parts[0] = str(target_id)
    return " ".join(parts)


def _merge_source(source: dict, target_classes: list[str], output: Path) -> dict[str, int]:
    data_yaml = Path(source["data_yaml"])
    source_root = data_yaml.parent
    prefix = source["name"]
    source_names = _load_source_names(data_yaml)
    remap = build_class_remap(source_names, source["class_map"], target_classes)
    max_area_norm = {
        target_classes.index(name): area for name, area in source.get("max_area_norm", {}).items()
    }

    counts: dict[str, int] = {}
    for split in _SPLITS:
        images_dir = source_root / split / "images"
        labels_dir = source_root / split / "labels"
        if not images_dir.is_dir():
            continue

        out_images_dir = output / split / "images"
        out_labels_dir = output / split / "labels"
        out_images_dir.mkdir(parents=True, exist_ok=True)
        out_labels_dir.mkdir(parents=True, exist_ok=True)

        n = 0
        for image_path in images_dir.iterdir():
            if not image_path.is_file():
                continue
            out_name = f"{prefix}_{image_path.name}"
            shutil.copy2(image_path, out_images_dir / out_name)
            n += 1

            label_path = labels_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                continue
            remapped_lines = [
                remapped
                for line in label_path.read_text().splitlines()
                if (remapped := remap_label_line(line, remap, max_area_norm)) is not None
            ]
            (out_labels_dir / f"{prefix}_{image_path.stem}.txt").write_text(
                "\n".join(remapped_lines) + ("\n" if remapped_lines else "")
            )
        counts[split] = n
    return counts


def main() -> None:
    args = parse_args()
    logging.basicConfig(level="INFO", format="%(message)s")

    config = json.loads(args.config.read_text())
    target_classes = config["target_classes"]
    output = Path(config["output"])

    total_counts: dict[str, int] = {split: 0 for split in _SPLITS}
    for source in config["sources"]:
        logger.info("merging source %s (%s)", source["name"], source["data_yaml"])
        counts = _merge_source(source, target_classes, output)
        for split, n in counts.items():
            total_counts[split] += n
        logger.info("  %s", counts)

    data_yaml_out = {
        "train": str(output / "train" / "images"),
        "val": str(output / "valid" / "images"),
        "nc": len(target_classes),
        "names": target_classes,
    }
    if (output / "test" / "images").is_dir():
        data_yaml_out["test"] = str(output / "test" / "images")
    (output / "data.yaml").write_text(yaml.safe_dump(data_yaml_out, sort_keys=False))

    logger.info("Merged dataset written to %s", output)
    logger.info("Total images per split: %s", total_counts)
    logger.info("Wrote %s", output / "data.yaml")


if __name__ == "__main__":
    main()
