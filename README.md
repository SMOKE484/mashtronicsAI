# driveway-guard

Driveway camera anomaly detection. Watches a fixed driveway camera feed and
flags candidate hijacking/carjacking events for human review.

**Current scope: weapon-at-window detection only.** The pipeline is being
rebuilt incrementally, one event type at a time, each validated against real
footage before the next is added (see `HANDOVER.md` for the full history and
rationale). Four other event types — struggle, boxing-in, sprint-approach,
multi-directional convergence — are designed and were previously implemented,
but their code is currently retired pending this incremental rebuild. It's
still in git history (`git log`) and comes back one type at a time.

v1 is offline (processes video files, not live streams yet) and scores risk
with a hand-written rule (confidence + sustained-duration debounce) rather
than a trained scoring model — see `.claude/plans` (or ask) for the full
design rationale.

## Setup

```
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

```
python -m driveway_guard.run --video path\to\clip.mp4 --out out\run1\ --weapon-model weapon_model.pt
```

`--weapon-model` is required — see "Weapon detection model" below for how to
get one; none is bundled with the repo. Outputs land in the `--out`
directory: `annotated.mp4` (boxes/tracks/weapon-hit overlay),
`events.json` / `events.csv` (flagged `weapon_at_window` events), and
`run_meta.json`. Run `python -m driveway_guard.run --help` for the full list
of tunable flags (detector confidence/model, weapon-detector proximity/pad
ratio, weapon confidence threshold, minimum sustained duration, cooldown).

## Diagnosing a clip / inspecting individual weapon hits

`scripts/inspect_weapon_hits.py` and `scripts/export_weapon_snapshots.py`
still work against the current weapon-only pipeline — see their own
docstrings (`--help`) for usage; both were built for exactly this kind of
spot-checking. `scripts/diagnose_pipeline.py` and `scripts/inspect_frame.py`
currently **do not run** — they import the retired pose/feature/calibration
modules and will fail at import time until those come back with the next
event type.

## Weapon detection model

`--weapon-model` takes a YOLO checkpoint fine-tuned for weapons. None is
bundled — most ready-made checkpoints found on GitHub/Roboflow Universe
during sourcing didn't hold up: unlicensed research one-offs with no
reported metrics, visibly noisy labels (stray `"0"`/`"d"` classes in one
9k-image Roboflow set), or off-domain taxonomies (`tank`/`missile` classes
in a "weapon detection" set aimed at war footage, per-firearm-brand classes
like `beretta`/`caracal` with no generic `pistol` class, or — found while
vetting a second dataset — a "gun" dataset whose sample images turned out to
be a toy water pistol and a stock product-comparison photo). Instead, train
your own nano checkpoint on Kaggle GPU with `scripts/train_weapon_model.py`.

**Datasets** (merged — see below for why two):
1. [weapon detection cctv v3 dataset](https://universe.roboflow.com/weapon-detection-cctv/weapon-detection-cctv-v3-dataset)
   (Roboflow Universe, CC BY 4.0 — attribution required if this ships
   anywhere public). Despite the page advertising 11 classes, the actual
   exported version is `nc=2`: `person`/`weapon`. Keeping a `person` class in
   the checkpoint matters — `WeaponDetector.detect()` denylists any
   non-threat class name (`person`/`hand`/`phone`) before picking the
   highest-confidence remaining box in a crop, so a checkpoint that can't
   recognize `person` at all would lose that safeguard.
2. [Gun-cctv-detection](https://universe.roboflow.com/dietest/gun-cctv-detection)
   (Roboflow Universe, CC BY 4.0) — 5,149 images, single `Guns` class, added
   to raise weapon-class recall (0.652 on dataset 1 alone — roughly a third
   of true weapons in validation were missed). Picked over several other
   candidates specifically because sampled thumbnails were genuine CCTV-style
   footage (elevated angle, on-frame timestamp/camera-ID overlay) rather than
   close-up stock photos — see `HANDOVER.md` "Part 1" for the full vetting
   table of rejected alternatives.

**`scripts/merge_yolo_datasets.py`** combines both into one `nc=2`
(`person`/`weapon`) dataset — dataset 1's classes already match the target
list; dataset 2's single `Guns` class remaps onto `weapon`. Its own docstring
has the full JSON config shape.

1. Free account at [roboflow.com](https://roboflow.com) (needed to export any
   dataset, including public ones) → open each dataset link above → Download
   Dataset → format **YOLOv8** (or **YOLO11** if offered) → copy the
   generated `roboflow` Python snippet (has your API key baked in — **use a
   fresh/rotated key**, a previous one was accidentally exposed in a shared
   screenshot).
2. In a Kaggle notebook (GPU T4, internet on):
   ```
   !pip install -q roboflow
   # paste both datasets' snippets here, e.g. downloading to
   # /kaggle/working/weapon-detection-cctv-v3-dataset-1 and
   # /kaggle/working/gun-cctv-detection-1
   !git clone https://github.com/SMOKE484/mashtronicsAI.git && cd mashtronicsAI && pip install -q -e .
   !python scripts/merge_yolo_datasets.py --config merge_config.json
   !python scripts/train_weapon_model.py --data /kaggle/working/merged_weapon_dataset/data.yaml --out /kaggle/working/weapon_model --device cuda:0
   ```
   (`merge_config.json` follows the shape documented in
   `scripts/merge_yolo_datasets.py`'s own docstring.)
3. Download `/kaggle/working/weapon_model/weapon_model.pt` and
   `metrics.json` (or, better, **save it as a proper Kaggle Dataset
   immediately** — `/kaggle/working` does not survive across separate Kaggle
   sessions, only within one continuous session/disk). Check `metrics.json`'s
   per-class precision/recall/mAP50 against the previous checkpoint's
   (`person`: 0.891/0.793/0.859; `weapon`: 0.833/0.652/0.744) before adopting
   it — only swap in the new checkpoint if it's actually better, and still
   spot-check it against real driveway/CCTV clips via `--weapon-model
   weapon_model.pt` before trusting a positive detection to raise risk score.
   If it's still weak, the single-class CC0
   [Pistols Dataset](https://public.roboflow.com/object-detection/pistols)
   (University of Granada, ~2973 images) or the larger but mixed-quality
   [gun-detection-1fbbu](https://universe.roboflow.com/gun-detection-1lttj/gun-detection-1fbbu)
   (9,256 images) are documented fallbacks/top-ups — see `HANDOVER.md` "Part 1".

## Status

Weapon-at-window detection: tracker (YOLO11 + ByteTrack) → proximity-gated
weapon detector → duration-debounced event, keyed per-vehicle (survives
person-track-ID churn — see `HANDOVER.md`/plan for why). Not yet re-validated
against real footage since the rewrite (see `HANDOVER.md` "Validation against
real footage" for the exact next steps). Struggle / boxing-in / sprint /
convergence are designed, previously implemented, and retired pending their
own turn in this incremental rebuild.
