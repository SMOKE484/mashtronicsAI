# driveway-guard

Driveway camera anomaly detection. Watches a fixed driveway camera feed and
flags candidate hijacking/carjacking events for human review:

- struggle / aggressive contact near a vehicle
- a second vehicle "boxing in" the resident's car (blocking the egress path)
- a weapon visible at a vehicle window
- a sprint approach toward a stationary vehicle
- multi-directional convergence (2+ people closing in from different angles)

v1 is offline (processes video files, not live streams yet) and scores risk
with hand-written rules over detection/tracking/pose features rather than a
trained model — see `.claude/plans` (or ask) for the full design rationale.

## Setup

```
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

```
python -m driveway_guard.run --video path\to\clip.mp4 --out out\run1\
```

Outputs land in the `--out` directory: `annotated.mp4` (boxes/tracks/overlay),
and, once later milestones land, `events.json` / `events.csv` (flagged
candidate events) and `run_meta.json`.

## Weapon detection model

`--weapon-model` takes a YOLO checkpoint fine-tuned for weapons; the stage
is skipped entirely if omitted. None is bundled — most ready-made checkpoints
found on GitHub/Roboflow Universe during sourcing didn't hold up: unlicensed
research one-offs with no reported metrics, visibly noisy labels (stray
`"0"`/`"d"` classes in one 9k-image Roboflow set), or off-domain taxonomies
(`tank`/`missile` classes in a "weapon detection" set aimed at war footage,
per-firearm-brand classes like `beretta`/`caracal` with no generic `pistol`
class). Instead, train your own nano checkpoint on Kaggle GPU with
`scripts/train_weapon_model.py`.

**Dataset**: [weapon detection cctv v3 dataset](https://universe.roboflow.com/weapon-detection-cctv/weapon-detection-cctv-v3-dataset)
(Roboflow Universe, CC BY 4.0 — attribution required if this ships anywhere
public) — ~2.6k images, 11 classes including `pistol`, `gun`, `Knife`, and
`rifle`, purpose-built for CCTV surveillance footage. Its inclusion of
`hand` and `phone` as separate classes is the main reason it was picked over
larger alternatives: phone-in-hand near a car window is a realistic
false-positive this project's design notes already flagged, so training the
model to actively distinguish "phone" from "gun" is worth more here than
raw image count. (The pipeline itself doesn't care which of the 11 class
names fires — `WeaponDetector.detect()` just takes the highest-confidence
box in a gated crop — so this was a straight swap-in for the single-class
option, no pipeline code changes needed.)

1. Free account at [roboflow.com](https://roboflow.com) (needed to export
   any dataset, including public ones) → open the dataset link above →
   Download Dataset → format **YOLOv8** (or **YOLO11** if offered) → copy
   the generated `roboflow` Python snippet (has your API key baked in).
2. In a Kaggle notebook (GPU T4, internet on):
   ```
   !pip install -q roboflow
   # paste the snippet from step 1 here, downloading to /kaggle/working/weapons
   !git clone https://github.com/SMOKE484/mashtronicsAI.git && cd mashtronicsAI && pip install -q -e .
   !python scripts/train_weapon_model.py --data /kaggle/working/weapons/data.yaml --out /kaggle/working/weapon_model --device cuda:0
   ```
3. Download `/kaggle/working/weapon_model/weapon_model.pt` and
   `metrics.json`. Check `metrics.json` for precision/recall/mAP50 — this is
   the highest-uncertainty component of v1 (small/dark-object detection,
   now across 11 classes on only ~2.6k images, some classes may be thin),
   so treat these numbers as a sanity check, not a pass/fail: still spot-check
   the checkpoint against real driveway/CCTV clips via `--weapon-model
   weapon_model.pt` before trusting a positive detection to raise risk score.
   If per-class precision on `pistol`/`gun`/`rifle`/`Knife` looks weak in
   `metrics.json` or in spot-checks, the single-class CC0
   [Pistols Dataset](https://public.roboflow.com/object-detection/pistols)
   (University of Granada, ~2973 images) is the fallback — same script,
   swap `--data` to point at it instead.

## Status

Milestone 1-2 (detection + ByteTrack tracking + annotated video output).
Pose gating, calibration/geometry, feature extraction, weapon detection, and
the rule-based scorer land in subsequent milestones.
