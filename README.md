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

## Status

Milestone 1-2 (detection + ByteTrack tracking + annotated video output).
Pose gating, calibration/geometry, feature extraction, weapon detection, and
the rule-based scorer land in subsequent milestones.
