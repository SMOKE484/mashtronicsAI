# Handover: driveway-guard (mashtronicsAI)

Context for picking up this project in a new chat session, without the user
re-explaining. Originally written 2026-08-26 at the end of the session that
built v1; updated same day after a follow-up session that picked up weapon
detection. Written to be detailed enough that a new session doesn't need to
re-derive anything by re-reading code or re-running commands that already
have known answers below.

## What this project is

A production-track computer vision system that watches a **fixed driveway
security camera** (target hardware: Dahua CCTV, RTSP-capable — not yet
wired up) and flags candidate carjacking/hijacking events for human review.
Conversation started as a general "what can Dahua cameras do" question,
moved into brainstorming hijacking detection, then narrowed to driveways
specifically (fixed camera + predictable normal patterns made the problem
much more tractable than open street/parking-lot detection).

Five event types, decided collaboratively with the user across the
brainstorm:
1. **Struggle / aggressive contact** near a vehicle (hijacking-style)
2. **Boxing-in** — a second vehicle blocking the resident's car's exit path
   (a known carjacking precursor)
3. **Weapon at window** — a gun visible at a vehicle window
4. **Sprint approach** — someone closing on a stationary vehicle at running
   speed
5. **Multi-directional convergence** — 2+ people approaching a vehicle from
   different angles simultaneously (cutting off escape routes)

## Key design decisions (why things are built the way they are)

- **No labeled training data exists** (real hijacking footage is rare; user
  is sourcing clips independently). So v1 does **not** train a learned
  model for the core risk logic — it's a hand-written **rule-based risk
  scorer** operating on detection/tracking/pose features. Crucially, the
  feature schema (`FrameFeatureVector`) is designed to double as a future
  training row, so this isn't throwaway work once real labeled data
  exists. (Weapon detection is the one sub-component that *is* a trained
  model — see "Weapon detection" section, that's a separate, narrower
  training problem than the overall risk scorer.)
- **v1 processes offline video files, not live RTSP.** Live streaming is an
  explicit, deliberate follow-up — not built yet. `FrameSource` is an ABC
  specifically so a future `RtspStreamSource` is a drop-in swap.
- **Boxing-in got its own record type** (`BlockingObservation`), not fields
  bolted onto the person-vehicle pair record — boxing-in is a
  vehicle-vehicle relationship that doesn't always involve a person, so it
  doesn't fit a schema shaped around person-vehicle pairs. This was a
  deliberate deviation from the original plan sketch, made during
  implementation.
- **Convergence is a per-vehicle aggregate** (`VehicleConvergenceFeatureVector`),
  computed by reusing already-extracted pair records rather than
  recomputing velocities — it's inherently a cross-person signal.
- **Convergence gets a "vehicle fleeing" bonus, added this session** (user
  request: on top of multiple people surrounding the car, also look at
  whether the car is driving/backing out of the driveway — a resident
  attempting to flee under duress while surrounded is a materially
  higher-risk situation than a stationary convergence). Implemented as a
  bonus on the existing `multi_directional_convergence` event rather than a
  new sixth event type, matching the existing `blocking_with_exit_bonus`
  pattern on boxing-in. `features/convergence.py: compute_convergence()`
  now optionally takes `calibration` and projects the vehicle's velocity
  onto the unit `egress_path.direction_vector` (already used for boxing-in
  overlap math — this vector points *out of* the driveway, confirmed from
  `blocking_overlap_ratio`'s corridor-extends-backward-from-exit-point
  logic) via dot product, giving `vehicle_egress_speed_px_s` — positive
  means moving toward the exit. `scoring/rules.py: score_convergence()`
  adds `convergence_fleeing_bonus` (0.35) when that speed clears
  `convergence_egress_speed_px_s_threshold` (60.0 px/s). **Requires
  calibration** (an `egress_path`) to do anything — without it,
  `vehicle_egress_speed_px_s` is `None` and convergence scoring behaves
  exactly as before (no regression, just no bonus).
- **Follow-up fix, same session: the simultaneous-frame bonus above misses
  the scenario the user actually meant.** The user clarified this isn't
  only about the resident fleeing — it also (maybe primarily) covers
  hijackers getting into the car themselves and reversing it out, driver
  possibly still inside, which matters specifically because weapon
  detection has real recall gaps (~65%, see "Weapon detection" section) —
  this needs to work as an independent detection path, not one that only
  fires when a weapon or visible struggle was also caught. The bug: the
  moment people get into the vehicle, they stop being tracked as separate
  "approaching person" tracks, so `num_simultaneous_approachers` drops to 0
  right when the car starts moving — meaning the bonus above, which
  required convergence and egress motion in the *same frame*, would never
  fire for exactly this sequence (surround → get in → drive off).
  **Fix**: decoupled "was this vehicle recently surrounded" from "is it
  moving right now."
  - `compute_convergence()` (`features/convergence.py`) now emits a record
    for *every* currently-tracked vehicle each frame (via a new
    `vehicles` param), not only vehicles someone is actively approaching
    — so `vehicle_egress_speed_px_s` keeps flowing after the people who
    surrounded it are no longer visible as separate tracks. Vehicle
    velocity is sourced from a new public
    `FeatureExtractor.vehicle_velocity_px_s()` (reads the same
    `TrackHistory` that was already being kept per-vehicle, just wasn't
    exposed) via a `vehicle_velocities` dict built in `pipeline.py` and
    passed in alongside `vehicles`.
  - `RuleBasedScorer` (`scoring/rules.py`) now keeps
    `_last_qualifying_convergence: dict[vehicle_id, timestamp]`, updated
    whenever a vehicle's convergence gate (`num_simultaneous_approachers`
    + `angular_spread_deg`) is actually met, independent of the debounce
    state in `EventAggregator`.
  - `score_convergence()` gained a `recently_surrounded: bool` param and a
    third scoring path: if the convergence gate *isn't* met this frame but
    the vehicle *was* recently surrounded (within
    `convergence_recent_window_s`, default 20s) and is now moving along
    egress, it scores `convergence_recent_fleeing_score` (0.85) outright —
    deliberately high on its own, not a small bonus, since "vehicle driven
    off shortly after being surrounded" is treated as a strong signal in
    its own right, not contingent on catching anything else. Still the
    same `multi_directional_convergence` event type, just a different
    qualifying path — not a new sixth event.
  - Covered by 4 new tests in `tests/test_rule_scorer.py`: two pure
    `score_convergence()` cases
    (`test_convergence_recently_surrounded_and_fleeing_scores_without_current_approachers`,
    `test_convergence_zero_when_fleeing_but_not_recently_surrounded`) and
    one full `RuleBasedScorer` integration test
    (`test_scorer_flags_vehicle_driven_off_shortly_after_being_surrounded`)
    that replays surround → everyone gets in → drives off across 4 frames
    and asserts the event only fires on the drive-off, not before.
    47/47 tests passing after this change.
  - **Not yet tested against real footage** — both this and the bonus
    above are code changes verified only by unit tests, not a pipeline
    run. Still requires calibration to do anything.
- **Event debouncing is duration-based (elapsed seconds), not frame-count
  based.** The plan originally sketched `event_min_consecutive_frames`, but
  frame count isn't robust to varying fps or `--frame-stride`, so
  `EventAggregator` in `scoring/events.py` uses `min_duration_s` instead
  (`RuleThresholds.event_min_duration_s` / `weapon_min_duration_s`). This is
  a deliberate implementation improvement over the literal plan text.
- **Weapon detection was originally class-agnostic by design, but that broke
  once the actual checkpoint's classes were known — fixed this session.**
  `WeaponDetector.detect()` (`src/driveway_guard/detection/weapon_detector.py`)
  loops over proximity-gated persons, runs the weapon model on each padded
  crop (a crop centered on the *person*), and originally just did
  `confs.argmax()` — highest-confidence box, full stop, no class check. That
  was fine as long as every class in the checkpoint was weapon-relevant, but
  once training confirmed the actual checkpoint's two classes are literally
  `person`/`weapon` (see "Weapon detection" section below), it became a bug:
  the model reliably detects a `person` box in a crop centered on a person,
  and `confs.argmax()` had no way to know that box wasn't a threat —
  `pipeline.py` sets `weapon_detected = True` off *any* hit regardless of
  class, so this would false-flag `weapon_at_window` for essentially anyone
  standing near a car. **Fix applied**: `detect()` now reads `boxes.cls` and
  `self._model.names`, filters out a small denylist of known non-threat
  class names (`_NON_THREAT_CLASS_NAMES = {"person", "hand", "phone"}`,
  case-insensitive, overridable via the `non_threat_class_names` constructor
  arg) before taking `argmax()`. This keeps the original intent — the
  dataset's exact *threat* taxonomy (however many weapon classes) still
  needs no pipeline code changes — while no longer letting a confirmed
  non-threat class win. `hand`/`phone` are in the default denylist
  preemptively (mentioned in the CCTV dataset's advertised-but-not-actually-
  exported 11-class taxonomy — see "Weapon detection" section) even though
  the trained `nc=2` checkpoint doesn't currently have those classes, so a
  future retrain/dataset swap that reintroduces them doesn't reopen this bug.
  The scorer (`src/driveway_guard/scoring/rules.py: score_weapon`) still only
  reads `record.weapon_detected`/`record.weapon_confidence` off the feature
  record, unchanged — the fix is entirely inside `WeaponDetector.detect()`.
  **Not yet re-tested against real footage post-fix** — this was a code
  read/fix, not a pipeline run.
- **CPU-only locally** (see "Environment" below) — nano YOLO models
  (`yolo11n.pt`, `yolo11n-pose.pt`) were chosen specifically for this. The
  user tests on **Kaggle notebooks** (free T4 GPU) instead of running
  inference locally, because local free RAM is tight (see below).

## Architecture / repo layout

Package: `driveway_guard`, under `src/driveway_guard/`. Full detailed
design (feature schema field lists, calibration JSON schema, exact rule
thresholds, build-order rationale) lives in the original approved plan at
`C:\Users\vhule\.claude\plans\frolicking-toasting-hearth.md` on this same
machine — read that for full depth, it was not repeated here. Summary of
the module layout:

```
src/driveway_guard/
├── run.py                    # CLI entry point (python -m driveway_guard.run)
├── config.py                 # RunConfig dataclass — video_path, out_dir, calib_path,
│                              #   detector_model, pose_model, pose_proximity_norm,
│                              #   weapon_model, conf, device, frame_stride, write_video, log_level
├── pipeline.py                # wires every stage together per frame
├── sources/                  # FrameSource ABC + VideoFileSource
├── detection/
│   ├── tracker.py            # YOLO11 + ByteTrack (via ultralytics .track())
│   ├── types.py               # TrackedObject, ObjectClass
│   └── weapon_detector.py     # WeaponDetector class — proximity-gated
│                              #   detect(), argmax over non-denylisted
│                              #   classes only (denylist default:
│                              #   person/hand/phone). Constructor:
│                              #   model_path, device="cpu", conf=0.4,
│                              #   proximity_norm=0.15, pad_ratio=0.4,
│                              #   non_threat_class_names=None.
│                              #   gated_person_ids() finds persons within
│                              #   proximity_norm * frame_diagonal of any vehicle centroid.
├── pose/estimator.py          # proximity-gated YOLO11-pose
├── calibration/
│   ├── schema.py              # pydantic: driveway polygon + egress corridor
│   └── geometry.py            # point-in-polygon, egress-corridor overlap math
├── features/
│   ├── schema.py               # FrameFeatureVector, VehicleConvergenceFeatureVector, BlockingObservation
│   ├── track_state.py          # TrackHistory (velocity/dwell), ProximityDwellTracker
│   ├── extractor.py            # FeatureExtractor: per-pair + blocking records
│   └── convergence.py          # compute_convergence: per-vehicle aggregate
├── scoring/
│   ├── base.py                 # RiskScorer ABC
│   ├── rules.py                # RuleThresholds + RuleBasedScorer (the v1 scorer)
│   │                            #   weapon_confidence_threshold=0.5, weapon_min_duration_s=0.5
│   └── events.py               # EventAggregator (duration-based debounce), FlaggedEvent
├── output/
│   ├── overlay.py               # draws boxes/track IDs/skeleton/risk banner
│   ├── video_writer.py
│   └── event_log.py             # events.json / events.csv writers
└── imaging.py                  # shared crop_with_padding() used by pose + weapon detector

scripts/
└── train_weapon_model.py       # standalone utility (NOT part of the driveway_guard
                                 #   package, not imported anywhere at pipeline runtime).
                                 #   Fine-tunes a YOLO checkpoint on any YOLO-format
                                 #   dataset. Full CLI/behavior documented in the
                                 #   "Weapon detection" section below.

calib/example_driveway.json     # placeholder calibration, 1920x1080 — must match your video's exact resolution or it errors loudly
tests/                          # 42 passing unit tests, no video/model needed (pure logic)
```

CLI: `python -m driveway_guard.run --video <path> --out <dir> [--calib <path>] [--weapon-model <path>] [--device cpu|cuda:0] ...`
Full flag list (from `run.py: parse_args`): `--video` (required), `--out`
(required), `--calib`, `--detector-model` (default `yolo11n.pt`),
`--pose-model` (default `yolo11n-pose.pt`), `--pose-proximity-norm`
(default 0.15), `--weapon-model` (default None — stage skipped if
omitted), `--conf` (default 0.35), `--device` (default `cpu`),
`--frame-stride` (default 1), `--no-video-output`, `--log-level` (default
`INFO`).
Outputs land in `--out`: `annotated.mp4`, `events.json`, `events.csv`, `run_meta.json`.
**Added this session**: `run.py` now also prints a plain-language summary at
the end of the run (`logger.info`, so visible in normal terminal/notebook
output, not just inside `events.json`) — every flagged event, one line
each, sorted by time: type, start/end timestamp, `peak_score`, and
`track_ids`. Before this, `peak_score` was already being computed and
written to `events.json`/`events.csv`, it just wasn't visible without
opening those files or picking it out of scattered per-frame `WARNING`
log lines.

## Current status

- All v1 milestones from the plan are built (scaffolding through polish).
- **42/42 local pytest tests passing** (pure-logic tests only — geometry,
  scorer, track state, feature extractor; no video/model dependency).
  Confirmed passing again this session (`.venv/Scripts/python.exe -m
  pytest -q` → `42 passed in 24.56s`) before committing.
- Pushed to GitHub: **https://github.com/SMOKE484/mashtronicsAI** (public
  repo). `main` is up to date through commit `7fa9c2f`. Full commit
  history as of end of session:
  - `b2f156b` — "Add v1 driveway anomaly detection pipeline" (prior session)
  - `30ff36a` — "Declare lap dependency explicitly; fix blocking-overlap
    test expectation" (this session; these were pending-uncommitted at
    session start, carried over from the prior session)
  - `7fa9c2f` — "Add weapon detection training script and sourcing docs"
    (this session)
  - No uncommitted changes as of the end of this session — `git status`
    was clean after the push.
- **First real end-to-end pipeline run happened on Kaggle GPU** (T4), using
  the user's own downloaded video (`Normal.mp4`, in Kaggle dataset
  `training1`, real mount path
  `/kaggle/input/datasets/vhulendamashamba/training1/Normal.mp4` — the
  Kaggle sidebar display path is misleading, don't trust it, always verify
  with `find /kaggle/input/ -iname "*.mp4"`). Result: ran cleanly through
  300+ frames, **0 events flagged**. Good news for false-positive control
  *if* detection was actually finding people/vehicles — **this has not yet
  been visually confirmed** (annotated.mp4 from that run hasn't been
  reviewed). Still a loose end, see "Next steps".

## Weapon detection (picked up this session — training done, wiring/spot-check not done)

This was the explicit thing the user asked to pick up this session
("pickup the weapon-detection"). Wiring (`--weapon-model` CLI flag,
`WeaponDetector` class, proximity gating) already existed from the v1
build. What was missing, and what this session worked on, was an actual
trained checkpoint — no `.pt` weapon checkpoint exists yet as of end of
session; training was mid-run when the session ended (see below).

### Sourcing — candidates rejected, with specifics

Vetted several ready-made checkpoints/datasets via WebSearch + WebFetch +
raw `curl` (Roboflow Universe pages are a JS SPA and return HTTP 403
without a browser-like User-Agent header — `curl -A "Mozilla/5.0 ..."`
was needed to fetch them at all). None were trustworthy enough to use
as-is:
- **`https://universe.roboflow.com/yolo-xkggu/guns-mms73`** ("Guns" model,
  9.3k images, CC BY 4.0) — noisy labels: the interactive class widget on
  the page showed classes `gun`, `rifle`, `0`, `d` — the last two are
  garbage/placeholder labels, a strong signal of sloppy or auto-merged
  annotation.
- **`https://github.com/JoaoAssalim/Weapons-and-Knives-Detector-with-YOLOv8`**
  — checked via GitHub's contents API
  (`api.github.com/repos/.../contents/models`). Only ships `.onnx` weights
  (`best.onnx`, `best-wave.onnx`, `db.onnx`, `haar.onnx`, `normal.onnx`,
  `symlet.onnx` — six variants named after wavelet-denoising techniques,
  unexplained which is "the" model to use), zero reported precision/
  recall/mAP metrics anywhere in the README, and a self-contradictory
  license (README literally says "distributed under the [MIT] license"
  while the repo's actual `LICENSE` file is GPL-3.0).
- **`https://universe.roboflow.com/weopon-detection/weapon-detection-using-yolov8`**
  — wrong domain entirely; the class widget showed `Tank`, `Knife`,
  `Handgun`, `Rifle`, `Missile` — a general "weapons of war" dataset, not
  street/CCTV-level footage. Only 671 images, CC BY 4.0.
- **`https://universe.roboflow.com/yolo-otbw9/weapon-detection-o4mdd`**
  ("Weapon Detection", 2.9k images, CC BY 4.0) — despite the generic
  project name, the actual classes are per-firearm-**brand**:
  `battleaxe`, `beretta`, `caracal`, `charterarmsbulldog`, `dagger`. No
  generic `pistol`/`gun`/`handgun` class exists at all — useless for
  "is a gun present," which is all this project needs.
- **`https://universe.roboflow.com/edi-detection/weapon-yolo8`** — 10k
  images, CC BY 4.0, classes `hand`, `Pistol`, `Gun`, `Knife`, `handgun`.
  Larger and reasonably on-domain, but `Pistol`/`Gun`/`handgun` as three
  separate near-duplicate classes suggests multiple source datasets merged
  without harmonizing class names. Kept as a mental fallback but not
  chosen (see below for why the CCTV set won instead).
- Roboflow's algorithmic **"Similar Projects" sidebar** (shown on the
  chosen dataset's own page): `weapons_detection` (by Altechproject, 583
  images), `Knife dataset` (by Home, 4.65k), `knives` (by research, 4.08k),
  `guns 2 2` (by 3laas Workspace, 3.25k), `Prevención de accidentes` (by
  Tesis2proyecto, 8.71k). None of these were vetted (no URLs were even
  obtained — this was purely a screenshot the user shared of the sidebar
  widget). Explicitly told the user not to use these: it's an algorithmic
  "related content" suggestion, not curated, and two of the five
  (`knives` and `Prevención de accidentes`) shared the *identical*
  thumbnail image — a close-up webcam-distance photo of a hand pointing an
  object at the camera, the wrong framing/distance entirely for a driveway
  CCTV angle (person-sized, distant, elevated).
- The University of Granada **Pistols Dataset**
  (`https://public.roboflow.com/object-detection/pistols`) was the one
  candidate that held up well on inspection — CC0/public domain
  (`https://creativecommons.org/publicdomain/zero/1.0/`), ~2973/2986
  images (page showed both numbers in different places — dataset overview
  said 2986, the "resize-416x416" version listing said 2973 — treat
  ~2970-2990 as the real count), single `pistol`/"Guns" annotation class,
  shared by an actual academic weapons-detection research group
  (`sci2s.ugr.es/weapons-detection`). This one was **not chosen as the
  primary pick** (see next section for why) but is documented as the
  fallback in README.md.

### Decision: train our own, don't trust a mystery checkpoint

None of the rejected candidates above had trustworthy license/metric/label
provenance for a "would raise risk score on a detection" use case, so
`scripts/train_weapon_model.py` was written to fine-tune a checkpoint from
scratch instead, matching the project's existing pattern of nano YOLO11
models trained/run rather than downloaded blind.

**Script details** (so this doesn't need re-reading next session):
- Standalone script, not part of the `driveway_guard` package, not
  imported at pipeline runtime.
- CLI args: `--data` (required, path to a YOLO-format `data.yaml`),
  `--out` (required, output directory), `--base-model` (default
  `yolo11n.pt`), `--epochs` (default 60), `--imgsz` (default 640),
  `--batch` (default -1, meaning Ultralytics' AutoBatch picks it based on
  free GPU memory), `--device` (default `cpu`), `--patience` (default 15,
  early-stopping epochs without improvement).
- Behavior: loads `--base-model`, calls `model.train(...)` with those
  args plus `project=<out>, name="train"` (so the actual run lands in
  `<out>/train/`), copies `<out>/train/weights/best.pt` to
  `<out>/weapon_model.pt`, then runs `model.val(...)` and writes
  `<out>/metrics.json` with this exact schema:
  ```json
  {
    "precision": <float, mean across classes>,
    "recall": <float, mean across classes>,
    "mAP50": <float, mean across classes>,
    "mAP50-95": <float, mean across classes>,
    "per_class": {
      "<class name>": {"precision": <float>, "recall": <float>, "mAP50": <float>, "mAP50-95": <float>},
      ...
    },
    "epochs_trained": <int>,
    "base_model": "<str>",
    "data": "<str, path to data.yaml used>"
  }
  ```
- The per-class breakdown (`Metric.class_result(i)` returning
  `(p[i], r[i], ap50[i], ap[i])`) was specifically verified against the
  installed local `ultralytics==8.4.128` API before writing it, since
  getting that wrong would fail silently-ish on Kaggle mid-run. (Kaggle
  itself is running `ultralytics==8.4.129` per the training log — one
  patch version ahead of local; API used is stable across that gap, no
  issue observed.)

**Dataset picked**: [weapon detection cctv v3 dataset](https://universe.roboflow.com/weapon-detection-cctv/weapon-detection-cctv-v3-dataset)
on Roboflow Universe — workspace slug `weapon-detection-cctv`, project
slug `weapon-detection-cctv-v3-dataset`, CC BY 4.0 (**attribution required**
if this ever ships publicly — unlike the CC0 Pistols Dataset fallback,
which needs no attribution). The project page's interactive "Detecting
classes" widget showed `hand`, `phone`, `pistol`, `gun`, `Knife` (and the
page's own marketing description claimed "eleven distinct classes total,
including knives, pistols, rifles, and other hand-held objects" — this
claim did **not** match what was actually downloaded, see next section).
Picked over the single-class Pistols Dataset specifically because of the
`hand`/`phone` classes: training the model to actively distinguish
"phone in hand" from "gun" directly hedges the exact false-positive
scenario this project's own design notes already called out (small/dark
handheld objects near a car window being mistaken for a weapon). 2.6k
images at the time of inspection (per the page's "See all 2.6k images"
link — see below for why the actual downloaded count differs).

**Surprise found mid-training**: despite the above, the actual exported
dataset version (project version `1`, downloaded via the Roboflow Python
package, export format string `"yolov11"`) only has **`nc=2`** in its
generated `data.yaml` — Ultralytics logged
`Overriding model.yaml nc=80 with nc=2` at model-build time. One
weight-remapping log line — `Remapped 1/2 cls head rows from pretrained
weights by class name` — is a real clue: Ultralytics only does that
name-based remap when one of the dataset's class *names* exactly matches
one of COCO's 80 pretrained class names, and COCO does not contain
`gun`/`pistol`/`knife`/`phone`/`hand` but **does** contain `person`
(COCO class index 0). So there's a good chance one of the 2 classes is
literally named `person`, meaning the other is presumably the actual
weapon/threat class — but **this was asked for and not yet confirmed** by
end of session (`cat data.yaml` output was requested from the user, not
yet returned). Do not assume this is correct without checking — it's an
inference from one log line, not a confirmed fact.

Training set stats actually seen in the Kaggle log: 3620 training images
scanned, 540 validation images, 9253 total box labels in train after
filtering; ~20 images were dropped with `ignoring corrupt image/label:
labels mix segment and detection rows` (a handful of images in this
dataset were annotated as filled polygons instead of bounding boxes;
Ultralytics drops those specific labels automatically and proceeds with
box-only ones — minor data loss, not a functional blocker, no action
needed). The `nc=2` surprise and the dropped images required no pipeline
code changes either way — but **the 2 classes turning out to include
`person`** did require one: `WeaponDetector` assumed class-agnostic argmax
was safe, and it wasn't once one of the classes was the same object the
crop is centered on. See the "Weapon detection is class-agnostic..." design
decision bullet above (updated this session) for the bug and the fix
actually applied.

### State at end of session — training finished, checkpoint not yet wired in or spot-checked

Sequence of commands actually run on Kaggle this session (T4 GPU, internet
on), including two errors hit and fixed along the way:

1. `!pip install roboflow` then the Roboflow-generated download snippet
   (workspace `weapon-detection-cctv`, project
   `weapon-detection-cctv-v3-dataset`, version `1`, format `yolov11`) —
   succeeded, dataset landed in
   `/kaggle/working/weapon-detection-cctv-v3-dataset-1/`.
2. `!git clone https://github.com/SMOKE484/mashtronicsAI.git && cd
   mashtronicsAI && ...` — **failed**: `fatal: destination path
   'mashtronicsAI' already exists and is not an empty directory` (a stale
   clone from an earlier Kaggle session persisted in `/kaggle/working`,
   which Kaggle can carry over across notebook edits within the same
   session/disk). Following command (`python3
   /kaggle/working/scripts/train_weapon_model.py`) then **also failed**:
   `No such file or directory`, because it ran from `/kaggle/working`
   directly instead of inside the (uncloned, stale) `mashtronicsAI/` dir.
3. **Fix applied**: `%cd /kaggle/working/mashtronicsAI` then `!git pull`
   (this pulled the `7fa9c2f` commit — the stale clone predated the
   training script's existence, which is *why* it wasn't found) then
   `!pip install -q -e .` then the training command below. This worked —
   training started successfully.
4. Training command actually run:
   ```
   !python scripts/train_weapon_model.py \
     --data /kaggle/working/weapon-detection-cctv-v3-dataset-1/data.yaml \
     --out /kaggle/working/weapon_model \
     --device cuda:0
   ```
   (uses script defaults: `--epochs 60`, `--imgsz 640`, `--batch -1`
   auto, `--patience 15`, `--base-model yolo11n.pt`.)

Environment confirmed by the training log: `Ultralytics 8.4.129`,
`Python-3.12.13`, `torch-2.10.0+cu128`, `Tesla T4 (14912MiB)`. AutoBatch
picked batch size 25 (using ~61% / 8.87GB of the T4's 14.56GB). Optimizer
auto-selected as `AdamW(lr=0.001667, momentum=0.9)`.

**Update: this run finished cleanly** (confirmed via a pasted training log,
not yet re-verified by re-running anything). 60/60 epochs completed in
0.609 hours, `patience=15` early-stopping never triggered. Final combined
validation (`weapon_model/train/weights/best.pt`, re-confirmed by the
script's separate `model.val()` call that writes `metrics.json`):

| | precision | recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| **all** | 0.862 | 0.722 | 0.802 | 0.524 |
| person | 0.891 | 0.793 | 0.859 | 0.599 |
| weapon | 0.833 | 0.652 | 0.744 | 0.449 |

**The `nc=2` class-name question is resolved**: the val output rows are
literally labeled `person` and `weapon` — the COCO-remap speculation above
was correct. This is the coarse-but-useful outcome the decision tree below
was hoping for, not the unhelpful-classes case.

Reading these numbers: person detection is solid (mAP50 0.86). Weapon
detection is usable but not strong — precision 0.83 means most positive
calls are real, but recall 0.65 means roughly a third of true weapons in
the validation set are missed, and mAP50-95 0.449 (vs person's 0.599)
shows looser box localization on weapons specifically (small/occluded
objects, consistent with the class being harder). This clears the "keep,
don't fall back to Pistols Dataset" bar from the decision tree below, but
it's a first checkpoint, not a finished one — false negatives (missed
weapons) are the likely failure mode to watch for during spot-checking,
more so than false positives.

Weights confirmed written to `/kaggle/working/weapon_model/weapon_model.pt`
(per the script's own log line). **Not yet downloaded locally, not yet
wired into an actual `--weapon-model` pipeline run, not yet spot-checked
against real footage** — those are still the open items.

**Immediate next steps for the next session** (in order):
1. ~~Check whether the Kaggle run finished~~ — done, see above, it finished.
2. ~~Get `data.yaml` class names~~ — done, confirmed `person`/`weapon` from
   the val output directly, no need to separately `cat data.yaml`.
3. ~~Get `metrics.json`~~ — done, see table above (the training log's own
   `INFO: Metrics: {...}` dump matches the schema documented earlier in
   this section, so treat that log as authoritative — no need to
   re-fetch the file unless double-checking).
4. Decision made: numbers are reasonable enough to proceed with this
   checkpoint (not fall back to the Pistols Dataset) — proceed to step 5.
5. Download `/kaggle/working/weapon_model/weapon_model.pt` from Kaggle (or
   keep using it directly on Kaggle, given local is CPU-only — Kaggle is
   already the validated workflow for running the pipeline on video, see
   "Kaggle workflow" below).
6. Wire it into an actual end-to-end pipeline run via
   `--weapon-model weapon_model.pt` against real driveway/CCTV clips and
   spot-check for false negatives *and* false positives (phone in hand,
   dark clothing folds — the exact risk this project's design notes
   flagged from the start). This is still the highest-uncertainty part of
   the whole system — don't call it "working" off training metrics alone,
   actually look at the annotated output.
7. Unrelated older Kaggle friction point (**unresolved**): user couldn't
   find a "New Version"/upload option to add more clips to the existing
   `training1` dataset (used for the earlier pipeline smoke-test video,
   not the weapon dataset). Check whether they're on the dataset's own
   page (not the notebook's read-only Input side-panel) and whether their
   account has phone verification completed, if it's still blocking them.

## Standing instructions / preferences (carry these forward)

- **Never add Claude as a co-author/contributor on commits in this repo.**
  The user explicitly said not to early on, so all commits so far omit the
  usual `Co-Authored-By: Claude` trailer. Keep doing that unless told
  otherwise.
- Git identity is configured locally: name `smoke`, email
  `vhulendamashamba4@gmail.com`.
- User prefers testing on **Kaggle (free GPU)** over local CPU runs, due to
  tight local RAM (see Environment below). Default to that workflow for
  anything involving actually running the pipeline (or training) on video.
- When sourcing a pretrained checkpoint or dataset from GitHub/Roboflow,
  **actually vet it** — real class list (fetched from the page, not
  trusted from a search-result summary or the project's own marketing
  description), real image count, license, reported metrics if any. This
  session found multiple plausible-looking candidates (by name/search
  ranking) that turned out wrong-domain or badly labeled once actually
  inspected (see "Sourcing — candidates rejected" above for the full
  list and specifics). Roboflow Universe pages require a browser-like
  User-Agent to fetch via curl/WebFetch (plain requests get HTTP 403).
  Even after picking a well-vetted dataset, the *actual exported* content
  can still differ from what the project page advertises (see the `nc=2`
  surprise above) — so the real verification point is always the
  downloaded `data.yaml` / trained `metrics.json`, not the source page.

## Environment notes

- Machine: Windows 11, repo at
  `c:\Users\vhule\OneDrive\Desktop\Projects\mashtronicsAI`.
- Local venv at `.venv/` (`.venv\Scripts\python.exe`), all deps from
  `requirements.txt` installed (including `lap`, now properly declared in
  both `requirements.txt` and `pyproject.toml` as of commit `30ff36a` —
  install it into the venv too if it's a fresh clone). Local
  `ultralytics` version confirmed this session: `8.4.128`.
- **No NVIDIA GPU locally** — CPU-only inference. Nano YOLO models chosen
  because of this.
- **RAM was tight in the first session**: 15.63 GB total, only ~3.67 GB
  free observed at one point. Worth rechecking if local runs feel
  sluggish; `--frame-stride` exists as an escape hatch if needed.
- Disk space is not a concern (397 GB free of 475 GB at last check).

## Kaggle workflow (validated as working across two sessions now)

1. New notebook → Accelerator: GPU T4 x2 → Internet: On.
2. If `/kaggle/working/mashtronicsAI` doesn't already exist:
   `!git clone https://github.com/SMOKE484/mashtronicsAI.git && cd mashtronicsAI && pip install -q -e .`
   If it **does** already exist from a prior run in the same session/disk
   (Kaggle can persist `/kaggle/working` across notebook edits — confirmed
   this session, this bit the user), `%cd /kaggle/working/mashtronicsAI`
   and `!git pull` instead of re-cloning — re-cloning into a non-empty
   directory fails with `fatal: destination path ... already exists`.
   Always `git pull` even if it does exist, in case the clone predates a
   later push (also happened this session — the stale clone didn't have
   `scripts/train_weapon_model.py` yet until pulled).
3. For pipeline runs on video: attach video via "Add Input" → Dataset. Get
   the real mount path with `find /kaggle/input/ -iname "*.mp4"` — don't
   trust the sidebar's displayed path, it's been wrong before (missing a
   `datasets/<username>/` path segment).
4. For weapon-model training: see README.md's "Weapon detection model"
   section for the up-to-date Roboflow dataset-export snippet and training
   command — don't duplicate it here beyond what's in the "Weapon
   detection" section above, keep README as the source of truth since
   it's more likely to be kept in sync with the actual script if the
   dataset choice changes again.
5. Known Kaggle friction point (**unresolved**): see "Standing
   instructions"/"Weapon detection next steps" item 7 above re: adding a
   new dataset version to `training1`.

## Next steps (in priority order)

1. **See "Weapon detection" section above** — this is the active thread.
   Training finished with usable numbers (mAP50 0.744 on the `weapon`
   class, classes confirmed `person`/`weapon`) and the keep-vs-fallback
   decision is made (keep). What's left: download `weapon_model.pt`,
   wire it in via `--weapon-model`, and spot-check on real footage.
2. Once a weapon checkpoint's quality is confirmed and it's wired in,
   resolve the older loose end: visually confirm the first Kaggle
   pipeline run's `annotated.mp4` actually shows correct person/vehicle
   detection (0 events flagged is only good news if detection was working
   at all — never confirmed).
3. Once weapon detection and basic detection/tracking are both confirmed
   working, the previously-agreed testing roadmap continues: a person
   actively approaching/interacting with a vehicle → a busier benign clip
   (multiple people, false-positive check) → a two-vehicle boxing-in clip
   (needs a **new calibration JSON** matching that clip's exact resolution
   + a `resident_vehicle_hint` zone — boxing-in is silently skipped without
   calibration) → real struggle/aggressive-approach footage for tuning
   `RuleThresholds` in `scoring/rules.py`.
4. AI-generated (Gemini/Veo) prompts for synthetic test clips were drafted
   in the first session for each event type as a fallback/smoke-test
   option, but the user has been sourcing real videos instead, which is
   preferable — real footage should stay the priority for anything used to
   tune thresholds.
