# Handover: driveway-guard (mashtronicsAI)

Context for picking up this project in a new chat session, without the user
re-explaining. Written 2026-08-26 at the end of the session that built v1.

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
  model — it's a hand-written **rule-based risk scorer** operating on
  detection/tracking/pose features. Crucially, the feature schema
  (`FrameFeatureVector`) is designed to double as a future training row, so
  this isn't throwaway work once real labeled data exists.
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
- **Event debouncing is duration-based (elapsed seconds), not frame-count
  based.** The plan originally sketched `event_min_consecutive_frames`, but
  frame count isn't robust to varying fps or `--frame-stride`, so
  `EventAggregator` in `scoring/events.py` uses `min_duration_s` instead
  (`RuleThresholds.event_min_duration_s` / `weapon_min_duration_s`). This is
  a deliberate implementation improvement over the literal plan text.
- **Weapon detection is explicitly the highest-uncertainty component.** No
  model is bundled; the stage is disabled unless `--weapon-model` is passed.
  This is where the next chat session picks up (see "Next steps" below).
- **CPU-only locally** (see "Environment" below) — nano YOLO models
  (`yolo11n.pt`, `yolo11n-pose.pt`) were chosen specifically for this. The
  user tests on **Kaggle notebooks** (free T4 GPU) instead of running
  inference locally, because local free RAM is tight (see below).

## Architecture / repo layout

Package: `driveway_guard`, under `src/driveway_guard/`. Full detailed
design (feature schema field lists, calibration JSON schema, exact rule
thresholds, build-order rationale) lives in the original approved plan at
`C:\Users\vhule\.claude\plans\frolicking-toasting-hearth.md` on this same
machine — read that for full depth. Summary of the module layout:

```
src/driveway_guard/
├── run.py                    # CLI entry point (python -m driveway_guard.run)
├── config.py                 # RunConfig
├── pipeline.py                # wires every stage together per frame
├── sources/                  # FrameSource ABC + VideoFileSource
├── detection/
│   ├── tracker.py            # YOLO11 + ByteTrack (via ultralytics .track())
│   ├── types.py               # TrackedObject, ObjectClass
│   └── weapon_detector.py     # proximity-gated firearm detection — NEEDS A MODEL
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
│   └── events.py               # EventAggregator (duration-based debounce), FlaggedEvent
├── output/
│   ├── overlay.py               # draws boxes/track IDs/skeleton/risk banner
│   ├── video_writer.py
│   └── event_log.py             # events.json / events.csv writers
└── imaging.py                  # shared crop_with_padding() used by pose + weapon detector

calib/example_driveway.json     # placeholder calibration, 1920x1080 — must match your video's exact resolution or it errors loudly
tests/                          # 42 passing unit tests, no video/model needed (pure logic)
```

CLI: `python -m driveway_guard.run --video <path> --out <dir> [--calib <path>] [--weapon-model <path>] [--device cpu|cuda:0] ...`
Outputs land in `--out`: `annotated.mp4`, `events.json`, `events.csv`, `run_meta.json`.

## Current status

- All v1 milestones from the plan are built (scaffolding through polish).
- **42/42 local pytest tests passing** (pure-logic tests only — geometry,
  scorer, track state, feature extractor; no video/model dependency).
- Pushed to GitHub: **https://github.com/SMOKE484/mashtronicsAI** (public
  repo, pre-existing remote the user had already created). One commit on
  `main` so far (`b2f156b`).
- **First real end-to-end run happened on Kaggle GPU** (T4), using the
  user's own downloaded video (`Normal.mp4`, in Kaggle dataset
  `training1`, real mount path
  `/kaggle/input/datasets/vhulendamashamba/training1/Normal.mp4` — the
  Kaggle sidebar display path is misleading, don't trust it, always verify
  with `find /kaggle/input/ -iname "*.mp4"`). Result: ran cleanly through
  300+ frames, **0 events flagged**. Good news for false-positive control
  *if* detection was actually finding people/vehicles — **this has not yet
  been visually confirmed** (annotated.mp4 from that run hasn't been
  reviewed). That's a loose end, see below.

### Uncommitted local changes (not yet pushed)

`git status` currently shows three modified files, made after the initial
push, not yet committed:
- `requirements.txt`, `pyproject.toml` — added `lap>=0.5.12` as an explicit
  dependency (ByteTrack needs it; Ultralytics auto-installed it mid-run on
  Kaggle with a warning, so it's now declared properly instead of relying
  on auto-update).
- `tests/test_calibration_geometry.py` — fixed a wrong test expectation
  (`test_blocking_overlap_full_when_vehicle_centered_in_corridor` asserted
  `1.0` for a bbox narrower than the corridor, which can only ever achieve
  `0.5` coverage even when centered — the implementation was correct, the
  test wasn't; replaced with two tests covering both the full-coverage and
  partial-coverage cases).

The user was asked whether to commit+push these and the conversation moved
on without an explicit answer — **check `git status` at the start of the
next session and confirm with the user before committing.**

## Standing instructions / preferences (carry these forward)

- **Never add Claude as a co-author/contributor on commits in this repo.**
  The user explicitly said not to, so all commits so far omit the usual
  `Co-Authored-By: Claude` trailer. Keep doing that unless told otherwise.
- Git identity is configured locally: name `smoke`, email
  `vhulendamashamba4@gmail.com`.
- User prefers testing on **Kaggle (free GPU)** over local CPU runs, due to
  tight local RAM (see Environment below). Default to that workflow for
  anything involving actually running the pipeline on video.

## Environment notes

- Machine: Windows 11, repo at
  `c:\Users\vhule\OneDrive\Desktop\Projects\mashtronicsAI`.
- **Python was not installed at session start** — only the Windows Store
  execution-alias stub existed. Installed CPython 3.11.9 via
  `winget install --id Python.Python.3.11`. Local venv at `.venv/`
  (`.venv\Scripts\python.exe`), all deps from `requirements.txt` installed
  (including the `lap` fix above — install that into the venv too if it's
  a fresh clone).
- **No NVIDIA GPU locally** — CPU-only inference. Nano YOLO models chosen
  because of this.
- **RAM was tight during this session**: 15.63 GB total, only ~3.67 GB free
  observed at one point. Worth rechecking if local runs feel sluggish;
  `--frame-stride` exists as an escape hatch if needed.
- Disk space is not a concern (397 GB free of 475 GB at last check).

## Kaggle workflow (already validated as working)

1. New notebook → Accelerator: GPU T4 x2 → Internet: On.
2. `!git clone https://github.com/SMOKE484/mashtronicsAI.git && cd mashtronicsAI && pip install -q -e .`
3. Attach video via "Add Input" → Dataset. **Get the real mount path with
   `find /kaggle/input/ -iname "*.mp4"` — don't trust the sidebar's
   displayed path**, it's been wrong once already (missing a
   `datasets/<username>/` path segment).
4. `!python -m driveway_guard.run --video <real_path> --out /kaggle/working/out --device cuda:0`
5. Known Kaggle friction point (**unresolved**): user couldn't find a "New
   Version" / upload option to add more clips to the existing `training1`
   dataset. Last guidance given: check whether they're on the dataset's own
   page (not the notebook's read-only Input side-panel) and whether their
   account has phone verification completed. No confirmation received yet
   on which it was — pick this up if it's still blocking them.

## Next steps (in priority order)

1. **This is what the user explicitly asked to pick up in the next
   session**: get weapon/gun recognition actually working.
   - Source a YOLO checkpoint fine-tuned for firearms — Roboflow Universe
     was suggested (search "pistol detection" / "gun detection YOLO"),
     quality varies and needs vetting.
   - Wire it in via `--weapon-model <path>`.
   - Test on Kaggle GPU.
   - **Treat this as the highest-uncertainty, most false-positive-prone
     part of the whole system** (small/dark-object detection on CCTV
     footage) — needs its own dedicated precision evaluation before
     trusting it, per the design notes above and in the plan file. Don't
     skip straight to declaring it "working" off a couple of test clips.
2. Resolve the two loose ends above: (a) decide whether to commit the
   pending local fixes, (b) confirm whether `annotated.mp4` from the first
   Kaggle run actually shows correct detection/tracking, (c) unblock the
   Kaggle "add new dataset version" issue if still stuck.
3. Once confirmed working on more clips, the previously-agreed testing
   roadmap continues: a person actively approaching/interacting with a
   vehicle → a busier benign clip (multiple people, false-positive check)
   → a two-vehicle boxing-in clip (needs a **new calibration JSON**
   matching that clip's exact resolution + a `resident_vehicle_hint` zone —
   boxing-in is silently skipped without calibration) → real
   struggle/aggressive-approach footage for tuning `RuleThresholds` in
   `scoring/rules.py`.
4. AI-generated (Gemini/Veo) prompts for synthetic test clips were drafted
   earlier in the conversation for each event type as a fallback/smoke-test
   option, but the user has been sourcing real videos instead, which is
   preferable — real footage should stay the priority for anything used to
   tune thresholds.
