# Handover: driveway-guard (mashtronicsAI)

Written 2026-08-26, replacing the previous handover after a full incremental
rewrite. The old handover covered four sessions of building and testing a
five-event rule-based pipeline (struggle, boxing-in, weapon-at-window,
sprint-approach, multi-directional convergence) against real Kaggle footage.
That testing surfaced real bugs and real fixes, but also showed the number of
coupled moving parts (pose estimation, calibration-dependent convergence
math, five separate debounce states) made it hard to trust any single result
— most of a session went into disentangling which of several systems was
responsible for a given number.

**Decision made this session: stop tuning the tangle, rebuild deliberately.**
Prove one event type works end-to-end and is properly tuned on real footage,
*then* add the next, one at a time, each validated before the next starts.
Starting with weapon detection, since it already has the most validated
groundwork (a trained checkpoint, confirmed genuine detections on real
footage from the previous effort). This document reflects **only the current
state after that rewrite** — the old session-by-session narrative is not
reproduced here, but nothing is lost: everything retired is still committed
on `main` through commit `d647e3f`, pushed to GitHub, recoverable at any time
via `git log` / `git show d647e3f:<path>`.

**tl;dr of where things stand right now**: the code rewrite (Part 2, below)
is done locally and passing 9/9 tests, but **not yet committed** — waiting on
the user before committing. The dataset-expansion-and-retrain work (Part 1,
below) is designed and the merge script is written, but **has not been run
yet** — no new checkpoint exists. **Nothing has been validated against real
footage since the rewrite.** The single most important next action is
running Part 1 and Part 2's validation step on Kaggle, starting completely
fresh there too (new notebook, fresh clone, fresh dataset downloads — not
continuing from any prior Kaggle session's state).

## What this project is

A production-track computer vision system that watches a **fixed driveway
security camera** (target hardware: Dahua CCTV, RTSP-capable — not yet wired
up) and flags candidate carjacking/hijacking events for human review.

Five event types were designed collaboratively with the user (full rationale
in `.claude/plans` if needed):
1. **Struggle / aggressive contact** near a vehicle (hijacking-style)
2. **Boxing-in** — a second vehicle blocking the resident's car's exit path
3. **Weapon at window** — a gun visible at a vehicle window — **the only one
   currently implemented, this phase's focus**
4. **Sprint approach** — someone closing on a stationary vehicle at running
   speed
5. **Multi-directional convergence** — 2+ people approaching a vehicle from
   different angles simultaneously

Types 1, 2, 4, 5 are designed and were previously implemented and tested, but
their code is currently retired (see "What's retired" below) pending this
incremental rebuild. They come back one at a time, using the retired code in
git history as a starting point rather than being redesigned from zero.

## Part 1: expand the weapon dataset and retrain — designed, not yet run

The existing `weapon_model.pt` (if you still have it downloaded locally —
otherwise it needs retraining regardless, see "Kaggle workflow" below on why
`/kaggle/working` doesn't persist across sessions) was trained on Roboflow's
`weapon-detection-cctv-v3-dataset` (`nc=2`: `person`/`weapon`). Validation
metrics: precision 0.833 / recall 0.652 / mAP50 0.744 on the `weapon` class
specifically (person: 0.891/0.793/0.859) — solid precision, weak recall
(roughly a third of true weapons missed in validation).

**Plan: add a second dataset and retrain**, rather than replacing the first
(keeping the first dataset's `person` class matters — see "Weapon detection"
in README.md for why). Dataset research this session, vetted by actually
fetching each Roboflow Universe page (they return HTTP 403 without a
browser-like User-Agent — `curl -A "Mozilla/5.0 ..."`) and spot-checking real
sample thumbnails, not just page descriptions or search-result summaries:

| Dataset | Images | Class | License | Verdict |
|---|---|---|---|---|
| `dietest/gun-cctv-detection` | 5,149 | `Guns` (single, clean) | CC BY 4.0 | **Picked.** 4/4 sampled thumbnails genuine CCTV-style footage (elevated angle, timestamp/camera-ID overlay). |
| `gun-detection-1lttj/gun-detection-1fbbu` | 9,256 | `gun` | CC BY 4.0 | Larger but mixed quality — 1 of 2 samples was a watermarked Alamy stock photo. Documented fallback/top-up, not first pick. |
| `mahad-ahmed/gun-detection-uemtc` | 6,824 | `gun` | CC BY 4.0 | Rejected — sampled thumbnail was a toy water gun on a Christmas tree. |
| `augustus/guns_dataset_kaggle_cctv` | 3,356 | `guns` | CC BY 4.0 | Rejected — sampled thumbnail was a stock "pistol vs. revolver" product photo, not CCTV footage, despite the "kaggle_cctv" name. |
| `simuletic/cctv-weapon-detection-dataset-vcloz` | 141 | `person-weapon` | — | Rejected — too small, and the merged class name is the same red flag pattern as previously-rejected noisy-label datasets. |

**New script**: `scripts/merge_yolo_datasets.py` (written, not yet run).
Combines two-or-more Roboflow YOLO exports into one, remapping each source's
own class names onto a unified target list via a JSON config (full shape in
the script's own docstring). Output is a normal-shaped YOLO export
(`train/valid[/test]` images+labels, `data.yaml`), so
`scripts/train_weapon_model.py` consumes it with **zero changes** via
`--data <merged>/data.yaml`.

**Not yet done, in order**:
1. Download both datasets fresh on Kaggle (a **fresh notebook** — see
   "Kaggle workflow" below, don't assume anything from a prior Kaggle session
   is still there).
2. Run `scripts/merge_yolo_datasets.py` to combine them.
3. Run `scripts/train_weapon_model.py --data <merged>/data.yaml --out ...
   --device cuda:0` (same script, same flags as before — see README.md
   "Weapon detection model" for the exact command shape).
4. Compare the new `metrics.json` against the baseline above — only adopt
   the new checkpoint if it's actually better, don't swap blind.
5. **Save `weapon_model.pt` as a proper Kaggle Dataset immediately** —
   `/kaggle/working` does not survive across separate Kaggle sessions, only
   within one continuous session/disk. This has already cost one checkpoint
   once before.
6. If the Roboflow API key you use was the one exposed in a screenshot in an
   earlier session, rotate it first (still unconfirmed whether that was ever
   done).

## Part 2: rebuild the pipeline code — done locally, not committed, not validated

### What's reused as-is (unchanged from before)

- `src/driveway_guard/detection/tracker.py` — `Tracker` (YOLO + ByteTrack).
- `src/driveway_guard/detection/types.py` — `TrackedObject`, `ObjectClass`.
- `src/driveway_guard/detection/weapon_detector.py` — `WeaponDetector`,
  including the non-threat-class denylist fix
  (`_NON_THREAT_CLASS_NAMES = {"person", "hand", "phone"}`).
- `src/driveway_guard/imaging.py` — `crop_with_padding`.
- `src/driveway_guard/sources/` — `FrameSource`, `VideoFileSource`.
- `src/driveway_guard/scoring/events.py` — `EventAggregator`, `FlaggedEvent`
  (generic over `(event_type, key_tuple)`, reused directly by the new
  weapon scorer).
- `src/driveway_guard/output/event_log.py`, `output/video_writer.py`.
- `src/driveway_guard/output/overlay.py` — kept `draw_tracks`/
  `draw_text_banner`; removed `draw_skeleton` (no pose stage right now).

### What's retired (git rm'd locally, not yet committed; fully recoverable via `git show d647e3f:<path>`)

- `src/driveway_guard/pose/` (struggle-only)
- `src/driveway_guard/features/` — `schema.py`, `extractor.py`,
  `convergence.py`, `track_state.py`
- `src/driveway_guard/calibration/` — `schema.py`, `geometry.py`
- `src/driveway_guard/scoring/rules.py` — old `RuleBasedScorer` +
  `score_struggle`/`score_boxing_in`/`score_sprint`/`score_convergence` +
  `RuleThresholds`
- `src/driveway_guard/scoring/base.py` — old `RiskScorer` ABC (tied to the
  5-event feature schema; not reintroduced until there's a second scorer
  implementation to actually abstract over)
- `calib/example_driveway.json`
- `tests/test_rule_scorer.py`, `tests/test_feature_extractor.py`,
  `tests/test_track_state.py`, `tests/test_calibration_geometry.py`

`pydantic` dropped from `requirements.txt`/`pyproject.toml` (only
`calibration/schema.py` used it). `pyyaml` added (used by the new merge
script).

`scripts/diagnose_pipeline.py` and `scripts/inspect_frame.py` are left in
place but **currently broken** — they import the retired modules
(`FeatureExtractor`, `PoseEstimator`, `compute_convergence`,
`CalibrationConfig`) and will fail at import time until those come back.
`scripts/inspect_weapon_hits.py`, `scripts/export_weapon_snapshots.py`, and
`scripts/train_weapon_model.py` are untouched and still work.

### New: weapon-only pipeline

**`src/driveway_guard/scoring/weapon.py`** (new):
- `WeaponThresholds`: `weapon_confidence_threshold=0.5`,
  `weapon_min_duration_s=0.5`, `event_cooldown_s=5.0` — same numeric
  defaults as the old code's *intent*, but see the threshold-behavior note
  below, it's not a pure port.
- `score_weapon_hit(confidence, threshold)` — `0.0` below threshold or
  `None`, else passthrough confidence.
- `WeaponScorer` — wraps one `EventAggregator`. **Keyed by vehicle track ID
  alone, not `(person, vehicle)`.** This was an open decision in the old
  handover, now implemented: real footage (`video3.mp4`, previous session)
  showed weapon hits landing on 5 different person track IDs within a ~3.5s
  span as the tracker lost and reacquired the same person — which would
  reset a `(person, vehicle)`-keyed debounce clock on every switch. Keying
  by vehicle alone survives that churn; contributing person IDs are still
  recorded in the emitted event's `track_ids`, just not part of the key.
  Regression-tested in `tests/test_weapon_scorer.py` by simulating hits on
  the same vehicle landing on different person IDs across frames.

**Behavior change worth knowing about**: the old `RuleBasedScorer` had a
quirk — `score_weapon()` zeroed anything below `weapon_confidence_threshold`
(0.5), but the *actual* `EventAggregator` gate that had to be cleared for an
event to ever start accumulating duration was `risk_score_flag_threshold`
(0.65), because that one shared field was reused across all 5 event types.
So real events effectively needed confidence ≥0.65, not ≥0.5, even though
0.5 was the "documented" weapon threshold. The new `WeaponScorer` collapses
this to one clear threshold (0.5 default, fully configurable via
`--weapon-confidence-threshold`). This is a deliberate simplification, not
an accidental regression — but it does mean the new pipeline will pass more
lower-confidence hits through to the duration debounce than the old one did.
Worth watching for in the real-footage validation below.

**`pipeline.py`, `config.py`, `run.py`** — rewritten, same filenames. CLI
shape: `python -m driveway_guard.run --video <path> --out <dir>
--weapon-model <path> [--detector-model yolo11n.pt] [--conf 0.35]
[--device cpu] [--frame-stride 1] [--no-video-output] [--log-level INFO]
[--weapon-conf 0.4] [--weapon-proximity-norm 0.15] [--weapon-pad-ratio 0.4]
[--weapon-confidence-threshold 0.5] [--weapon-min-duration-s 0.5]
[--event-cooldown-s 5.0]`. `--weapon-model` is now **required** — no more
"skip the stage if omitted," since weapon detection is the entire point of
this phase. No `--calib`, no `--pose-model` (those flags are gone, not just
defaulted).

### New tests

- `tests/test_weapon_detector.py` — `WeaponDetector.detect()` had **no** unit
  test before (only real-footage validation). Now covers the non-threat-class
  denylist argmax fix directly (mocks the YOLO model via
  `unittest.mock.patch("driveway_guard.detection.weapon_detector.YOLO")`,
  since real inference isn't needed to test the class-filtering logic), plus
  a proximity-gating test (far person never even reaches the model).
- `tests/test_weapon_scorer.py` — `score_weapon_hit` pure-function cases,
  a duration-debounce integration test, a "never clears confidence
  threshold" negative test, and the vehicle-only-keying regression test
  described above.

**9/9 tests passing** (`.venv/Scripts/python.exe -m pytest -q`).

## Validation against real footage — not yet done, highest-priority next step

Using the Kaggle workflow below, **starting completely fresh** (new
notebook — the user was explicit this session that even Kaggle should start
from the bottom, not continue from any prior session's leftover state):

1. Get a `weapon_model.pt` — either finish Part 1 first (recommended, since
   it should be a strictly better checkpoint), or use whatever checkpoint is
   available locally/on Kaggle already if Part 1 hasn't been run yet. Parts 1
   and 2 have no dependency on each other.
2. Re-run the new `run.py` against `video1.mp4` and `video3.mp4` (the
   `hijackings` dataset — real mount path via
   `find /kaggle/input/ -iname "*.mp4"`, don't trust the sidebar's displayed
   path) with `--weapon-model weapon_model.pt` at current defaults. Confirm
   `weapon_at_window` fires on both — this would be the first real
   end-to-end confirmation of a correctly-firing event on real footage for
   the whole project.
3. If it doesn't fire on `video1.mp4` (whose cleanest previous run was a
   genuine 0.32s continuous detection, under the current 0.5s
   `weapon_min_duration_s`), lower `--weapon-min-duration-s` (e.g. toward
   0.3s) and re-run rather than guessing blind.
4. Re-run against `Normal.mp4` (benign clip, `training1` dataset) and
   confirm zero false-positive `weapon_at_window` events at whatever
   settings get chosen in step 3.
5. Update this handover with the outcome once confirmed working, so the next
   phase (adding the next event type — likely struggle, pulling its retired
   code back from `git show d647e3f:src/driveway_guard/...`) starts with an
   accurate record.

## Explicitly out of scope right now

Struggle, boxing-in, sprint-approach, multi-directional convergence, the
cross-event correlator design (still agreed with the user, never built —
anchor on resident vehicle track ID, co-occurrence within ~45-60s, not a
stage-ordered state machine), and the still-open base-detector
confidence/model-size tuning thread (`yolo11s.pt` + lower confidence — found
to genuinely fix a missed second-gunman detection on `video1.mp4`, but also
measurably raised false-positive risk on `Normal.mp4`: `max_struggle_score`
0.0→0.4, `max_sprint_score` 0.0→0.51 — needs an intermediate confidence value
tested, not resolved) are all deliberately untouched. Full detail on all of
these, if needed, is recoverable from this file's git history
(`git log -- HANDOVER.md`) or from `.claude/plans`.

## Standing instructions / preferences

- **Never add Claude as a co-author/contributor on commits in this repo.**
- Git identity: name `smoke`, email `vhulendamashamba4@gmail.com`.
- User prefers testing on **Kaggle (free GPU)** over local CPU runs — tight
  local RAM (see Environment below).
- **Actually vet any sourced dataset/checkpoint** — real class list fetched
  from the actual page/export (not a search-result summary or the project's
  own marketing description), real image count, license, and now also:
  **spot-check real sample thumbnails**, not just the page description —
  this session found two "CCTV"-named datasets whose actual sample images
  were a toy gun and a stock product photo. Roboflow Universe pages need a
  browser-like User-Agent to fetch via curl/WebFetch (plain requests get
  HTTP 403).
- This session's other standing instruction: **when a chunk of work
  fundamentally changes direction (like this rewrite), replace the handover
  rather than appending another session section to an already-long one.**

## Environment notes

- Machine: Windows 11, repo at
  `c:\Users\vhule\OneDrive\Desktop\Projects\mashtronicsAI`.
- Local venv at `.venv/` (`.venv\Scripts\python.exe`). **No NVIDIA GPU
  locally** — CPU-only inference, which is why nano YOLO models are used and
  why Kaggle is the validated workflow for actually running the pipeline.
- RAM was tight in earlier sessions (15.63 GB total, ~3.67 GB free observed
  at one point) — `--frame-stride` exists as an escape hatch if needed.

## Kaggle workflow

1. **New notebook** → Accelerator: GPU T4 → Internet: On. Given this
   session's "start from the bottom" instruction, don't reuse an existing
   notebook's leftover `/kaggle/working` state even if one happens to still
   have files in it.
2. `!git clone https://github.com/SMOKE484/mashtronicsAI.git && cd
   mashtronicsAI && pip install -q -e .` — if the repo somehow already exists
   in `/kaggle/working` (same session, re-run cell), `%cd
   /kaggle/working/mashtronicsAI && !git pull` instead of re-cloning.
   **Remember**: the local rewrite in this handover is not yet pushed to
   GitHub — `git pull` won't have it until it's committed and pushed.
3. For video: attach via "Add Input" → Dataset, then confirm the real mount
   path with `find /kaggle/input/ -iname "*.mp4"` (the sidebar's displayed
   path has been wrong before — missing a `datasets/<username>/` segment).
4. **`/kaggle/working` does not survive across separate Kaggle sessions** —
   only within one continuous session/disk. Anything worth keeping (trained
   checkpoints especially) needs to be saved as a proper Kaggle Dataset, not
   left sitting in `/kaggle/working`. This has already cost one checkpoint
   once.
5. Jupyter kernel gotcha: `!pip install -q -e .` doesn't get picked up by the
   *already-running* kernel for a direct `from driveway_guard... import ...`
   — only by new subprocess-spawned `python` processes (`!python -m ...` or
   `subprocess.run([...])`). Fix for a cell that needs to import the package
   directly: `sys.path.insert(0, "/kaggle/working/mashtronicsAI/src")` as
   its first line.
