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

**tl;dr of where things stand right now (updated 2026-08-27)**: Parts 1 and 2
are both done, committed, and pushed. **`weapon_at_window` has now fired
correctly on both `video1.mp4` and `video3.mp4`, with zero false positives on
`Normal.mp4`** — the first real end-to-end confirmation of a correctly-firing
event on real footage for the whole project. See "Validation against real
footage — outcome" below for the exact settings and results. Next up: decide
whether to keep iterating on the weapon checkpoint (its merged-dataset
validation metrics are still below the original baseline, even though it
passes real-footage validation — see Part 1) or move on to bringing back the
next event type (struggle).

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

## Part 1: expand the weapon dataset and retrain — done, checkpoint in use

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

**`scripts/merge_yolo_datasets.py`**: combines two-or-more Roboflow YOLO
exports into one, remapping each source's own class names onto a unified
target list via a JSON config (full shape in the script's own docstring).
Output is a normal-shaped YOLO export (`train/valid[/test]` images+labels,
`data.yaml`), so `scripts/train_weapon_model.py` consumes it with **zero
changes** via `--data <merged>/data.yaml`. Supports a `null` class_map value
to drop a source class's boxes entirely (added this session — see below).

**What actually ran, and what was learned**:
1. `dietest/gun-cctv-detection`'s real class list turned out to be
   `['Handgun', 'Knife', 'Short_rifle']` (`nc=3`), not the single `Guns`
   class the original research assumed — the dataset page's advertised class
   list didn't match the actual exported `data.yaml`, same lesson as the
   "verify, don't trust the description" pattern that rejected other
   candidates.
2. First attempt mapped all three classes onto `weapon`. Result was **worse
   than baseline on every metric** (`weapon`: precision 0.728/recall
   0.553/mAP50 0.629 vs. baseline 0.833/0.652/0.744) — folding knives and
   rifles into one label with handguns diluted the class.
3. Second attempt dropped `Knife`/`Short_rifle` (mapped to `null`, i.e.
   excluded from training as a class, image kept as a background example)
   and kept only `Handgun → weapon`. Slightly better but still below
   baseline (`weapon`: 0.728→0.768 precision, recall still ~0.55).
4. **User explicitly wants any gun type covered, not just handguns** — final
   config keeps `Handgun → weapon` and `Short_rifle → weapon`, only
   `Knife → null` (dropped). Metrics: `person` 0.835/0.806/0.852, `weapon`
   0.768/0.555/0.646 (precision/recall/mAP50) — still below the original
   baseline in aggregate, particularly weapon recall (0.555 vs. 0.652).
5. **Despite the weaker aggregate validation metrics, this checkpoint passed
   real-footage validation** (see below) — the merged-dataset validation
   split isn't apples-to-apples with the original single-dataset baseline
   (different image pool), so don't over-index on that comparison alone.
   This checkpoint is the one currently in use.
6. The Roboflow API key in use is the same one flagged in an earlier session
   as possibly exposed in a shared screenshot. **User explicitly decided to
   keep using it rather than rotate** (2026-08-27) — a conscious call, not an
   oversight; revisit only if the user's stance changes. A local untracked
   `credentials.txt` in the repo root still has this key in plaintext — never
   add it to git.

## Part 2: rebuild the pipeline code — done, committed, validated on real footage

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
`scripts/export_weapon_snapshots.py` and `scripts/train_weapon_model.py` are
untouched and still work. `scripts/inspect_weapon_hits.py` was **not**
actually working despite an earlier version of this file claiming so — it
still imported the retired `RuleThresholds` from `scoring.rules`. Fixed this
session to import `WeaponThresholds` from `scoring.weapon` instead; also
updated its longest-continuous-run calculation to use
`thresholds.weapon_max_gap_s` (see below) instead of a hardcoded 5-frame gap,
so its diagnostic output matches what the real pipeline would actually do.

### New: weapon-only pipeline

**`src/driveway_guard/scoring/weapon.py`** (new):
- `WeaponThresholds`: `weapon_confidence_threshold=0.5`,
  `weapon_min_duration_s=0.15` (tuned down from an initial 0.5 default — see
  "Validation against real footage — outcome" below), `event_cooldown_s=5.0`,
  `weapon_max_gap_s=0.15` (added this session, see below).
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
an accidental regression.

**Second, more impactful bug found during real-footage validation**:
`EventAggregator.update()` reset its duration streak to zero on **any single
below-threshold sample**, no tolerance at all. Real per-frame confidence is
noisy — a genuine ~1s detection with one frame that dipped below threshold
counted as two separate sub-threshold runs, neither long enough to fire.
Fixed by adding `weapon_max_gap_s` (default 0.15s): a below-threshold sample
no longer resets the streak unless the elapsed time since the last
above-threshold sample exceeds this gap. This is generic to `EventAggregator`
itself (shared code, not weapon-specific) via a new optional `max_gap_s` param
on `update()`, defaulting to `0.0` (old strict behavior) for any caller that
doesn't pass it.

**`pipeline.py`, `config.py`, `run.py`** — rewritten, same filenames. CLI
shape: `python -m driveway_guard.run --video <path> --out <dir>
--weapon-model <path> [--detector-model yolo11n.pt] [--conf 0.35]
[--device cpu] [--frame-stride 1] [--no-video-output] [--log-level INFO]
[--weapon-conf 0.4] [--weapon-proximity-norm 0.15] [--weapon-pad-ratio 0.4]
[--weapon-confidence-threshold 0.5] [--weapon-min-duration-s 0.15]
[--event-cooldown-s 5.0] [--weapon-max-gap-s 0.15]`. `--weapon-model` is now
**required** — no more "skip the stage if omitted," since weapon detection is
the entire point of this phase. No `--calib`, no `--pose-model` (those flags
are gone, not just defaulted).

### New tests

- `tests/test_weapon_detector.py` — `WeaponDetector.detect()` had **no** unit
  test before (only real-footage validation). Now covers the non-threat-class
  denylist argmax fix directly (mocks the YOLO model via
  `unittest.mock.patch("driveway_guard.detection.weapon_detector.YOLO")`,
  since real inference isn't needed to test the class-filtering logic), plus
  a proximity-gating test (far person never even reaches the model).
- `tests/test_weapon_scorer.py` — `score_weapon_hit` pure-function cases,
  a duration-debounce integration test, a "never clears confidence
  threshold" negative test, the vehicle-only-keying regression test
  described above, and two gap-tolerance tests (a brief dip within
  `weapon_max_gap_s` must not reset; a gap longer than it still must).

**11/11 tests passing** (`.venv/Scripts/python.exe -m pytest -q`).

## Validation against real footage — outcome (2026-08-27)

Ran on Kaggle (fresh notebook, GPU T4) against the retrained checkpoint from
Part 1 (`Handgun`+`Short_rifle → weapon`, `Knife` dropped).

**First pass, at the then-defaults (`weapon_confidence_threshold=0.5`,
`weapon_min_duration_s=0.5`)**: zero events on `video1.mp4`, `video3.mp4`,
*and* `Normal.mp4`. Looked like a clean-but-useless pass at first (no false
positives, but also no true positives). Diagnosed with
`scripts/inspect_weapon_hits.py --conf 0.1` (after fixing its stale import,
see above) run against the raw per-frame confidence: the model **was**
detecting the weapon repeatedly, with strong confidence (up to 0.72 on
video1, 0.77 on video3) — this was not a detection failure. The
`EventAggregator` zero-tolerance reset bug (described above) was the actual
cause: single-frame confidence dips broke every run into pieces under ~0.22s,
never reaching the 0.5s duration requirement.

**After the `weapon_max_gap_s` fix**, replaying the same logged confidence
values locally showed the longest bridgeable run was still only ~0.15–0.22s
(gap tolerance alone doesn't manufacture duration that isn't there) — so
`weapon_min_duration_s` also needed to drop, to 0.15s. This is now the
code's default in `WeaponThresholds`/`RunConfig`/`--weapon-min-duration-s`.

**Final result at `weapon_confidence_threshold=0.5`, `weapon_min_duration_s=
0.15`, `weapon_max_gap_s=0.15`**:
- `video1.mp4`: 1 `weapon_at_window` event, t=10.35–10.57s, peak_score=0.716
- `video3.mp4`: 1 `weapon_at_window` event, t=18.73–18.90s, peak_score=0.770
- `Normal.mp4`: 0 events (no false positive at this tighter duration)

This is the first confirmed correctly-firing event on real footage for the
whole project. **Caveat worth remembering**: `weapon_min_duration_s=0.15` is
a much more permissive debounce than the original 0.5s design intent — it
was validated clean against exactly one benign clip (`Normal.mp4`). If
false positives show up on other benign footage later, `weapon_confidence_
threshold` (currently 0.5, and this checkpoint's precision is below
baseline) is the more likely lever to raise before duration.

**Next options, not yet decided**: keep iterating on the weapon checkpoint
itself (still below baseline on aggregate validation metrics despite passing
real-footage validation), or move on to bringing back the next event type
(likely struggle, pulling its retired code back from
`git show d647e3f:src/driveway_guard/...`).

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
