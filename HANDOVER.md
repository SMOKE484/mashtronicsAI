# Handover: driveway-guard (mashtronicsAI)

Context for picking up this project in a new chat session, without the user
re-explaining. Originally written 2026-08-26 at the end of the session that
built v1; updated same day after a follow-up session that picked up weapon
detection; updated again 2026-08-27 after a third session that: fixed a real
bug in weapon detection, added and then fixed a new "vehicle flees after
being surrounded" detection capability, designed (but did not yet build) a
cross-event correlator, pushed everything to GitHub, and started a real-
footage Kaggle testing round that got **blocked on an unexplained
zero-events result**. Updated again after a fourth session that picked up
exactly there — ran the diagnostic that session 3 never got to, found and
fixed a real gating bug it exposed, confirmed the weapon-detection
checkpoint is genuinely working on real footage, then chased a user-reported
"second gunman never detected" report all the way down to its actual root
cause. See "Session 4" below — that section, not "Kaggle testing round 2",
now reflects the current state and the current most-urgent next action.
Written to be detailed enough that a new session doesn't need to re-derive
anything by re-reading code or re-running commands that already have known
answers below.

**tl;dr of where things stand right now**: code is committed and pushed
(`main` at `d647e3f`), 50/50 unit tests pass. The zero-events mystery from
session 3 is resolved: it was two things at once — a real gating bug in
struggle scoring (now fixed) and thresholds/duration requirements that are
simply stricter than what fast real-world footage produces. **Weapon
detection is confirmed to genuinely work** — the retrained checkpoint found
a real, clean, high-confidence gun detection in `video1.mp4`, it's just
~0.18s short of the 0.5s duration the pipeline currently requires before
flagging an event. Separately, the user visually confirmed `video1.mp4`
actually shows **two** people holding guns, and a frame-level diagnostic
proved the second one (dark clothing, low contrast, possibly partially
occluded by the vehicle) was **never detected as a person at all** by the
base tracker — not a weapon-detection bug, a person-detection/tracking
recall gap. An experiment to test whether a lower detector confidence or a
larger base model (`yolo11s.pt`) fixes that was handed to the user as ready-
to-run Kaggle cells, and **the session ended before results came back**.
That, plus two still-open threshold/design decisions (shorten
`weapon_min_duration_s`; re-key `weapon_at_window` debouncing to
vehicle-only to survive track-ID churn), are the highest-priority next
actions — see "Session 4" and "Next steps" below.

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
- **Struggle's dwell gate was comparing against the wrong proximity zone —
  fixed session 4.** `score_struggle()` (`scoring/rules.py`) requires, on
  the same frame, both `person_vehicle_distance_norm ≤
  proximity_norm_threshold` (0.08 — genuinely close) and `dwell_time_s ≥
  struggle_dwell_min_s` (1.5s — sustained). The bug: `dwell_time_s`
  (`features/extractor.py`) was tracked against `_RECORD_PROXIMITY_NORM`
  (0.5 — a much looser "is this pair even worth recording" zone, deliberately
  wide so approach/sprint/convergence signals are captured before someone
  reaches the vehicle), not the strict 0.08 struggle zone. So a person could
  dwell near the car for 6+ seconds without that dwell ever coinciding with
  a frame where they were also within 0.08 — the two gates were checking
  different notions of "close" and rarely lined up. Confirmed directly
  against real footage (`video3.mp4` from "Kaggle testing round 2" below):
  `max_dwell_s` (loose zone) reached 6.59s while `min_distance_norm` only
  ever dropped to 0.06 briefly, and `max_struggle_score` stayed exactly 0.0
  the entire clip. **Fix**: added a second field,
  `FrameFeatureVector.close_dwell_time_s`, tracked by a second
  `ProximityDwellTracker` keyed to a new `FeatureExtractor.__init__`
  parameter `close_proximity_norm` (default 0.08, matching
  `RuleThresholds.proximity_norm_threshold`). `run.py` now builds one
  `RuleThresholds` instance and passes `thresholds.proximity_norm_threshold`
  into `FeatureExtractor` explicitly, so the two proximity notions used by
  the extractor and the scorer share a single source of truth and can't
  silently drift apart again. `score_struggle()`'s two `dwell_time_s` checks
  (the proximity+dwell gate, and the `person_person_contact` contact-term
  check) now both read `close_dwell_time_s` instead. The old loose
  `dwell_time_s` field is kept as-is (still useful as a wider-context
  training-row signal, nothing else consumes it). 3 new tests added (2 in
  `test_feature_extractor.py` covering the extractor-level distinction, 1
  regression test in `test_rule_scorer.py` reproducing the exact bug
  pattern: high loose dwell + low close dwell ⇒ struggle score must stay
  0). 50/50 tests passing. Committed as `fc804fa`.
  **Re-tested against real footage post-fix** (unlike the weapon-class-filter
  fix above, which still hasn't been) — see "Session 4" below:
  `close_dwell_time_s` now correctly reports the true (much shorter, 0.44s)
  close-proximity duration on `video3.mp4`, confirming the fix works as
  intended. Struggle still scores 0 on that clip, but now for a real,
  understood reason (the interaction is fast, not sustained) rather than a
  gating bug — see "Session 4" for why that's actually a tuning question now,
  not a code defect.

## Real-world hijacking chronology (reference material, sourced this session)

The user shared a detailed real-world tactical breakdown of South African
driveway hijackings, sourced from the National Hijack Prevention Academy
(NHPA) and SAPS data (>70% of carjackings happen at/near home driveways).
This directly grounded two design decisions this session (the convergence
fleeing-signal fix, and the correlator design below) and should keep
informing threshold tuning once real footage is available. Five stages,
strict timeline, **whole event typically under 30-45 seconds start to
finish**:

1. **Tailgate / driveway stakeout** — attackers follow a target vehicle
   2-3 cars back from a shopping center/petrol station, or stake out a
   high-risk driveway in advance. **Not observable by this system** — it's
   watching the driveway, not the road; no vehicle-loitering-before-
   resident-arrival tracking exists. Acknowledged blind spot, correctly
   out of scope for v1.
2. **Box-in** — as the resident's car stops at the gate waiting for it to
   open, a second vehicle pulls in behind at an angle within seconds,
   sealing off reverse. → maps to `boxing_in`.
3. **Blitz / sprint** — within ~2 seconds of the block, 2-3 suspects exit
   the blocking vehicle and sprint to the driver/passenger windows
   (roughly opposite flanks — clears the convergence angular-spread gate).
   → maps to `sprint_approach` + `multi_directional_convergence`.
4. **Forced extraction** — a weapon is shown/tapped at the window, the
   driver is physically pulled out. → maps to `weapon_at_window` +
   `struggle`.
5. **The split** — one hijacker gets into the driver's seat (**driver
   possibly still inside** in some variants), both vehicles pull away
   rapidly in opposite directions. → this is exactly the scenario the
   "recently surrounded, now fleeing" fix above targets: the person who
   converged is no longer a separate tracked "approaching person," so the
   signal has to survive that disappearance.

Practical implication for the correlator design below: because each event
type has a different debounce duration (`boxing_in` needs 4s continuous
before it can even register a nonzero score, `weapon_at_window` only needs
0.5s), the **order events actually get flagged in the log will not
necessarily match the real-world order above** — `sprint_approach` could
get timestamped before `boxing_in` does, even though the block physically
happened first. Any cross-event logic must be order-agnostic.

## Cross-event correlator (designed and agreed this session — NOT YET BUILT)

The five event types fire independently right now. The chronology above
makes clear they're not independent in a real attack — `boxing_in` →
`sprint_approach`/`convergence` → `struggle`/`weapon_at_window` → fleeing,
all on the *same vehicle*, all within ~45 seconds, is a much stronger
signal than any one event alone (each individually still has plausible
innocent explanations — awkward parking, an argument, someone jogging up
to say hi). Proposed design, explained in plain language to the user and
explicitly confirmed correct ("Thats right") before the session moved on to
other things — **this was never actually implemented**, it's a green-lit
design waiting to be built:

- **Anchor entity**: the resident's vehicle `track_id` — the one thread
  that runs through `boxing_in` (as `resident_vehicle_track_id`),
  `sprint_approach`/`struggle`/`weapon_at_window`/`multi_directional_convergence`
  (all as `vehicle_track_id`). The blocking/getaway vehicle's track ID is
  *not* the anchor — it only appears inside `boxing_in` as context.
- **Co-occurrence within a window, not strict order** (see chronology
  section above for why order can't be trusted) — check for 2+ distinct
  event types landing on the same anchor vehicle within roughly 45-60
  seconds (real-world number + some slack), not a stage-ordered state
  machine.
- **Adds a layer, doesn't replace anything** — the five events keep firing
  exactly as they do now, independently useful for debugging/tuning. The
  correlator would emit one additional "this looks like one connected
  incident, treat as urgent" signal referencing which underlying events
  triggered it.
- **Explicit honesty caveat, already agreed with the user**: this compounds
  several still-untuned, still-unvalidated rule-based heuristics. Agreeing
  with each other raises *relative* confidence, it doesn't prove anything.
  Frame output as "elevated priority for review," never "confirmed
  hijacking."
- **Known risk inherited, not introduced**: if people crowding the vehicle
  cause the tracker to lose and reassign its ID mid-sequence (plausible
  during the struggle/extraction stage with real footage), the correlator
  loses the thread — same tracker-continuity weakness the rest of the
  system already has, just more exposed here since it spans a longer
  window than any single event type does.

**Next session should ask the user whether they still want this built now**
(likely yes — it was the direct topic before testing took over), or
whether to keep prioritizing the zero-events diagnostic first. Given the
diagnostic is actively blocking any real validation, probably diagnose
first, build the correlator once individual events are confirmed to fire
correctly on real footage — no point correlating five signals that aren't
individually trustworthy yet.

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

scripts/                         # all standalone utilities below: NOT part of the
                                 #   driveway_guard package, not imported anywhere at
                                 #   pipeline runtime, each has its own --help via argparse.
├── train_weapon_model.py       # Fine-tunes a YOLO checkpoint on any YOLO-format
│                                #   dataset. Full CLI/behavior in "Weapon detection" below.
├── diagnose_pipeline.py         # Added session 4. Runs the full detection/pose/weapon/
│                                #   feature/scoring pipeline across a whole clip and
│                                #   reports the BEST value every signal ever reached
│                                #   (min_distance_norm, max_close_dwell_s, max_dwell_s,
│                                #   max_approach_speed_px_s, max_struggle_score,
│                                #   max_sprint_score, max_weapon_score,
│                                #   max_convergence_approachers/spread_deg/score,
│                                #   frames_with_pair), bypassing EventAggregator's
│                                #   threshold+debounce entirely. This is "Diagnostic 2"
│                                #   from "Kaggle testing round 2" below, committed this
│                                #   time instead of being left in a chat transcript.
│                                #   Optional --out writes the summary as JSON.
├── inspect_weapon_hits.py       # Added session 4. Runs only Tracker + WeaponDetector
│                                #   (skips pose/features) and prints every individual
│                                #   raw weapon-detector hit, frame by frame, with
│                                #   timestamp + confidence + which person track. Reports
│                                #   the longest continuous run of consecutive
│                                #   above-threshold frames, converted to seconds, so a
│                                #   single-frame spike can be told apart from a genuine
│                                #   sustained detection. See "Session 4" below for what
│                                #   this revealed on video1/video3.
├── export_weapon_snapshots.py   # Added session 4. Picks the N highest-confidence
│                                #   weapon hits on a clip (at least --min-gap-s apart in
│                                #   time, to avoid near-duplicate consecutive frames),
│                                #   draws the weapon/person boxes, and saves a full-frame
│                                #   PNG + a padded close-up crop for each — for visually
│                                #   confirming a hit is really a gun. Set --count high and
│                                #   --min-gap-s 0 to export literally every hit instead of
│                                #   a representative sample.
└── inspect_frame.py             # Added session 4. Deep-dive on one exact frame
                                 #   (--frame-idx or --at-seconds): lists every tracked
                                 #   person/vehicle, whether each person was gated as
                                 #   "near a vehicle" for weapon checking at all, and (if
                                 #   --weapon-model given) EVERY raw box the weapon model
                                 #   found in each gated person's crop — not just the
                                 #   single top-1 confidence box WeaponDetector.detect()
                                 #   actually keeps. Also writes an annotated PNG. Built to
                                 #   tell apart three failure modes that all look identical
                                 #   from events.json alone: a person never tracked at all,
                                 #   tracked-but-gated-out, or tracked+gated+detected but
                                 #   silently dropped by the top-1-per-crop logic. See
                                 #   "Session 4" below for the real finding on video1.

calib/example_driveway.json     # placeholder calibration, 1920x1080 — must match your video's exact resolution or it errors loudly
tests/                          # 50 passing unit tests, no video/model needed (pure logic)
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
- **50/50 local pytest tests passing** (pure-logic tests only — geometry,
  scorer, track state, feature extractor, event debounce; no video/model
  dependency). 3 new tests added session 4 for the struggle dwell-gate fix
  (2 extractor-level, 1 scorer-level regression test — see "Key design
  decisions" above). Confirmed passing (`.venv/Scripts/python.exe -m
  pytest -q` → `50 passed`) before every commit.
- Pushed to GitHub: **https://github.com/SMOKE484/mashtronicsAI** (public
  repo). `main` is up to date through commit `d647e3f`. Full commit
  history as of end of session 4:
  - `b2f156b` — "Add v1 driveway anomaly detection pipeline" (session 1)
  - `30ff36a` — "Declare lap dependency explicitly; fix blocking-overlap
    test expectation" (session 2)
  - `7fa9c2f` — "Add weapon detection training script and sourcing docs"
    (session 2)
  - `7f0477f` — "Fix weapon class filtering; detect vehicle fleeing after
    convergence" (session 3) — bundles the `WeaponDetector`
    non-threat-class-denylist fix, the convergence fleeing-bonus +
    recently-surrounded-decoupling fix, and the `run.py` flagged-events
    console summary. 10 files changed.
  - `fc804fa` — "Fix struggle dwell gate comparing against wrong proximity
    zone" (session 4) — the `close_dwell_time_s` fix, see "Key design
    decisions" above. 7 files changed, 3 new tests.
  - `52042d6` — "Commit the zero-events diagnostic script instead of
    leaving it in chat" (session 4) — adds `scripts/diagnose_pipeline.py`.
  - `12dc4b6` — "Add frame-by-frame weapon hit inspector" (session 4) —
    adds `scripts/inspect_weapon_hits.py`.
  - `c11d62f` — "Add weapon-hit snapshot exporter for visual spot-checking"
    (session 4) — adds `scripts/export_weapon_snapshots.py`.
  - `d647e3f` — "Add single-frame deep-dive to diagnose the second-gun-
    missed report" (session 4) — adds `scripts/inspect_frame.py`.
  - No uncommitted changes as of the end of session 4 — `git status` was
    clean after the last push. (An `output.txt` scratch file — pasted
    Kaggle terminal output, most recently the weapon-hit-inspector results
    — is untracked and deliberately excluded from every commit; harmless
    to delete if it's cluttering the working tree, or leave it, it keeps
    getting overwritten with whatever was last pasted.)
- **Three real end-to-end Kaggle test rounds so far.** Weapon detection is
  now confirmed to genuinely detect real weapons on real footage (round 3);
  the other four event types still have not produced a single confirmed-
  correct flagged event on real footage, though round 3 now explains why in
  concrete, per-signal terms rather than a mystery zero:
  - **Round 1** (session 2): `Normal.mp4` (`training1` Kaggle dataset,
    real mount path
    `/kaggle/input/datasets/vhulendamashamba/training1/Normal.mp4` — the
    sidebar display path is misleading, always verify with
    `find /kaggle/input/ -iname "*.mp4"`). Ran cleanly through 300+
    frames, 0 events flagged. Never visually confirmed whether detection
    was actually working — annotated.mp4 from that run was never reviewed.
  - **Round 2** (session 3): batch of 6 clips including 5 real sourced
    hijacking videos, 0 events across the board, cause not yet identified
    — see "Kaggle testing round 2" below for the full historical record of
    that investigation (still accurate as a record of what was tried and
    found; superseded as a statement of current priority by "Session 4"
    below).
  - **Round 3 (session 4 — resolved, see "Session 4" below for full
    detail)**: re-ran the "Diagnostic 2" script session 3 never got to run
    (now `scripts/diagnose_pipeline.py`), found and fixed the struggle
    dwell-gate bug, then used two new inspection scripts to confirm the
    retrained weapon checkpoint is genuinely detecting real guns in
    `video1.mp4` and `video3.mp4` — just not for long enough, continuously,
    to survive the pipeline's 0.5s debounce. A follow-up user report ("two
    people had guns, only one was detected") led to a frame-level
    diagnostic that found the second gunman (dark clothing, low contrast)
    was never detected as a person at all by the base tracker — a
    detection/tracking recall gap, not a weapon-detection bug. An
    experiment to test a lower detector confidence and a larger base model
    was handed to the user as ready-to-run cells; **results not yet back
    when the session ended.**

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

**Update, session 3: this checkpoint is now gone.** It only ever lived in
`/kaggle/working`, which does not persist across separate Kaggle sessions
(only within one continuous session/disk — see "Kaggle workflow" below,
this distinction bit the user twice now). A fresh Kaggle session this
session found `/kaggle/working` completely empty. **Needs retraining**
(same command as above, ~35 minutes on a T4) **and this time saving
`weapon_model.pt` as its own Kaggle Dataset** so it survives future
sessions instead of relying on `/kaggle/working`. See "Kaggle testing
round 2" below for the full account of what happened when testing
proceeded without it.

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

## Kaggle testing round 2 (session 3 — historical record; resolved in "Session 4" below)

**Status update from session 4**: the "Diagnostic 2" script this section
describes as not-yet-run *was* run in session 4 (as the newly-committed
`scripts/diagnose_pipeline.py`), and the zero-events mystery is resolved —
see "Session 4" below for the actual results and what they meant. This
section is kept as-is as the historical record of round 2's setup and
Diagnostic 1, which are both still accurate; only Diagnostic 2's "NOT YET
RUN" status below is stale.

Goal: run the pipeline against real sourced hijacking footage and confirm
events actually fire correctly. Result at the time: **0 events on every
clip tested**, cause not yet identified.

### Security note (unresolved — check this first)

A screenshot the user shared mid-session contained a **real Roboflow API
key in plaintext** (visible in a `Roboflow(api_key="...")` cell from
yesterday's training notebook). The user was told to rotate/regenerate it
on Roboflow immediately (Settings → API Keys), since it had been exposed
in the conversation. **Whether they actually did this was never
confirmed** — worth checking early next session, and worth treating any
Roboflow-auth errors as possibly explained by this if it comes up.

### Setup this session

- New Kaggle session (fresh `/kaggle/working` — confirmed empty,
  `git clone` needed, not just `git pull`). This is the session where
  `weapon_model.pt` was discovered to be gone (see "Weapon detection"
  section above).
- Video dataset uploaded: `/kaggle/input/datasets/vhulendamashamba/hijackings/`
  containing `video1.mp4` through `video5.mp4`.
- **Important correction**: earlier in this same session, 3 detailed
  Gemini/Veo prompts were drafted for synthetic normal/near-miss/hijacking
  test clips (see "Synthetic test clip prompts" section below) — but the
  `hijackings` dataset turned out to be **real sourced CCTV footage, not
  those synthetic clips**. Confirmed by inspecting a frame: burned-in
  timestamp `08/27/2018 17:12:13`, a `CAMERA03` watermark, 640x480
  resolution, ~41 fps, 1212 frames (~29.6s duration) — notably close to
  the real-world "under 30-45 seconds" hijacking duration from the
  chronology section above, consistent with this being genuine incident
  footage. Don't assume future references to "the hijackings dataset" mean
  AI-generated content — it doesn't.
- No `--weapon-model` (lost) and no `--calib` (never built for this
  dataset's resolution — the existing `calib/example_driveway.json` is
  1920x1080, this footage is 640x480, incompatible) were used for the
  batch run. So `weapon_at_window` and `boxing_in` (and the
  calibration-gated convergence fleeing-bonus) were structurally disabled
  for this run — only `struggle`, `sprint_approach`, and base
  `multi_directional_convergence` (no fleeing bonus) were actually live.

### Batch run and result

Ran all 6 clips (5 hijacking videos + `Normal.mp4` from `training1`)
through `python -m driveway_guard.run` via a loop, one output dir each
under `/kaggle/working/results/<clip_name>/`. **Every single clip produced
0 flagged events**, including clips the user confirms show real hijackings.
Zero across the board (not "mostly zero, one partial hit") is the reason
this reads as a likely systemic issue rather than just conservative
thresholds — if detection/scoring were basically working, at least
*something* on at least one real hijacking clip should have gotten close.

### Diagnostic 1 — confirms detection is not totally broken

Ran a standalone diagnostic (bypassing pose/weapon/scoring, just
`Tracker.track()` directly) on the first 60 frames of `video3.mp4`:

- `resolution: 640x480 | fps: 40.997 | frame_count: 1212`
- `person detected in 12/60 frames, vehicle detected in 40/60 frames`
- A displayed annotated frame showed a real, correctly-boxed vehicle
  detection, confidence `0.37` — notably **right at the edge of the
  `0.35` conf threshold** used by both the tracker and the default
  `--conf` CLI flag.

**Conclusion**: the detector genuinely does see people and vehicles in
this real footage — this is not a codec/decode failure or a "model sees
nothing" situation. Vehicle detections firing in only 40/60 frames (67%)
at a borderline 0.37 confidence is a plausible early hint that detection
may be flickering in and out across frames on real (lower-res, compressed,
possibly motion-blurred) CCTV footage in a way it likely wouldn't on clean
synthetic test data — worth keeping in mind, since intermittent
tracking/detection would repeatedly reset the duration-based dwell/
proximity timers that `struggle` and `boxing_in` depend on
(`ProximityDwellTracker.update()` zeroes the clock the instant a pair
isn't "active" for even one frame).

### Diagnostic 2 — run in session 4, see "Session 4" below for results

A second, deeper diagnostic script was written and handed to the user as
"cell 8," but **the session ended before it was run and before any results
came back**. *(Session 4 update: it was run, on `video3.mp4` — see
"Session 4" below for the actual numbers and what they revealed. The
description of what the script does, immediately below, is unchanged; it's
now committed as `scripts/diagnose_pipeline.py` instead of living only in a
chat transcript.)* It runs the full `Tracker` + `PoseEstimator` + `FeatureExtractor` +
`compute_convergence` + raw `score_struggle`/`score_sprint`/
`score_convergence` (bypassing `EventAggregator`'s threshold+debounce
entirely) across the **whole** clip (not just 60 frames), and tracks the
best/max value each signal ever reaches:
`min_distance_norm`, `max_dwell_s`, `max_approach_speed_px_s`,
`max_struggle_score`, `max_sprint_score`, `max_convergence_approachers`,
`max_convergence_spread_deg`, `max_convergence_score`,
`frames_with_pair`. The full script is in the conversation transcript
(search for "Cell 8" — it imports directly rather than going through the
CLI, using `sys.path.insert(0, "/kaggle/working/mashtronicsAI/src")` to
work around the Jupyter kernel quirk below).

**Once this runs, compare its output against these thresholds** (all in
`RuleThresholds`, `scoring/rules.py`) to diagnose which of two very
different problems this is:
- If the max values come back *close to but under* threshold (e.g.
  `max_struggle_score` = 0.4-0.6, `min_distance_norm` just above 0.08,
  `max_dwell_s` just under 1.5s) → this is a **threshold-tuning problem**,
  the scenario is being seen roughly correctly, thresholds are just too
  strict for real footage (plausible — they're explicitly documented as
  "starting guesses" never tuned against real clips).
- If the max values come back *nowhere close* (e.g. `min_distance_norm`
  never drops much below 0.5, `frames_with_pair` is near zero, convergence
  never sees 2+ simultaneous approachers) → this is a **detection/tracking
  robustness problem** on real footage (intermittent detection breaking
  continuity, pose estimation failing more often on compressed/low-res
  footage so `struggle`'s joint-velocity/arms-raised terms stay `None`,
  or the framing/distance in this specific footage genuinely doesn't
  bring bounding-box *centroids* as close together as the thresholds
  assume even when a person is visibly at the window) — this needs
  code/robustness work, not just threshold tuning.
Key reference thresholds to compare against: `proximity_norm_threshold`
(struggle) = 0.08, `struggle_dwell_min_s` = 1.5,
`sprint_speed_px_s_threshold` = 650, `convergence_min_approachers` = 2,
`convergence_angle_threshold_deg` = 90, and the flag threshold everything
ultimately needs to clear, `risk_score_flag_threshold` = 0.65.

### Jupyter/Kaggle operational gotcha discovered and solved this session

`!pip install -q -e .` run in a notebook cell does **not** get picked up
by that notebook's own already-running kernel process for a direct
`import driveway_guard` — only by *new* subprocess-spawned `python`
processes (e.g. `subprocess.run(["python", "-m", "driveway_guard.run", ...])`
or `!python -m ...`), because editable-install path entries are only read
by Python's `site` module at interpreter startup, and the kernel was
already running before the install happened. This is why the batch run
(subprocess-based) worked fine while a later cell doing
`from driveway_guard.detection.tracker import Tracker` directly failed
with `ModuleNotFoundError`. **Fix, no kernel restart needed**: put
`sys.path.insert(0, "/kaggle/working/mashtronicsAI/src")` as the first
line of any cell that needs to `import driveway_guard` directly rather
than going through the CLI/subprocess. Apply this in every future
diagnostic-style cell.

### Notebook cell layout as of end of session (for resuming exactly)

1. Yesterday's Roboflow dataset-download cell (leftover, not needed today).
2. Yesterday's `cd`+`pull`+train command (leftover — don't reuse the
   training line, checkpoint doesn't need retraining until it's actually
   time to fix the "weapon_model.pt is gone" problem).
3. This session's fresh `cd`/clone-if-missing + `git pull` + `pip install -e .`.
4. `glob` over `/kaggle/input/**/*.mp4` to list all 6 clips.
5. Batch-run loop (`subprocess.run`) over all 6 clips, no `--weapon-model`,
   no `--calib`, one output dir each under `/kaggle/working/results/`.
6. Read + print `events.json` from each result dir side by side — showed
   0 events everywhere.
7. Diagnostic 1 (raw tracker only, first 60 frames of `video3.mp4`, with
   the `sys.path` fix) — ran, results above.
8. Diagnostic 2 (full feature+scoring pipeline, whole clip, max-value
   tracking) — **written and handed to the user, not yet run**. This is
   where the session ended.

## Session 4 (picks up exactly where session 3 left off — READ THIS FIRST for current state)

Continuation of the same Kaggle notebook/session the user had open at the
end of session 3 (same `/kaggle/working/mashtronicsAI` checkout, same
`hijackings` dataset already attached). Four things happened in order:
the zero-events mystery got resolved, a real code bug got fixed, weapon
detection got confirmed to actually work, and a user-reported "second gun
missed" observation got traced to its real root cause. Ends with an
experiment handed to the user, unrun.

### 1. Diagnostic 2 finally ran, on `video3.mp4`

Result (whole clip, `EventAggregator` bypassed):
```
min_distance_norm: 0.0606695288833145
max_dwell_s: 6.585891089108912
max_approach_speed_px_s: 1683.930407390042
max_struggle_score: 0.0
max_sprint_score: 1.0
max_convergence_approachers: 2
max_convergence_spread_deg: 24.26892617080597
max_convergence_score: 0.0
frames_with_pair: 536
```
Reading this against the reference thresholds (`proximity_norm_threshold`
0.08, `struggle_dwell_min_s` 1.5, `sprint_speed_px_s_threshold` 650,
`convergence_angle_threshold_deg` 90, `risk_score_flag_threshold` 0.65):

- **Struggle stayed 0.0 despite both individual gates looking satisfied**
  (`min_distance_norm` 0.06 < 0.08; `max_dwell_s` 6.59 > 1.5) — this
  contradiction is exactly what led to finding the dwell-gate bug (see "Key
  design decisions" above): the two numbers were never true on the *same*
  frame, because `dwell_time_s` was measuring a much looser proximity zone
  than the 0.08 the distance gate checks.
- **Sprint hit a perfect 1.0** (`max_approach_speed_px_s` 1683, over double
  `sprint_speed_px_s_threshold`) — but recall this bypasses debounce, so it
  doesn't by itself mean an event would have fired; a single-frame spike
  (e.g. a tracker ID jump) wouldn't survive `EventAggregator`'s
  ≥2-consecutive-samples / ≥`event_min_duration_s` (0.3s) requirement. Not
  independently re-verified this session which case it was — worth checking
  before trusting sprint detection on this clip.
- **Convergence's angular spread topped out at 24° vs. the 90° gate** —
  `num_simultaneous_approachers` did reach the required 2, but they
  approached from similar angles rather than opposite flanks. Given the
  user later confirmed (see below) the victim never left the vehicle,
  this is consistent with both attackers converging on the same side/window
  rather than surrounding the car from multiple directions — the scenario
  `multi_directional_convergence` is shaped for may just not be what this
  clip shows, rather than a bug.

### 2. Struggle dwell-gate bug found and fixed

Covered in full under "Key design decisions" above (the `close_dwell_time_s`
bullet) — don't duplicate here, just the headline: `dwell_time_s` was
measuring the wrong (too loose) proximity zone, fixed, 3 new tests, commit
`fc804fa`, pushed.

**Re-running Diagnostic 2 on the same clip post-fix** (this is the
important confirmation step — the fix was verified against real footage,
not just unit tests):
```
min_distance_norm: 0.0606695288833145
max_dwell_s: 6.585891089108912
max_close_dwell_s: 0.43905940594059345    <-- NEW, this is the fixed field
max_approach_speed_px_s: 1683.930407390042
max_struggle_score: 0.0
max_sprint_score: 1.0
max_weapon_score: 0.6758406758308411      <-- weapon model now wired in, see below
max_convergence_approachers: 2
max_convergence_spread_deg: 24.26892617080597
max_convergence_score: 0.0
```
`close_dwell_time_s` (the correctly-gated field) tops out at **0.44s** —
far below the old misleading `max_dwell_s` of 6.59s. This confirms the fix
works: it's now telling the truth about how long the person was actually
*close*, not how long they were merely somewhere in the wide recording
zone. Struggle still scores 0, but now for a real, legible reason: nobody
held tight proximity to the vehicle continuously for anywhere near the
required 1.5 seconds. **The user's own read on why, independently
volunteered**: in this clip the driver never gets out of the car — she
stays seated, the attacker reaches in/at the window. That matches the
data — a quick reach-and-point is not the sustained two-people-in-contact
struggle the `struggle` event type's formula assumes (it needs either
`person_person_contact`, which requires a second distinct visible person
track, or pose-derived arm-raise/joint-velocity signals from an exposed
person — neither holds up well for "attacker leans into a window at a
seated victim"). **This is `weapon_at_window`'s actual scenario**, which is
exactly what got checked next.

**Also notable**: `max_weapon_score` is now 0.676, clearing
`weapon_confidence_threshold` (0.5) — first real positive signal of the
whole testing effort. But the full batch run (with `--weapon-model` wired
in, no `--calib` still — see "Kaggle testing round 2" setup notes above for
why) still produced **0 events on all 6 clips**. That gap — a max score
above threshold but still no event — is exactly what section 3 below
digs into.

### 3. Weapon detection confirmed genuinely working — it's a duration/tracking problem, not a detection failure

New script `scripts/inspect_weapon_hits.py` (committed `12dc4b6`) prints
every individual raw weapon-model hit frame-by-frame and reports the
longest unbroken run of consecutive above-threshold frames, converted to
seconds. Run on both `video1.mp4` and `video3.mp4` (the two clips the user
independently confirmed show a gun at the window):

**video1.mp4** (816x448, ~40.5fps, 595 frames):
- Two bursts: t≈5.0–6.1s (person track 9, confidence 0.53–0.72, patchy) and
  a much stronger one at t≈9.8–11.7s (person track 14).
- Inside the second burst, **frames 461–474 (t=11.39s–11.71s) are 14
  consecutive frames, every single one ≥0.5 confidence**, several 0.6–0.83,
  zero dips. Genuinely clean, confident, sustained-looking detection.
- Problem: 14 frames at ~41fps is only **0.32 seconds** of real time.
  `weapon_min_duration_s` requires **0.5s** continuous. This is a real
  detection that's simply shorter than the debounce window assumes — not a
  detector failure.
- Summary: `frames_with_hit_above_score_threshold=28`,
  `longest_continuous_run_above_threshold_s=0.32`, `max_confidence_seen=0.828`.

**video3.mp4** (640x480, ~41fps, 1212 frames):
- Weaker and choppier: longest clean run only **0.17s**.
- **Track-ID churn**: hits land on five different person track IDs (31, 35,
  37, 38, 45) across t=16.0–19.6s (~3.5s span) — the tracker is losing and
  re-acquiring what's plausibly the same person, likely from occlusion by
  the vehicle or another person. Since `weapon_at_window`'s debounce state
  in `EventAggregator` is keyed by `(person_track_id, vehicle_track_id)`,
  every ID switch resets accumulated duration back to zero regardless of
  confidence — compounding the already-short detection windows.
- Summary: `frames_with_hit_above_score_threshold=19`,
  `longest_continuous_run_above_threshold_s=0.17`, `max_confidence_seen=0.676`.

**Two design decisions were surfaced but explicitly left open, pending user
confirmation — do not silently pick one without checking**:
1. Lower `weapon_min_duration_s` (currently 0.5s) — video1's clean 0.32s
   run is real, concrete evidence it may be calibrated too strict for fast
   real-world gun-at-window moments. ~0.3s was the number discussed but not
   agreed.
2. Re-key `weapon_at_window`'s `EventAggregator` state to the vehicle
   alone (matching the precedent already set by convergence's
   "recently surrounded" fix — see "Key design decisions" above), instead
   of `(person, vehicle)`, so tracker ID churn like video3's doesn't reset
   progress. Not yet implemented.

The user asked to have the results explained in plain language before any
fix was proposed (both explanations were given, in full, in-conversation —
not reproduced here); **no final decision was made on either point before
the session moved on** to the next finding below. Both are open decisions
for the next session — see "Next steps".

### 4. "Two people had guns, only one detected" — traced to a person-detection miss, not a weapon-detection bug

New script `scripts/export_weapon_snapshots.py` (committed `c11d62f`) was
used first to export/view the highest-confidence weapon hits from video1 as
annotated images. After reviewing them, **the user reported that video1
actually shows two people holding guns at once, and only one was ever
flagged** — with a specific, important detail: the second person was
wearing all-black clothing, holding a black gun, pointed at the window.

Built `scripts/inspect_frame.py` (committed `d647e3f`) to isolate exactly
which of three possible causes this was (they're indistinguishable from
`events.json` alone): the second person never tracked at all; tracked but
gated out (not counted as "near a vehicle", so the weapon model never even
ran on their crop); or tracked, gated, and detected, but silently dropped
because `WeaponDetector.detect()` only keeps the single highest-confidence
threat box per person crop (relevant if two people end up in the same
padded crop).

Ran it at t=11.5s on video1 (the frame with the 0.828-confidence hit). The
annotated output showed **only one green "person" box** (around the
lighter-clothed person reaching toward the window, with the weapon box
correctly drawn at confidence 0.62) — **the person in black near the
driver's door had no bounding box at all**. Confirmed: this is cause #1,
**the base tracker's person detector never detected the second person as a
person in this frame, at all**. Weapon detection never had a chance to run
on them, because it only ever runs on already-tracked, already-gated
persons — this is upstream of anything weapon-specific.

Likely explanation (not yet confirmed): low contrast between dark clothing
and a dark gun/shadowed area, possibly combined with partial occlusion by
the vehicle from this camera's angle — a known hard case for the small
"nano" COCO detector (`yolo11n.pt`) used for the base person/vehicle
tracker (this is unrelated to the weapon model's own recall gap discussed
in "Weapon detection" below — this miss happens one stage earlier, before
the weapon model is ever invoked).

**Experiment designed but not yet run** — handed to the user as ready
Kaggle cells, session ended before results came back:
- Re-run `inspect_frame.py` on the same t=11.5s frame of video1 with
  `--tracker-conf 0.15` (down from the default 0.35) — tests whether this
  is simply a confidence-threshold tuning issue.
- Also try `--detector-model yolo11s.pt` (a larger YOLO11 variant than the
  nano model currently used everywhere, still Kaggle-GPU-lightweight) —
  tests whether it's a base-model capacity limitation instead.
- Also try both together.
- **If the person still isn't detected even with a bigger model and lower
  confidence**, that would point toward a genuine training-data-level gap
  in the base detector for this kind of low-contrast/occluded case — not
  fixable by a config change, would need either a fine-tuned person
  detector or accepting the limitation.

**This is the single most important next action** — see "Next steps"
below.

## Synthetic test clip prompts (drafted this session, not currently the priority)

Three detailed Gemini/Veo video-generation prompts were drafted for
normal / near-miss / full-hijacking scenarios, all sharing one consistent
fixed-CCTV-angle scene description so the same calibration file could be
reused across all three. Full prompt text is in the conversation
transcript, not reproduced here (they're long). Key points if picked back
up later:
- Keep the exact same static elevated driveway-camera scene description
  across every prompt for framing consistency.
- Content-filter guidance: avoid words like "gun"/"firearm"/"attack"/
  "victim" in the weapon/extraction parts — phrase around neutral,
  behavior-focused language ("dark handheld object," "gestures," "is
  escorted away") since Gemini/Veo's safety filters react to the words,
  not the underlying detection-relevant geometry, which doesn't need
  graphic content anyway.
- Individual Veo/Gemini clips are capped around ~8 seconds — a full
  ~30-40s hijacking sequence needs generating as several per-stage clips
  and stitching them, not one continuous generation.
- **Not currently the priority** — the user is testing against real
  sourced footage instead (see "Kaggle testing round 2" above), which is
  preferable per the existing standing preference for real footage. Revisit
  only if more controlled/repeatable synthetic variety is needed later
  (e.g. isolating one event type cleanly, which real footage rarely gives
  you cleanly).

Also researched (in case a free video-gen tool is needed later): as of
2026, Sora has no free tier at all (discontinued January-April 2026); Pika
and Runway have free tiers but are low-resolution (Pika free tier is
480p), short (5-10s), and watermarked — likely too degraded for reliable
YOLO detection at driveway-camera distances; Kling/Luma no longer have
meaningful free tiers. The more promising free route if needed: **open-
source self-hosted models runnable on the same Kaggle GPU workflow already
in use** — Wan 2.2 (Apache 2.0, Alibaba) or HunyuanVideo-1.5 (Tencent,
tuned for consumer GPUs) — no watermark, no clip-count cap beyond Kaggle's
own 30 GPU-hrs/week, though the commonly-recommended Wan 2.2 config wants
a 24GB GPU vs. Kaggle's free ~15GB T4, so the lighter 5B variant would be
needed.

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

## Kaggle workflow (validated as working across four sessions now)

1. New notebook → Accelerator: GPU T4 x2 → Internet: On.
2. If `/kaggle/working/mashtronicsAI` doesn't already exist:
   `!git clone https://github.com/SMOKE484/mashtronicsAI.git && cd mashtronicsAI && pip install -q -e .`
   If it **does** already exist from a prior run in the same session/disk,
   `%cd /kaggle/working/mashtronicsAI` and `!git pull` instead of
   re-cloning — re-cloning into a non-empty directory fails with
   `fatal: destination path ... already exists`. Always `git pull` even
   if it does exist, in case the clone predates a later push.
   **Important correction from session 3**: `/kaggle/working` only
   persists *within* one continuous session/disk (confirmed across
   multiple notebook edits in session 2) — it does **not** survive into a
   genuinely new session. Session 3 started with a completely empty
   `/kaggle/working` despite session 2 having built a whole training run
   there the day before, and lost `weapon_model.pt` as a result. **Never
   treat anything in `/kaggle/working` as durable across sessions** —
   anything worth keeping (trained checkpoints especially) needs to be
   saved as a proper Kaggle Dataset, not left sitting in `/kaggle/working`.
3. For pipeline runs on video: attach video via "Add Input" → Dataset. Get
   the real mount path with `find /kaggle/input/ -iname "*.mp4"` — don't
   trust the sidebar's displayed path, it's been wrong before (missing a
   `datasets/<username>/` path segment).
4. For weapon-model training: see README.md's "Weapon detection model"
   section for the up-to-date Roboflow dataset-export snippet and training
   command — don't duplicate it here beyond what's in the "Weapon
   detection" section above, keep README as the source of truth since
   it's more likely to be kept in sync with the actual script if the
   dataset choice changes again. **This time, save `weapon_model.pt` as a
   Kaggle Dataset immediately after training** — see the point above about
   why.
5. **Jupyter kernel gotcha (session 3)**: after `!pip install -q -e .`, a
   direct `from driveway_guard... import ...` in a notebook cell will fail
   with `ModuleNotFoundError` even though `!python -m driveway_guard.run`
   works fine — the notebook's own kernel process was already running
   before the install happened. Fix: put
   `sys.path.insert(0, "/kaggle/working/mashtronicsAI/src")` as the first
   line of any cell that imports the package directly instead of going
   through the CLI. See "Kaggle testing round 2" above for the full
   account.
6. Known Kaggle friction point (**unresolved**): see "Standing
   instructions"/"Weapon detection next steps" item 7 above re: adding a
   new dataset version to `training1`.

## Next steps (in priority order)

1. **Run the person-detection experiment from "Session 4" section 4
   above — the single most important next action.** On `video1.mp4` at
   t=11.5s, re-run `scripts/inspect_frame.py` with `--tracker-conf 0.15`,
   then with `--detector-model yolo11s.pt`, then both together, to find out
   whether the missed second gunman (dark clothing, low contrast, no person
   box at all from the base tracker) is a cheap confidence-threshold fix, a
   base-model-capacity issue, or a deeper training-data gap that needs
   neither knob. Exact ready-to-paste cells were handed to the user; just
   need the results read back and interpreted.
2. Resolve the two open weapon-detection design decisions from "Session 4"
   section 3 above (both were explained to the user but never decided):
   - Whether to lower `weapon_min_duration_s` (0.5s → ~0.3s was discussed),
     backed by video1's genuine, clean, uninterrupted 0.32s detection run
     that only just missed the current bar.
   - Whether to re-key `weapon_at_window`'s `EventAggregator` state to the
     vehicle alone instead of `(person, vehicle)`, mirroring the precedent
     already set by convergence's "recently surrounded" fix, so video3's
     person-ID churn (5 different track IDs in ~3.5s) stops resetting
     accumulated duration. Both are `scoring/rules.py` / `scoring/events.py`
     changes, not big ones, but they're judgment calls about the
     false-positive/false-negative tradeoff — confirm before implementing,
     don't just pick one.
3. Once 1 and 2 are settled, re-run the full 6-clip batch (with
   `--weapon-model` wired in, still no `--calib`) and check whether
   `video1.mp4`/`video3.mp4` finally produce a real, correct
   `weapon_at_window` flagged event — this would be the first
   confirmed-correct detection of the entire testing effort.
4. Decide whether `struggle_dwell_min_s` (1.5s) also needs lowering for
   fast real-world interactions, now that `close_dwell_time_s` is reporting
   accurate numbers (video3's real max was 0.44s) — same category of
   decision as weapon's duration threshold above, likely worth tuning both
   together rather than separately.
5. Build a calibration JSON matching the `hijackings` dataset's actual
   640x480 resolution (the existing `calib/example_driveway.json` is
   1920x1080 and won't work) if boxing-in and the convergence
   fleeing-signal are to be tested against this footage — currently
   neither has been exercised at all on real footage.
6. Confirm whether the exposed Roboflow API key (see "Security note" under
   "Kaggle testing round 2" above) was actually rotated — still unconfirmed,
   carried forward from session 3 unresolved.
7. Once individual events are confirmed firing correctly on real footage,
   build the cross-event correlator — full design already agreed with the
   user, documented in "Cross-event correlator" above, not yet implemented.
   Confirm the user still wants it before building (very likely yes, but
   confirm rather than assume after a gap in the conversation).
8. Once the above is solid, the previously-agreed testing roadmap
   continues: a person actively approaching/interacting with a vehicle →
   a busier benign clip (multiple people, false-positive check) → real
   struggle/aggressive-approach footage for further tuning.
9. Older still-open loose end, low priority: visually confirm the very
   first Kaggle run (`Normal.mp4`, session 2) actually showed correct
   detection in its `annotated.mp4` — never reviewed.
