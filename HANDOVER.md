# Handover: driveway-guard (mashtronicsAI)

Written 2026-08-29, replacing the previous handover (2026-08-27) after a
session investigating *why* the retrained weapon checkpoint keyed off
false detections (car roof, side mirror) instead of real guns. **Root
cause found and confirmed. A partial fix was tried and made things worse
in a different way. Still not resolved — do not treat any checkpoint as
validated.** Everything from before this session is still committed on
`main` and recoverable via `git log` / `git show <hash>:<path>` — this
document reflects only the current state, per this repo's standing
practice of replacing the handover on a direction change rather than
appending another session section.

## tl;dr — where things stand right now

**Root cause of the false positives, confirmed with data, not just visual
suspicion**: the *original baseline* dataset
(`weapon-detection-cctv-v3-dataset`, called `cctv_v3` in code/configs) —
trusted all along as "the validated one" — was never actually spot-checked
against its real images. This session opened its real training images and
found a large fraction are **not CCTV footage at all**: gun-shop photos,
product/gallery shots, stock photography (some iStock-watermarked), guns
filling most of the frame in bright, close-up conditions. Measured on the
real label files: `cctv_v3`'s "weapon" boxes average **20.6% of the
frame** (median 4.17%), vs. the properly-CCTV-style `gun_cctv` source's
**0.16%** (median 0.10%) — a ~40x scale mismatch for boxes meant to be the
same object class. That mismatch, not the Part-1 retrain itself, is the
most likely cause of the model learning "large/reflective/dark region" as
a proxy for "weapon" instead of an actual gun shape.

**Consequence**: the pre-retrain baseline checkpoint (previously assumed
to be the safe fallback) was trained on this same contaminated data and
was **never itself visually spot-checked** either — don't trust it without
doing that first. Neither of the two fallback options from the previous
handover (revert to baseline, or just raise the confidence threshold) are
safe as-is; see "Why the simple fallbacks don't work" below.

**Two corrective retrains were tried this session, both rejected:**
1. Dropping `cctv_v3`'s weapon class entirely, training weapon-detection
   purely on `gun_cctv` (1,741 boxes) — recall collapsed to near-zero on
   real footage (max confidence 0.121 across an entire clip that
   previously had 0.72-confidence hits). `cctv_v3` was contributing real
   signal despite the contamination; an all-or-nothing cut was too blunt.
2. Keeping both sources but capping `cctv_v3`'s box size to ≤2% of frame
   area (new `max_area_norm` merge option, see below) — kept 1,762 of
   `cctv_v3`'s 4,278 boxes, shrunk its median box area from 4.17% to
   0.63% (much closer to `gun_cctv`'s 0.10%). This **did suppress** the
   specific car-roof/mirror false positives from the first retrain (their
   confidence dropped from 0.72/0.67 down to ~0.14), but real-footage
   testing found new problems instead — see "Attempt 2 results" below.
   **Not adopted.**

**User's decision at end of session**: try training with a third,
larger candidate dataset the user found on Kaggle
(`Weapon_Detection_for_Yolo`, ~24,000 images, single unified `Weapon`
class, includes ~1,544 explicit negative/background images) — see "New
candidate dataset" below for what's known and what's still unverified
about it. **This has not been downloaded, vetted, or trained on yet** —
that's the next session's starting point, in a new chat.

## What this project is

A production-track computer vision system that watches a **fixed driveway
security camera** (target hardware: Dahua CCTV, RTSP-capable — not yet
wired up) and flags candidate carjacking/hijacking events for human
review. Five event types were designed collaboratively with the user:

1. **Struggle / aggressive contact** near a vehicle (hijacking-style)
2. **Boxing-in** — a second vehicle blocking the resident's car's exit path
3. **Weapon at window** — a gun visible at a vehicle window — **the only
   one implemented, this phase's focus. Firearms only, deliberately not
   knives** (see "New candidate dataset" below for why this matters).
4. **Sprint approach** — someone closing on a stationary vehicle at
   running speed
5. **Multi-directional convergence** — 2+ people approaching a vehicle
   from different angles simultaneously

Types 1, 2, 4, 5 are designed but retired pending this incremental
rebuild; recoverable from git history (`git show d647e3f:<path>`), not
touched this session.

## The investigation, in order

### 1. Reviewed the exported snapshots from both test clips

`snapshots_video1/` and `snapshots_video3/` (untracked, still in the repo
root) hold annotated hit exports from the Part-1 retrained checkpoint
(`cctv_v3` weapon boxes unfiltered + `gun_cctv`'s `Handgun`/`Short_rifle`).
Visual review of all 8 hits found a clear pattern:
- `video1.mp4`'s 4 hits were **inconsistent, clearly wrong shapes**: a
  wide box on the car's roofline reflection (conf 0.72 — higher than
  several of the "real-looking" hits below), a small box on the car's
  mirror (0.67), a box engulfing the person's entire dark silhouette
  (0.57), a box around a dark pant-leg (0.70).
- `video3.mp4`'s 4 hits (conf 0.51–0.77, exported previously but never
  actually reviewed until this session) were **consistent**: a small,
  tall box at the same chest/waist position on the same person across all
  4 frames as they walked.

**Key finding: confidence alone can't separate real from fake.** The
car-roof false positive (0.72) scored higher than 3 of `video3.mp4`'s
plausible real hits (0.51, 0.56, 0.56). Raising
`weapon_confidence_threshold` to filter out the car-roof hit would also
filter out most of the apparently-genuine ones. This rules out "just raise
the threshold" as a real fix.

### 2. Built a ground-truth label inspector

**`scripts/inspect_dataset_labels.py`** (new, committed at `0e0025d`):
reads a YOLO `data.yaml`, pulls every ground-truth box for a given class,
and reports box-shape stats (width/height/aspect-ratio/area, all
normalized). Since `merge_yolo_datasets.py` prefixes every copied filename
with its source name, this script can split stats **by source** just from
filenames in the *merged* dataset (`--group-by-prefix`) — no need to keep
the original per-source exports around separately. It also exports a
random sample and the most-elongated boxes per group as annotated crops,
for eyeballing ground truth directly rather than trusting page
descriptions.

Run against the Part-1 merged dataset's `train` split, class `weapon`:

| | `cctv_v3` (n=4278) | `gun_cctv` (n=1741) |
|---|---|---|
| median box area | 4.17% of frame | 0.10% of frame |
| mean box area | **20.6%** of frame | 0.16% of frame |
| median aspect ratio | 1.45 | 2.57 |

The random-sample crops from the `cctv_v3` group confirmed why: filenames
and image content include `Gun-shop-Ticker`, `buying-a-shotgun-at-...`,
`Hello_MG1648042266099`, `gallery233`, iStock-watermarked images, soldiers
aiming rifles outdoors, a rifle laid on a table — professional/stock gun
photography, not surveillance footage. `gun_cctv`'s crops, by contrast,
genuinely are elevated-angle hallway/retail security-camera frames with a
small distant box, matching what Part 1's original vetting claimed for
it. `cctv_v3`'s `most_elongated` group also showed a secondary, smaller
issue even within its legitimately-CCTV-style images: some boxes are
stretched around a person's leg with no gun visible in them at all — loose
ground truth even in the non-contaminated subset.

### 3. Added a per-source box-area filter to the merge script

**`scripts/merge_yolo_datasets.py`** (updated, committed at `f098ed2`):
new optional `max_area_norm` field on a source's config, e.g.
`{"max_area_norm": {"weapon": 0.02}}` — drops any box remapped to that
target class whose normalized area exceeds the cap, without dropping the
source's contribution to the class entirely. Existing behavior is
unchanged when the field is omitted. `remap_label_line()` takes an
optional `max_area_norm: dict[int, float]` (keyed by target class id).

### 4. Two corrective retrains, both on Kaggle (GPU T4), both rejected

**Attempt 1 — drop `cctv_v3`'s weapon class entirely** (`weapon: null` for
that source, `gun_cctv`'s `Handgun`/`Short_rifle → weapon` as the sole
source). Aggregate metrics: weapon precision 0.746 / recall 0.494 / mAP50
0.599 — worse than both prior checkpoints, which is expected (the earlier
"better" numbers were inflated by validating on the same easy stock-photo
domain the training data came from). Real-footage test on `video1.mp4`
(`scripts/inspect_weapon_hits.py --conf 0.1`): **max confidence 0.121
across the entire clip** — the model essentially can't detect the weapon
at all anymore. `gun_cctv` alone (1,741 boxes) isn't enough signal by
itself. **Rejected** — too blunt.

**Attempt 2 — `max_area_norm: {"weapon": 0.02}` on `cctv_v3`, `gun_cctv`
unchanged.** Merged dataset: `cctv_v3` kept 1,762/4,278 boxes (41%),
median area down to 0.63% (mean not re-measured after filtering, but the
worst outliers are gone by construction). Trained; **metrics.json from
this run was never captured/pasted in this session** — only real-footage
behavior was checked, on three clips:

- **`video1.mp4`**: two clusters clear the 0.5 threshold — t≈4.96–5.06s
  (peak 0.611) and t≈6.08–6.12s (peak 0.562) — but longest continuous run
  is 0.10s, just under the 0.15s `weapon_min_duration_s` gate, so no event
  fired. Visually, the box lands at the person's **ankle/foot**, against
  the pavement — not a plausible gun position. Likely a *different* shape
  shortcut (dark leg against light pavement) rather than a genuine
  detection, though not conclusively confirmed either way. **The old
  car-roof (0.72) and mirror (0.67) hits are gone / suppressed to ~0.14**
  — that part of the fix worked.
  - Note: this run's `video1.mp4` had a different resolution/frame count
    (816×448, 595 frames) than the file used earlier in the session
    (832×464, 828 frames) despite being the "same" filename
    (`/kaggle/input/datasets/vhulendamashamba/videos/video1.mp4`). Not
    resolved — worth confirming next session whether the Kaggle input
    dataset attachment changed, or the file itself did, before trusting
    frame-number-based comparisons across runs.
- **`video3.mp4`**: max confidence dropped hard, from the earlier
  checkpoint's 0.77 peak down to **0.224**, in roughly the same t≈18.2–
  18.8s window that looked genuinely plausible before. Real recall loss
  on footage that previously looked like the project's best evidence of a
  working detector.
- **`Normal.mp4`** (576×1024, portrait, 30fps, 340 frames — confirmed by
  the user this is genuinely the no-weapon control clip): 3 consecutive
  frames clear the 0.5 threshold, **peak confidence 0.693** — higher than
  most of what we've confirmed as genuine elsewhere — around t≈2.53–2.60s.
  No event fired only because the run was 0.07s, under the 0.15s gate.
  **This is too close for comfort on footage that's supposed to have zero
  weapon presence.**
- Snapshot exports were requested for both `video3.mp4`'s weaker hit and
  `Normal.mp4`'s 0.693 spike (`export_weapon_snapshots.py --min-conf
  0.15`/`0.4` respectively, output dirs
  `/kaggle/working/snapshots_v3_video3` and `snapshots_v3_normal`) **but
  the images were never reviewed before the session ended** — open item,
  do this first next session if this checkpoint direction is revisited.

**Conclusion**: capping `cctv_v3`'s box size suppressed the two specific
false positives it was targeted at, but didn't fix the underlying
problem — it just surfaced a different one (possible new shortcut on
`video1.mp4`'s ankle/leg region, a concerning near-miss on the `Normal.mp4`
control, and a real recall loss on `video3.mp4`). **Not adopted as the
checkpoint in use.**

### Why the simple fallbacks don't work

- **Revert to pre-retrain baseline**: also trained on the contaminated
  `cctv_v3` weapon class (that's the *only* weapon source the baseline
  ever had), and its "solid" 0.833 validation precision was measured on a
  held-out split of the same stock-photo-heavy data — never tested
  against anything resembling real driveway footage, and never visually
  spot-checked. Don't assume it's clean.
- **Just raise `weapon_confidence_threshold`**: ruled out directly — the
  car-roof false positive (0.72) scores higher than several of the
  hits we've judged most likely to be genuine (0.51–0.56 on `video3.mp4`).
  A confidence-only cutoff can't separate them.

## New candidate dataset — found by user, not yet vetted or used

Kaggle dataset `Weapon_Detection_for_Yolo` (exact URL/owner slug not yet
captured — get this from the user next session before trying to download
it). Per its listing page:
- ~24,000 images (18,186 train / 4,546 val / ~1,200 test).
- **Single unified `Weapon` class** — pistols, knives, handguns, etc. all
  merged into one label, no way to separate them back out. **This is a
  structural mismatch with the project**: `weapon_at_window` is
  deliberately firearms-only (Part 1 explicitly dropped `Knife → null`).
  Using this dataset's class as-is would reintroduce knife detections
  with no way to filter them back out after the fact.
- Merged from "Multiple Public Weapon Datasets" (Roboflow/GitHub, per its
  own description) plus **Sohas Weapon Detection** (converted from Pascal
  VOC), plus **~1,544 explicit negative/background images** (phones,
  laptops, empty rooms — empty label files). This is the most
  differentiated thing it offers: **neither `cctv_v3` nor `gun_cctv` has
  any labeled non-weapon images at all** — every image in both current
  sources has a gun in it somewhere. The complete absence of hard/soft
  negatives in current training data is a plausible contributor to the
  shortcut-learning problem this whole session has been chasing (a model
  never shown "this dark/reflective/elongated thing is NOT a weapon" has
  no reason not to fire on one).
- Two preview thumbnails were manually reviewed (from the Kaggle listing
  page, i.e. **cherry-picked by the uploader, not a random sample** — the
  same caveat that caught `cctv_v3` out): one eye-level indoor demo photo
  (reasonable gun-to-frame size, not stock-photo-huge, but not
  CCTV-angle either), one genuinely convincing elevated
  fisheye-lens hallway shot with a timestamp overlay
  (`2019-05-16 21:17:09`) — closer to this project's actual camera
  domain. Encouraging, but **two hand-picked preview images are not
  equivalent to actually pulling real random samples and box-size stats
  from the label files**, which is what `inspect_dataset_labels.py` was
  built to do and hasn't been run against this dataset yet.

**User's decision: proceed to train using this dataset next session.**
Before spending GPU time on it:
1. Get the actual Kaggle dataset reference from the user (URL or
   `kaggle datasets download` slug).
2. Download it and run `scripts/inspect_dataset_labels.py` directly
   against its own `data.yaml` (works standalone, doesn't require merging
   first) to get real box-size stats and random-sample crops, the same
   vetting every other candidate source has now gotten.
3. Decide the integration approach given the single-class/knife problem —
   options on the table but not decided: (a) use only its ~1,544 negative
   images as background examples merged into the existing `cctv_v3` +
   `gun_cctv` training set, leaving the weapon-class data untouched, which
   directly targets the negatives gap without inheriting the knife-mixing
   problem; (b) take its full weapon class anyway and accept the
   knife/gun conflation; (c) something in between (e.g. filter by
   filename/source if the merged dataset's provenance is distinguishable
   per-image, similar to how `cctv_v3`/`gun_cctv` could be told apart by
   filename prefix — untested whether this dataset preserves that kind of
   per-source traceability).
4. Whatever gets trained, run the **same real-footage validation this
   session established**: `inspect_weapon_hits.py --conf 0.1` on all
   three of `video1.mp4`, `video3.mp4`, `Normal.mp4` first (cheap), then
   `export_weapon_snapshots.py` + actual visual review of the crops for
   any clip with hits — do not trust event-firing or aggregate metrics
   alone. This session's whole finding was that both of those can look
   fine while the underlying detection is wrong.

## What's reused as-is / retired

Unchanged from the 2026-08-27 handover — see `git show
c209787:HANDOVER.md` for the full "What's reused as-is" / "What's
retired" breakdown if needed. Nothing in `src/driveway_guard/` changed
this session; only `scripts/inspect_dataset_labels.py` (new) and
`scripts/merge_yolo_datasets.py` (extended) changed.

## Standing instructions / preferences

- **Never add Claude as a co-author/contributor on commits in this repo.**
- Git identity: name `smoke`, email `vhulendamashamba4@gmail.com`.
- User prefers testing on **Kaggle (free GPU)** over local CPU runs.
- **Actually vet any sourced dataset/checkpoint** — this session is the
  strongest example yet of why: `cctv_v3` was trusted for two full
  sessions purely because of its name and the fact that it was already in
  use, and turned out to be the actual root cause once someone opened the
  real files. Real class list, real image count, license, real sample
  thumbnails (not page descriptions), and now also: **real box-size
  stats via `inspect_dataset_labels.py`** — a dataset can look right in
  a hand-picked preview and still be wrong in aggregate.
- Roboflow Universe pages need a browser-like User-Agent to fetch via
  curl/WebFetch (plain requests get HTTP 403) — but note their pages are
  client-rendered (no useful data in the raw HTML beyond the app shell);
  this doesn't help pull real stats, only confirms the page loads.
- **When a chunk of work fundamentally changes direction, replace the
  handover rather than appending another session section to an
  already-long one** (this document is itself an example).
- New script `scripts/inspect_dataset_labels.py` needs the real dataset
  to run — no dataset is cached locally (`data/input`, `data/samples` are
  both empty), same "Kaggle working doesn't persist" reason as everything
  else. It runs fine as a plain script via `!python scripts/...` from a
  subprocess-spawned Kaggle cell without needing a kernel restart, even
  right after `pip install -q -e .` in the same session (the "already-
  running kernel doesn't pick up new installs" gotcha only affects direct
  `import driveway_guard...` in a notebook cell, not `!python` subprocess
  calls).

## Environment notes

- Machine: Windows 11, repo at
  `c:\Users\vhule\OneDrive\Desktop\Projects\mashtronicsAI`.
- Local venv at `.venv/` (`.venv\Scripts\python.exe`). **No NVIDIA GPU
  locally** — CPU-only inference/no training locally; Kaggle is the
  validated workflow for anything involving the actual datasets or GPU
  training.
- Untracked in the repo root right now, all expected/known, none of them
  need attention: `credentials.txt` (Roboflow API key, plaintext, user
  has explicitly decided to keep using it rather than rotate — never add
  to git), `output.txt`, `snapshots_video1/`, `snapshots_video3/` (the
  Part-1-checkpoint exports reviewed in this session's step 1, kept for
  reference).

## Kaggle workflow

1. **New notebook** → Accelerator: GPU T4 (only needed for actual
   training steps; dataset-inspection-only steps run fine on CPU) →
   Internet: On.
2. `!git clone https://github.com/SMOKE484/mashtronicsAI.git && cd
   mashtronicsAI && pip install -q -e .` — if re-running in an
   already-cloned session, `%cd /kaggle/working/mashtronicsAI && !git
   pull` instead of re-cloning. **Watch for accidental double-nesting**:
   if `%cd mashtronicsAI` is run while already inside a `mashtronicsAI`
   directory (e.g. re-running Cell 1 in a session that didn't actually
   restart), `git clone` creates a nested `mashtronicsAI/mashtronicsAI`
   copy. Not fatal — the freshly-cloned nested copy is still complete and
   fine to use, just `%cd` into it and continue; no need to delete
   anything or re-clone.
3. For video: attach via "Add Input" → Dataset, then confirm the real
   mount path with `find /kaggle/input/ -iname "*.mp4"` (sidebar's
   displayed path has been wrong before). This session's actual path:
   `/kaggle/input/datasets/vhulendamashamba/videos/<name>.mp4`.
4. **`/kaggle/working` does not survive across separate Kaggle
   sessions** — save any checkpoint worth keeping as a Kaggle Dataset or
   download it immediately after training, before doing anything else.
5. `!pip install -q -e .` doesn't get picked up by an *already-running*
   kernel for direct `from driveway_guard... import ...` — only by new
   subprocess-spawned `python` processes (`!python -m ...` or
   `subprocess.run([...])`). Standalone scripts run via `!python
   scripts/...` are unaffected by this and can be run right after
   installing, same cell block or the next one.
6. **Downloading a zipped output folder from an active session can
   truncate/corrupt in transit** even when the zip is valid on Kaggle's
   own filesystem (verify with `zipfile.ZipFile(...).testzip()` +
   `ls -la` before assuming the zip itself is the problem). Easiest
   workaround: skip downloading entirely and render images inline in the
   notebook with `matplotlib`/`cv2` (see any of this session's "Cell 5"/
   inline-viewer snippets), then screenshot the notebook output instead.
