import csv
import json
from dataclasses import asdict
from pathlib import Path

from driveway_guard.scoring.events import FlaggedEvent


def write_json(events: list[FlaggedEvent], path: Path) -> None:
    Path(path).write_text(json.dumps([asdict(e) for e in events], indent=2))


def write_csv(events: list[FlaggedEvent], path: Path) -> None:
    fieldnames = [
        "event_type",
        "start_timestamp_s",
        "end_timestamp_s",
        "start_frame_idx",
        "end_frame_idx",
        "peak_score",
        "track_ids",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for event in events:
            row = asdict(event)
            row["track_ids"] = ",".join(str(t) for t in row["track_ids"])
            writer.writerow(row)
