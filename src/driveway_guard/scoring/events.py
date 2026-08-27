from dataclasses import dataclass, field

# Debouncing is duration-based (elapsed seconds) rather than frame-count
# based, since frame rate and --frame-stride both change how many frames
# correspond to a given wall-clock duration.
_MIN_SAMPLES = 2


@dataclass
class FlaggedEvent:
    event_type: str
    start_timestamp_s: float
    end_timestamp_s: float
    start_frame_idx: int
    end_frame_idx: int
    peak_score: float
    track_ids: list[int] = field(default_factory=list)


@dataclass
class _EventState:
    consecutive_samples: int = 0
    start_frame_idx: int = 0
    start_timestamp_s: float = 0.0
    last_above_timestamp_s: float = 0.0
    peak_score: float = 0.0
    emitted: bool = False
    cooldown_until: float = -1.0


class EventAggregator:
    """Per (event_type, entity_key) debouncing: requires the score to stay
    at/above threshold for at least `min_duration_s` (and at least two
    samples, to avoid a single-frame timestamp jump satisfying the duration
    check) before emitting a FlaggedEvent, then enforces a cooldown before
    the same key can flag again.

    `max_gap_s` tolerates brief below-threshold dips inside an otherwise
    sustained detection without resetting the streak -- real per-frame
    confidence is noisy, and requiring literally every sample to clear
    threshold made a genuine ~1s detection with one dipped frame count as
    two separate sub-threshold runs (see HANDOVER.md's real-footage
    validation notes). A gap longer than max_gap_s still resets normally."""

    def __init__(self):
        self._states: dict[tuple, _EventState] = {}

    def update(
        self,
        event_type: str,
        key: tuple,
        score: float,
        threshold: float,
        min_duration_s: float,
        cooldown_s: float,
        frame_idx: int,
        timestamp_s: float,
        track_ids: list[int],
        max_gap_s: float = 0.0,
    ) -> FlaggedEvent | None:
        state_key = (event_type, key)
        state = self._states.get(state_key)

        if score < threshold:
            if state is not None and state.consecutive_samples > 0:
                gap = timestamp_s - state.last_above_timestamp_s
                if gap > max_gap_s:
                    state.consecutive_samples = 0
                    state.emitted = False
            return None

        if state is None:
            state = _EventState()
            self._states[state_key] = state

        if state.consecutive_samples == 0:
            state.start_frame_idx = frame_idx
            state.start_timestamp_s = timestamp_s
            state.peak_score = score
        else:
            state.peak_score = max(state.peak_score, score)
        state.consecutive_samples += 1
        state.last_above_timestamp_s = timestamp_s

        duration = timestamp_s - state.start_timestamp_s
        ready = (
            not state.emitted
            and state.consecutive_samples >= _MIN_SAMPLES
            and duration >= min_duration_s
            and timestamp_s >= state.cooldown_until
        )
        if ready:
            state.emitted = True
            state.cooldown_until = timestamp_s + cooldown_s
            return FlaggedEvent(
                event_type=event_type,
                start_timestamp_s=state.start_timestamp_s,
                end_timestamp_s=timestamp_s,
                start_frame_idx=state.start_frame_idx,
                end_frame_idx=frame_idx,
                peak_score=state.peak_score,
                track_ids=list(track_ids),
            )
        return None
