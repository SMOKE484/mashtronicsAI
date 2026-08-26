from abc import ABC, abstractmethod

from driveway_guard.features.schema import (
    BlockingObservation,
    FrameFeatureVector,
    VehicleConvergenceFeatureVector,
)
from driveway_guard.scoring.events import FlaggedEvent


class RiskScorer(ABC):
    """Turns extracted features into flagged events. `RuleBasedScorer` is
    the v1 implementation; a learned model can implement this same
    interface later without touching pipeline wiring."""

    @abstractmethod
    def process_frame(
        self,
        frame_idx: int,
        timestamp_s: float,
        pair_records: list[FrameFeatureVector],
        blocking_observations: list[BlockingObservation],
        convergence_records: list[VehicleConvergenceFeatureVector],
    ) -> list[FlaggedEvent]: ...
