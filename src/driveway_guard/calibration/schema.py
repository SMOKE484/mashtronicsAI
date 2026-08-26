import json
from pathlib import Path

from pydantic import BaseModel, model_validator


class DrivewayZone(BaseModel):
    polygon: list[tuple[float, float]]


class EgressPath(BaseModel):
    exit_point: tuple[float, float]
    direction_vector: tuple[float, float]
    corridor_width_px: float
    corridor_length_px: float = 900.0


class ResidentVehicleHint(BaseModel):
    typical_start_zone_polygon: list[tuple[float, float]]


class CalibrationConfig(BaseModel):
    schema_version: int = 1
    source_name: str
    frame_width: int
    frame_height: int
    driveway_zone: DrivewayZone
    egress_path: EgressPath
    resident_vehicle_hint: ResidentVehicleHint | None = None
    notes: str | None = None

    @model_validator(mode="after")
    def _check_direction_vector(self) -> "CalibrationConfig":
        dx, dy = self.egress_path.direction_vector
        mag = (dx**2 + dy**2) ** 0.5
        if mag < 1e-6:
            raise ValueError("egress_path.direction_vector must not be the zero vector")
        return self

    @classmethod
    def load(
        cls, path: Path, expected_width: int | None = None, expected_height: int | None = None
    ) -> "CalibrationConfig":
        data = json.loads(Path(path).read_text())
        config = cls.model_validate(data)
        if expected_width is not None and config.frame_width != expected_width:
            raise ValueError(
                f"Calibration frame_width ({config.frame_width}) does not match "
                f"source video width ({expected_width}); recalibrate for this video."
            )
        if expected_height is not None and config.frame_height != expected_height:
            raise ValueError(
                f"Calibration frame_height ({config.frame_height}) does not match "
                f"source video height ({expected_height}); recalibrate for this video."
            )
        return config
