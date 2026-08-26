"""Request contracts shared by the production and detachable Demo APIs."""
from __future__ import annotations

from pydantic import BaseModel

import config


class LoginBody(BaseModel):
    username: str
    password: str


class ReviewBody(BaseModel):
    decision: str


class ModelNameBody(BaseModel):
    name: str


class OperatingStateOverrideBody(BaseModel):
    state: str
    expires_minutes: int = 480
    note: str = ""


class ThresholdBody(BaseModel):
    MAINTENANCE_PROB_URGENT: float
    MAINTENANCE_PROB_PLAN: float
    FAILURE_HEALTH_THRESHOLD: float
    MAINTENANCE_HEALTH_INSPECT: float
    MAINTENANCE_HORIZON_DAYS: float
    MAINTENANCE_URGENT_HORIZON_DAYS: float = config.MAINTENANCE_URGENT_HORIZON_DAYS
    TREND_MIN_POINTS: int
    TREND_SLOPE_Z_THRESHOLD: float
    TREND_SETTLE_TICKS: int
    MAINTENANCE_WARN_CONFIRM_MINUTES: float = config.MAINTENANCE_WARN_CONFIRM_MINUTES
    MAINTENANCE_CRITICAL_CONFIRM_MINUTES: float = config.MAINTENANCE_CRITICAL_CONFIRM_MINUTES
    MAINTENANCE_RECOVERY_MINUTES: float = config.MAINTENANCE_RECOVERY_MINUTES
    OPERATING_STATE_STOP_CONFIRM_TICKS: int
    OPERATING_STATE_START_CONFIRM_TICKS: int
    SOURCE_STALE_SECONDS: int
    HEALTH_SENSITIVITY_STD: float
    KALMAN_INIT_SAMPLES: int
    TREND_LOOKBACK_MINUTES: int
    WORKER_POLL_SECONDS: int
    NEAR_MISS_TREND_WINDOW_HOURS: float


class EnvBody(BaseModel):
    values: dict[str, str]


class UserCreate(BaseModel):
    username: str
    password: str
    role: str = "viewer"


class SimulationCaseBody(BaseModel):
    description: str
    start: str
    end: str
    machine_id: str | None = None
    expected_status: str


class SimulationRunBody(BaseModel):
    name: str
    cases: list[SimulationCaseBody]
    model_version: str | None = None


class BackfillBody(BaseModel):
    start: str
    end: str
    mode: str = "repredict"
    machine_id: str | None = None


class TrainingConfigBody(BaseModel):
    RETRAIN_BATCH_SIZE: int
    RETRAIN_TIME_CAP_DAYS: int
    REFERENCE_WINDOW_MONTHS: int
    REFERENCE_DEDUP_WINDOW_HOURS: float
    REFERENCE_COSINE_SIMILARITY: float
    RETRAIN_CHECK_INTERVAL_MINUTES: int
    RETRAIN_RETRY_COOLDOWN_HOURS: float
    RETRAIN_MAX_FP_RATE_INCREASE: float
    AUTO_RETRAIN_ENABLED: bool = config.AUTO_RETRAIN_ENABLED
