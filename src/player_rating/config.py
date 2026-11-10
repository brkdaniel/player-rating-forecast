"""Project-wide paths and constants."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "fifa_world_cup_2026_player_performance.csv"
DATA_SHA256 = "4ed9cc4b0e05b38f839b8411c1926974fc7d2477ac30b1948f5c0ea5b0d32632"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_DIR = REPORTS_DIR / "metrics"
MODELS_DIR = PROJECT_ROOT / "models"

SEED = 42
TARGET = "player_rating"
MIN_MINUTES = 30  # the model is conditional on the player taking part for at least 30 minutes
FORM_WINDOW = 5  # number of previous match dates behind the v1 "last 5" features

# Chronological split over match dates: first 70% train, next 15% validation, last 15% test.
TRAIN_FRACTION = 0.70
TRAIN_VAL_FRACTION = 0.85
