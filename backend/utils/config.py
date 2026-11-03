from pathlib import Path
import os

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    def load_dotenv(*args, **kwargs):
        return False

ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / ".env", override=False)

APP_NAME = os.getenv("APP_NAME", "AI Forest Fire Monitoring")
APP_ENV = os.getenv("APP_ENV", "development")
APP_DEBUG = os.getenv("APP_DEBUG", "true").lower() == "true"
BACKEND_HOST = os.getenv("BACKEND_HOST", "0.0.0.0")
BACKEND_PORT = int(os.getenv("BACKEND_PORT", "8001"))
CORS_ORIGINS = [
    item.strip()
    for item in os.getenv(
        "CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    ).split(",")
    if item.strip()
]
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{ROOT_DIR / 'backend' / 'database' / 'forest_fire_ai.db'}")
DEMO_MODE = os.getenv("DEMO_MODE", "true").lower() == "true"
MODEL_THRESHOLD = float(os.getenv("MODEL_THRESHOLD", "0.75"))
ALERT_THRESHOLD = float(os.getenv("ALERT_THRESHOLD", "0.80"))
UNCERTAIN_THRESHOLD = float(os.getenv("UNCERTAIN_THRESHOLD", "0.45"))
MAX_UPLOAD_MB = float(os.getenv("MAX_UPLOAD_MB", "10"))
DEFAULT_LATITUDE = float(os.getenv("DEFAULT_LATITUDE", "12.9716"))
DEFAULT_LONGITUDE = float(os.getenv("DEFAULT_LONGITUDE", "77.5946"))
DATASET_DIR = Path(os.getenv("DATASET_DIR", str(ROOT_DIR / "dataset")))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", str(ROOT_DIR / "outputs")))


def ensure_directories() -> None:
    for path in [DATASET_DIR, OUTPUT_DIR, OUTPUT_DIR / "predictions", OUTPUT_DIR / "alerts", OUTPUT_DIR / "graphs", OUTPUT_DIR / "models"]:
        path.mkdir(parents=True, exist_ok=True)
