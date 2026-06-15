"""Configuration centrale du projet."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
ARTIFACTS = ROOT / "artifacts"
WEB_DIR = ROOT / "web"

RAW_RESULTS = DATA_RAW / "results.csv"
PROCESSED_MATCHES = DATA_PROCESSED / "matches.csv"
MODEL_PATH = ARTIFACTS / "model.pkl"
BACKTEST_REPORT = ARTIFACTS / "backtest_report.json"

# Source de données ouverte (matchs internationaux 1872 -> aujourd'hui, licence MIT)
RESULTS_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
)

# Importance des compétitions (utilisée par l'Elo et comme feature).
# Plus la valeur est élevée, plus le match "compte".
TOURNAMENT_WEIGHTS = {
    "FIFA World Cup": 60,
    "FIFA World Cup qualification": 40,
    "UEFA Euro": 50,
    "UEFA Euro qualification": 35,
    "Copa América": 50,
    "African Cup of Nations": 45,
    "AFC Asian Cup": 45,
    "Gold Cup": 40,
    "Confederations Cup": 45,
    "UEFA Nations League": 35,
    "Friendly": 20,
}
DEFAULT_TOURNAMENT_WEIGHT = 30

# Fenêtre d'entraînement du modèle de buts (années récentes).
TRAIN_YEARS = 12
# Demi-vie de la pondération temporelle (Dixon-Coles), en jours.
HALF_LIFE_DAYS = 730  # ~2 ans
# Nombre maxi de buts considéré dans la matrice de score.
MAX_GOALS = 10
# Poids du modèle Dixon-Coles dans le mélange final (le reste -> XGBoost).
DC_BLEND_WEIGHT = 0.6


def tournament_weight(name: str) -> float:
    if not isinstance(name, str):
        return DEFAULT_TOURNAMENT_WEIGHT
    return TOURNAMENT_WEIGHTS.get(name, DEFAULT_TOURNAMENT_WEIGHT)


def ensure_dirs() -> None:
    for d in (DATA_RAW, DATA_PROCESSED, ARTIFACTS):
        d.mkdir(parents=True, exist_ok=True)
