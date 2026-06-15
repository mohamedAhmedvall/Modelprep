"""Modèle ML (gradient boosting) d'estimation des buts attendus.

Complète le modèle statistique Dixon-Coles avec une approche purement
"machine learning" : deux régresseurs (buts domicile / buts extérieur)
entraînés sur des features riches (Elo, forme récente, terrain neutre,
importance du match). Les intensités prédites sont ensuite mélangées avec
celles de Dixon-Coles.

Utilise XGBoost si disponible, sinon repli automatique sur le
GradientBoostingRegressor de scikit-learn (toujours fonctionnel).
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from src.config import tournament_weight

FEATURES: List[str] = [
    "home_elo_pre", "away_elo_pre", "elo_diff_pre",
    "home_gf5", "home_ga5", "away_gf5", "away_ga5",
    "neutral_int", "tw",
]


def _make_regressor():
    try:
        from xgboost import XGBRegressor

        return XGBRegressor(
            n_estimators=400, max_depth=4, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
            objective="count:poisson", n_jobs=-1, random_state=42,
        ), "xgboost"
    except Exception:
        from sklearn.ensemble import GradientBoostingRegressor

        return GradientBoostingRegressor(
            n_estimators=300, max_depth=3, learning_rate=0.03,
            subsample=0.8, random_state=42,
        ), "sklearn-gbr"


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["neutral_int"] = out["neutral"].astype(int)
    out["tw"] = out["tournament"].map(tournament_weight).astype(float)
    return out


class XGBGoals:
    def __init__(self):
        self.model_home = None
        self.model_away = None
        self.backend = None

    def fit(self, df: pd.DataFrame) -> "XGBGoals":
        data = build_features(df).dropna(subset=FEATURES)
        x = data[FEATURES].values
        self.model_home, self.backend = _make_regressor()
        self.model_away, _ = _make_regressor()
        self.model_home.fit(x, data["home_score"].astype(float).values)
        self.model_away.fit(x, data["away_score"].astype(float).values)
        return self

    def predict_lambdas(self, feat: dict):
        row = np.array([[feat[f] for f in FEATURES]], dtype=float)
        lam = float(self.model_home.predict(row)[0])
        mu = float(self.model_away.predict(row)[0])
        return max(0.05, min(lam, 8.0)), max(0.05, min(mu, 8.0))
