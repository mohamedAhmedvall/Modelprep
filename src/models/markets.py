"""Dérivation des marchés de paris à partir d'une matrice de score."""
from __future__ import annotations

from typing import Dict, List

import numpy as np


def markets_from_matrix(mat: np.ndarray, top_n: int = 5) -> Dict:
    n = mat.shape[0]
    idx = np.indices((n, n))
    home_g, away_g = idx[0], idx[1]

    p_home = float(mat[home_g > away_g].sum())
    p_draw = float(np.trace(mat))
    p_away = float(mat[home_g < away_g].sum())

    total = home_g + away_g
    p_over25 = float(mat[total >= 3].sum())
    p_btts = float(mat[(home_g >= 1) & (away_g >= 1)].sum())

    flat = mat.flatten()
    order = np.argsort(flat)[::-1][:top_n]
    top_scores: List[Dict] = []
    for k in order:
        i, j = divmod(int(k), n)
        top_scores.append({"score": f"{i}-{j}",
                           "home": i, "away": j,
                           "prob": round(float(flat[k]), 4)})

    best = top_scores[0]
    return {
        "predicted_score": best["score"],
        "predicted_prob": best["prob"],
        "top_scores": top_scores,
        "outcome_1x2": {
            "home": round(p_home, 4),
            "draw": round(p_draw, 4),
            "away": round(p_away, 4),
        },
        "over_under_2_5": {
            "over": round(p_over25, 4),
            "under": round(1 - p_over25, 4),
        },
        "btts": {"yes": round(p_btts, 4), "no": round(1 - p_btts, 4)},
        "fair_odds_1x2": {
            "home": round(1 / p_home, 2) if p_home > 0 else None,
            "draw": round(1 / p_draw, 2) if p_draw > 0 else None,
            "away": round(1 / p_away, 2) if p_away > 0 else None,
        },
    }
