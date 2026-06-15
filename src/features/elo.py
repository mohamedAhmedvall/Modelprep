"""Système de notation Elo pour sélections nationales.

Implémente la variante "World Football Elo" :
- avantage du terrain (+100) sauf si match sur terrain neutre,
- facteur K dépendant de l'importance de la compétition,
- multiplicateur lié à l'écart de buts (un 4-0 fait bouger plus qu'un 1-0).

La fonction principale `compute_elo` renvoie :
- le dataframe enrichi des Elo d'avant-match (sans fuite de données),
- l'état final {équipe: rating} après le dernier match.
"""
from __future__ import annotations

from typing import Dict, Tuple

import pandas as pd

from src.config import tournament_weight

BASE_RATING = 1500.0
HOME_ADVANTAGE = 100.0


def _k_factor(tournament: str) -> float:
    """K proportionnel à l'importance du match (échelle ~ 20 -> 60)."""
    return float(tournament_weight(tournament))


def _goal_multiplier(goal_diff: int) -> float:
    g = abs(goal_diff)
    if g <= 1:
        return 1.0
    if g == 2:
        return 1.5
    return (11.0 + g) / 8.0


def _expected(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10 ** (-(rating_a - rating_b) / 400.0))


def compute_elo(matches: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
    ratings: Dict[str, float] = {}
    home_pre, away_pre = [], []

    for r in matches.itertuples(index=False):
        ra = ratings.get(r.home_team, BASE_RATING)
        rb = ratings.get(r.away_team, BASE_RATING)
        home_pre.append(ra)
        away_pre.append(rb)

        adv = 0.0 if r.neutral else HOME_ADVANTAGE
        exp_home = _expected(ra + adv, rb)
        exp_away = 1.0 - exp_home

        if r.home_score > r.away_score:
            score_home, score_away = 1.0, 0.0
        elif r.home_score < r.away_score:
            score_home, score_away = 0.0, 1.0
        else:
            score_home, score_away = 0.5, 0.5

        k = _k_factor(r.tournament)
        g = _goal_multiplier(int(r.home_score) - int(r.away_score))
        ratings[r.home_team] = ra + k * g * (score_home - exp_home)
        ratings[r.away_team] = rb + k * g * (score_away - exp_away)

    out = matches.copy()
    out["home_elo_pre"] = home_pre
    out["away_elo_pre"] = away_pre
    out["elo_diff_pre"] = out["home_elo_pre"] - out["away_elo_pre"]
    return out, ratings
