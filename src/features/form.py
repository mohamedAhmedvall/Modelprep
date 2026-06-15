"""Forme récente des équipes (moyennes glissantes sans fuite de données).

Pour chaque match on calcule, pour les 2 équipes, la moyenne des buts
marqués / encaissés sur leurs N derniers matchs joués AVANT ce match.
On renvoie aussi un instantané final {équipe: {gf, ga}} servant à la
prédiction de matchs futurs.
"""
from __future__ import annotations

from collections import defaultdict, deque
from typing import Deque, Dict, Tuple

import pandas as pd

WINDOW = 5
PRIOR_GF = 1.2  # buts marqués moyens a priori (équipe sans historique)
PRIOR_GA = 1.2


def _avg(dq: Deque[int], prior: float) -> float:
    return sum(dq) / len(dq) if dq else prior


def compute_form(
    matches: pd.DataFrame, window: int = WINDOW
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    gf: Dict[str, Deque[int]] = defaultdict(lambda: deque(maxlen=window))
    ga: Dict[str, Deque[int]] = defaultdict(lambda: deque(maxlen=window))

    h_gf, h_ga, a_gf, a_ga = [], [], [], []

    for r in matches.itertuples(index=False):
        h_gf.append(_avg(gf[r.home_team], PRIOR_GF))
        h_ga.append(_avg(ga[r.home_team], PRIOR_GA))
        a_gf.append(_avg(gf[r.away_team], PRIOR_GF))
        a_ga.append(_avg(ga[r.away_team], PRIOR_GA))

        hs, as_ = int(r.home_score), int(r.away_score)
        gf[r.home_team].append(hs)
        ga[r.home_team].append(as_)
        gf[r.away_team].append(as_)
        ga[r.away_team].append(hs)

    out = matches.copy()
    out["home_gf5"] = h_gf
    out["home_ga5"] = h_ga
    out["away_gf5"] = a_gf
    out["away_ga5"] = a_ga

    snapshot = {
        t: {"gf": _avg(gf[t], PRIOR_GF), "ga": _avg(ga[t], PRIOR_GA)}
        for t in set(gf) | set(ga)
    }
    return out, snapshot
