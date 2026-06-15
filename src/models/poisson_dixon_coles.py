"""Modèle de buts Dixon-Coles (Poisson bivarié corrigé).

C'est le cœur de la prédiction de score exact. On modélise le nombre de
buts de chaque équipe par une loi de Poisson dont l'intensité dépend de
la force d'attaque de l'équipe, de la faiblesse défensive de l'adversaire
et de l'avantage du terrain. La correction de Dixon-Coles (paramètre rho)
ajuste la dépendance entre les deux scores pour les petits scores
(0-0, 1-0, 0-1, 1-1), historiquement mal capturés par un Poisson simple.

Les forces d'attaque/défense sont estimées par une régression de Poisson
(GLM statsmodels) pondérée dans le temps (demi-vie configurable), façon
Dixon-Coles : un match récent pèse plus qu'un match ancien.
"""
from __future__ import annotations

import math
from typing import Dict

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import poisson

from src.config import HALF_LIFE_DAYS, MAX_GOALS


def build_matrix(lam: float, mu: float, rho: float,
                 max_goals: int = MAX_GOALS) -> np.ndarray:
    """Matrice des probabilités de score à partir d'intensités données."""
    n = max_goals + 1
    ph = poisson.pmf(np.arange(n), lam)
    pa = poisson.pmf(np.arange(n), mu)
    mat = np.outer(ph, pa)
    for i in (0, 1):
        for j in (0, 1):
            mat[i, j] *= _dc_tau(i, j, lam, mu, rho)
    mat = np.clip(mat, 0.0, None)
    mat /= mat.sum()
    return mat


def _dc_tau(i: int, j: int, lam: float, mu: float, rho: float) -> float:
    """Facteur de correction Dixon-Coles pour les petits scores."""
    if i == 0 and j == 0:
        return 1.0 - lam * mu * rho
    if i == 0 and j == 1:
        return 1.0 + lam * rho
    if i == 1 and j == 0:
        return 1.0 + mu * rho
    if i == 1 and j == 1:
        return 1.0 - rho
    return 1.0


class DixonColesModel:
    OTHER = "__OTHER__"

    def __init__(self, half_life_days: int = HALF_LIFE_DAYS,
                 max_goals: int = MAX_GOALS, min_matches: int = 8):
        self.half_life_days = half_life_days
        self.max_goals = max_goals
        self.min_matches = min_matches
        self.attack: Dict[str, float] = {}
        self.defense: Dict[str, float] = {}
        self.intercept: float = 0.0
        self.home_coef: float = 0.0
        self.rho: float = 0.0
        self.teams: set = set()
        self._known: set = set()

    # ------------------------------------------------------------------ fit
    def fit(self, matches: pd.DataFrame, ref_date=None) -> "DixonColesModel":
        import statsmodels.formula.api as smf

        if ref_date is None:
            ref_date = matches["date"].max()
        ref_date = pd.Timestamp(ref_date)

        # Construction du format "long" : 2 lignes par match.
        age_days = (ref_date - matches["date"]).dt.days.clip(lower=0)
        decay = 0.5 ** (age_days / self.half_life_days)

        # Regroupement des équipes trop rares pour éviter la séparation
        # parfaite (qui fait diverger la régression de Poisson).
        counts = (pd.concat([matches["home_team"], matches["away_team"]])
                  .value_counts())
        self._known = set(counts[counts >= self.min_matches].index)

        def lab(s):
            return s.where(s.isin(self._known), self.OTHER)

        home = pd.DataFrame({
            "team": lab(matches["home_team"]),
            "opponent": lab(matches["away_team"]),
            "goals": matches["home_score"].astype(int),
            "home": np.where(matches["neutral"], 0.0, 1.0),
            "w": decay.values,
        })
        away = pd.DataFrame({
            "team": lab(matches["away_team"]),
            "opponent": lab(matches["home_team"]),
            "goals": matches["away_score"].astype(int),
            "home": 0.0,
            "w": decay.values,
        })
        long = pd.concat([home, away], ignore_index=True)
        long = long[long["w"] > 1e-6]
        # Normalisation des poids (stabilise l'IRLS).
        long["w"] = long["w"] / long["w"].mean()

        self.teams = set(matches["home_team"]) | set(matches["away_team"])

        import statsmodels.api as sm

        model = smf.glm(
            "goals ~ C(team) + C(opponent) + home",
            data=long,
            family=sm.families.Poisson(),
            var_weights=long["w"].values,
        )
        res = model.fit(maxiter=100)

        labels = set(self._known) | {self.OTHER}
        self.attack = {t: 0.0 for t in labels}
        self.defense = {t: 0.0 for t in labels}
        for name, val in res.params.items():
            if name.startswith("C(team)[T."):
                self.attack[name[len("C(team)[T."):-1]] = float(val)
            elif name.startswith("C(opponent)[T."):
                self.defense[name[len("C(opponent)[T."):-1]] = float(val)
        self.intercept = float(res.params.get("Intercept", 0.0))
        self.home_coef = float(res.params.get("home", 0.0))

        self._fit_rho(matches, decay.values)
        return self

    def _fit_rho(self, matches: pd.DataFrame, weights: np.ndarray) -> None:
        """Estime rho en maximisant la vraisemblance pondérée (1D)."""
        lams, mus, hs, as_, ws = [], [], [], [], []
        for r, w in zip(matches.itertuples(index=False), weights):
            lam, mu = self.predict_lambdas(r.home_team, r.away_team, r.neutral)
            lams.append(lam); mus.append(mu)
            hs.append(int(r.home_score)); as_.append(int(r.away_score))
            ws.append(w)
        lams = np.array(lams); mus = np.array(mus)
        hs = np.array(hs); as_ = np.array(as_); ws = np.array(ws)

        def neg_ll(rho: float) -> float:
            tau = np.ones_like(lams)
            m00 = (hs == 0) & (as_ == 0)
            m01 = (hs == 0) & (as_ == 1)
            m10 = (hs == 1) & (as_ == 0)
            m11 = (hs == 1) & (as_ == 1)
            tau[m00] = 1.0 - lams[m00] * mus[m00] * rho
            tau[m01] = 1.0 + lams[m01] * rho
            tau[m10] = 1.0 + mus[m10] * rho
            tau[m11] = 1.0 - rho
            tau = np.clip(tau, 1e-9, None)
            ll = ws * (np.log(tau)
                       + poisson.logpmf(hs, lams)
                       + poisson.logpmf(as_, mus))
            return -np.sum(ll)

        res = minimize_scalar(neg_ll, bounds=(-0.1, 0.1), method="bounded")
        self.rho = float(res.x) if res.success else 0.0

    # -------------------------------------------------------------- predict
    def _lab(self, team: str) -> str:
        return team if team in self._known else self.OTHER

    def predict_lambdas(self, home: str, away: str, neutral: bool):
        h, a = self._lab(home), self._lab(away)
        a_h = self.attack.get(h, 0.0)
        d_h = self.defense.get(h, 0.0)
        a_a = self.attack.get(a, 0.0)
        d_a = self.defense.get(a, 0.0)
        home_term = 0.0 if neutral else self.home_coef
        lam = math.exp(self.intercept + a_h + d_a + home_term)  # buts domicile
        mu = math.exp(self.intercept + a_a + d_h)               # buts extérieur
        # garde-fous numériques
        return min(lam, 8.0), min(mu, 8.0)

    def score_matrix(self, home: str, away: str, neutral: bool) -> np.ndarray:
        lam, mu = self.predict_lambdas(home, away, neutral)
        n = self.max_goals + 1
        ph = poisson.pmf(np.arange(n), lam)
        pa = poisson.pmf(np.arange(n), mu)
        mat = np.outer(ph, pa)
        for i in (0, 1):
            for j in (0, 1):
                mat[i, j] *= _dc_tau(i, j, lam, mu, self.rho)
        mat /= mat.sum()
        return mat
