"""Prédiction d'un match à partir d'un modèle entraîné."""
from __future__ import annotations

from typing import Optional

import joblib

from src import config
from src.models.markets import markets_from_matrix
from src.models.poisson_dixon_coles import build_matrix

_BUNDLE = None


def load_bundle(path=None):
    global _BUNDLE
    if _BUNDLE is None:
        path = path or config.MODEL_PATH
        if not path.exists():
            raise FileNotFoundError(
                f"Modèle introuvable ({path}). Lancez d'abord "
                f"`python -m src.train`."
            )
        _BUNDLE = joblib.load(path)
    return _BUNDLE


def _xgb_feature(bundle, home, away, neutral):
    from src.config import tournament_weight

    elo = bundle["elo"]
    form = bundle["form"]
    eh = elo.get(home, 1500.0)
    ea = elo.get(away, 1500.0)
    fh = form.get(home, {"gf": 1.2, "ga": 1.2})
    fa = form.get(away, {"gf": 1.2, "ga": 1.2})
    return {
        "home_elo_pre": eh, "away_elo_pre": ea, "elo_diff_pre": eh - ea,
        "home_gf5": fh["gf"], "home_ga5": fh["ga"],
        "away_gf5": fa["gf"], "away_ga5": fa["ga"],
        "neutral_int": int(neutral),
        "tw": tournament_weight("FIFA World Cup"),
    }


def predict_from_bundle(bundle, home: str, away: str,
                        neutral: bool = True) -> dict:
    dc = bundle["dc"]
    lam, mu = dc.predict_lambdas(home, away, neutral)

    xgb = bundle.get("xgb")
    if xgb is not None:
        try:
            xl, xm = xgb.predict_lambdas(_xgb_feature(bundle, home, away,
                                                      neutral))
            w = bundle.get("blend", config.DC_BLEND_WEIGHT)
            lam = w * lam + (1 - w) * xl
            mu = w * mu + (1 - w) * xm
        except Exception:
            pass

    mat = build_matrix(lam, mu, dc.rho, dc.max_goals)
    result = markets_from_matrix(mat)
    result.update({
        "home_team": home,
        "away_team": away,
        "neutral": neutral,
        "expected_goals": {"home": round(lam, 2), "away": round(mu, 2)},
        "known_teams": {
            "home": home in bundle["teams"],
            "away": away in bundle["teams"],
        },
    })
    return result


def predict(home: str, away: str, neutral: bool = True,
            path=None) -> dict:
    return predict_from_bundle(load_bundle(path), home, away, neutral)


if __name__ == "__main__":
    import argparse
    import json

    p = argparse.ArgumentParser(description="Prédire un match.")
    p.add_argument("home")
    p.add_argument("away")
    p.add_argument("--not-neutral", action="store_true",
                   help="Le match a un vrai terrain à domicile (sinon neutre).")
    args = p.parse_args()
    res = predict(args.home, args.away, neutral=not args.not_neutral)
    print(json.dumps(res, indent=2, ensure_ascii=False))
