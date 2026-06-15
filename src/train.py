"""Entraînement complet et sauvegarde du modèle.

Pipeline :
1. Chargement des matchs.
2. Calcul des features Elo + forme (sans fuite de données).
3. Ajustement du modèle Dixon-Coles sur la fenêtre récente.
4. Ajustement du modèle ML (XGBoost / GBR) sur l'historique enrichi.
5. Sauvegarde du "bundle" complet dans artifacts/model.pkl.

Usage :
    python -m src.train
"""
from __future__ import annotations

import datetime as dt

import joblib
import pandas as pd

from src import config
from src.data.ingest import load_matches
from src.features.elo import compute_elo
from src.features.form import compute_form
from src.models.poisson_dixon_coles import DixonColesModel
from src.models.xgb_goals import XGBGoals


def build_bundle(matches: pd.DataFrame, ref_date=None,
                 fit_xgb: bool = True) -> dict:
    """Construit un modèle complet à partir d'un historique de matchs.

    `matches` doit être trié par date et ne contenir que des matchs
    ANTÉRIEURS à la date de prédiction (pas de fuite de données).
    """
    matches = matches.sort_values("date").reset_index(drop=True)
    if ref_date is None:
        ref_date = matches["date"].max()
    ref_date = pd.Timestamp(ref_date)

    # Features chronologiques.
    elo_df, elo_final = compute_elo(matches)
    feat_df, form_final = compute_form(elo_df)

    # Fenêtre récente pour Dixon-Coles.
    cutoff = ref_date - pd.DateOffset(years=config.TRAIN_YEARS)
    recent = matches[matches["date"] >= cutoff]
    if len(recent) < 200:
        recent = matches
    dc = DixonColesModel().fit(recent, ref_date=ref_date)

    xgb = None
    if fit_xgb:
        # On entraîne le ML sur l'historique avec features valides.
        train_feat = feat_df[feat_df["date"] >= cutoff]
        if len(train_feat) < 200:
            train_feat = feat_df
        try:
            xgb = XGBGoals().fit(train_feat)
        except Exception as exc:  # pragma: no cover
            print(f"[train] XGB ignoré ({exc}); Dixon-Coles seul.")
            xgb = None

    teams = sorted(set(matches["home_team"]) | set(matches["away_team"]))
    return {
        "dc": dc,
        "xgb": xgb,
        "elo": elo_final,
        "form": form_final,
        "teams": teams,
        "blend": config.DC_BLEND_WEIGHT,
        "meta": {
            "trained_at": dt.datetime.utcnow().isoformat(),
            "ref_date": ref_date.isoformat(),
            "n_matches": int(len(matches)),
            "n_teams": len(teams),
            "train_years": config.TRAIN_YEARS,
            "half_life_days": config.HALF_LIFE_DAYS,
            "dc_rho": dc.rho,
        },
    }


def main() -> None:
    config.ensure_dirs()
    matches = load_matches()
    print(f"[train] {len(matches):,} matchs chargés.")
    bundle = build_bundle(matches)
    joblib.dump(bundle, config.MODEL_PATH)
    print(f"[train] Modèle sauvegardé -> {config.MODEL_PATH}")
    print(f"[train] rho (Dixon-Coles) = {bundle['meta']['dc_rho']:.4f}")
    print(f"[train] {bundle['meta']['n_teams']} équipes, "
          f"backend ML = "
          f"{getattr(bundle['xgb'], 'backend', 'aucun')}")


if __name__ == "__main__":
    main()
