"""Backtest walk-forward sur les Coupes du Monde passées.

Pour chaque édition de Coupe du Monde, on (ré)entraîne le modèle
UNIQUEMENT sur les matchs antérieurs au tournoi, puis on prédit chaque
match de la CDM. On agrège ensuite des métriques honnêtes :

- exact_acc : % de scores exacts correctement prédits (score le + probable)
- result_acc : % de résultats 1X2 corrects
- logloss / brier : qualité de calibration des probabilités 1X2
- roi_exact : ROI simulé si l'on mise une mise fixe sur le score prédit
  à une cote moyenne supposée (paramétrable, défaut 7.5)

Usage :
    python -m src.backtest --since 2010
"""
from __future__ import annotations

import argparse
import json
import math

import numpy as np
import pandas as pd

from src import config
from src.data.ingest import load_matches
from src.predict import predict_from_bundle
from src.train import build_bundle

EPS = 1e-12


def _result(home_g: int, away_g: int) -> str:
    if home_g > away_g:
        return "home"
    if home_g < away_g:
        return "away"
    return "draw"


def run(since_year: int = 2010, fit_xgb: bool = True,
        assumed_exact_odds: float = 7.5) -> dict:
    matches = load_matches()
    wc = matches[matches["tournament"] == "FIFA World Cup"].copy()
    wc["year"] = wc["date"].dt.year
    years = sorted(y for y in wc["year"].unique() if y >= since_year)
    if not years:
        raise SystemExit("Aucune Coupe du Monde dans la fenêtre demandée.")

    rows = []
    per_year = {}
    for year in years:
        year = int(year)
        wc_year = wc[wc["year"] == year].sort_values("date")
        start = wc_year["date"].min()
        train = matches[matches["date"] < start]
        if len(train) < 1000:
            continue
        print(f"[backtest] CDM {year} : entraînement sur "
              f"{len(train):,} matchs, {len(wc_year)} matchs à prédire...")
        bundle = build_bundle(train, ref_date=start, fit_xgb=fit_xgb)

        y_rows = []
        for r in wc_year.itertuples(index=False):
            pred = predict_from_bundle(bundle, r.home_team, r.away_team,
                                       neutral=bool(r.neutral))
            actual_score = f"{int(r.home_score)}-{int(r.away_score)}"
            actual_res = _result(int(r.home_score), int(r.away_score))
            o = pred["outcome_1x2"]
            probs = {"home": o["home"], "draw": o["draw"], "away": o["away"]}
            pred_res = max(probs, key=probs.get)
            row = {
                "year": year,
                "match": f"{r.home_team} v {r.away_team}",
                "actual": actual_score,
                "predicted": pred["predicted_score"],
                "exact_hit": int(actual_score == pred["predicted_score"]),
                "result_hit": int(actual_res == pred_res),
                "p_actual_res": max(probs[actual_res], EPS),
                "brier": sum((probs[k] - (1.0 if k == actual_res else 0.0)) ** 2
                             for k in probs),
            }
            y_rows.append(row)
            rows.append(row)

        ydf = pd.DataFrame(y_rows)
        per_year[year] = _summarize(ydf, assumed_exact_odds)
        print(f"           -> score exact {per_year[year]['exact_acc']:.1%}, "
              f"1X2 {per_year[year]['result_acc']:.1%}")

    alldf = pd.DataFrame(rows)
    summary = _summarize(alldf, assumed_exact_odds)
    summary["per_year"] = per_year
    summary["assumed_exact_odds"] = assumed_exact_odds
    summary["n_matches"] = int(len(alldf))
    summary["years"] = [int(y) for y in years]

    config.ensure_dirs()
    config.BACKTEST_REPORT.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))
    print("\n========== RÉSUMÉ GLOBAL ==========")
    print(f"Matchs évalués      : {summary['n_matches']}")
    print(f"Score exact         : {summary['exact_acc']:.1%}")
    print(f"Résultat 1X2        : {summary['result_acc']:.1%}")
    print(f"Log-loss 1X2        : {summary['logloss']:.3f}")
    print(f"Brier 1X2           : {summary['brier']:.3f}")
    print(f"ROI score exact*    : {summary['roi_exact']:+.1%} "
          f"(cote supposée {assumed_exact_odds})")
    print(f"Rapport -> {config.BACKTEST_REPORT}")
    return summary


def _summarize(df: pd.DataFrame, assumed_exact_odds: float) -> dict:
    n = len(df)
    if n == 0:
        return {}
    exact_acc = float(df["exact_hit"].mean())
    result_acc = float(df["result_hit"].mean())
    logloss = float(-np.log(df["p_actual_res"].clip(EPS, 1)).mean())
    brier = float(df["brier"].mean())
    # ROI : mise 1 unité par match sur le score prédit.
    roi_exact = float(df["exact_hit"].mean() * assumed_exact_odds - 1.0)
    return {
        "exact_acc": round(exact_acc, 4),
        "result_acc": round(result_acc, 4),
        "logloss": round(logloss, 4),
        "brier": round(brier, 4),
        "roi_exact": round(roi_exact, 4),
        "n": n,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--since", type=int, default=2010,
                   help="Année de départ des CDM à backtester.")
    p.add_argument("--no-xgb", action="store_true",
                   help="Dixon-Coles seul (plus rapide).")
    p.add_argument("--odds", type=float, default=7.5,
                   help="Cote moyenne supposée pour le ROI score exact.")
    args = p.parse_args()
    run(since_year=args.since, fit_xgb=not args.no_xgb,
        assumed_exact_odds=args.odds)
