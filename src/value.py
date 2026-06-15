"""Détection de paris à valeur (value betting).

Le seul vrai levier vers un ROI positif n'est pas de "deviner" le score,
mais de parier quand la probabilité du modèle est SUPÉRIEURE à celle
implicite dans la cote du bookmaker.

Pour une sélection de probabilité `p` (modèle) et de cote décimale `o` :
- valeur attendue (edge) = p * o - 1   (>0 => pari favorable à long terme)
- fraction de Kelly       = (p * o - 1) / (o - 1)

On applique un Kelly fractionné (par défaut 1/4) pour limiter la variance.
"""
from __future__ import annotations

from typing import Dict, List, Optional


def edge(prob: float, odds: float) -> float:
    return prob * odds - 1.0


def kelly(prob: float, odds: float) -> float:
    b = odds - 1.0
    if b <= 0:
        return 0.0
    return max(0.0, (prob * odds - 1.0) / b)


def _market_probs(pred: dict) -> Dict[str, float]:
    """Aplatit toutes les probabilités du modèle par 'clé de sélection'."""
    o = pred["outcome_1x2"]
    ou = pred["over_under_2_5"]
    bt = pred["btts"]
    probs = {
        "home": o["home"], "draw": o["draw"], "away": o["away"],
        "over25": ou["over"], "under25": ou["under"],
        "btts_yes": bt["yes"], "btts_no": bt["no"],
    }
    for s in pred.get("top_scores", []):
        probs[s["score"]] = s["prob"]  # ex: "2-1"
    return probs


def analyze(pred: dict, odds: Dict[str, float],
            threshold: float = 0.0, kelly_fraction: float = 0.25,
            bankroll: float = 100.0) -> dict:
    """Compare les cotes fournies aux probabilités du modèle.

    `odds` : dict {clé_sélection: cote_décimale}. Clés possibles :
    home/draw/away, over25/under25, btts_yes/btts_no, ou un score "2-1".
    """
    probs = _market_probs(pred)
    rows: List[dict] = []
    for sel, o in odds.items():
        if o is None or o <= 1.0:
            continue
        p = probs.get(sel)
        if p is None:
            rows.append({"selection": sel, "odds": o, "model_prob": None,
                         "note": "probabilité indisponible (hors top scores)"})
            continue
        e = edge(p, o)
        k = kelly(p, o) * kelly_fraction
        rows.append({
            "selection": sel,
            "odds": round(o, 2),
            "model_prob": round(p, 4),
            "implied_prob": round(1.0 / o, 4),
            "edge": round(e, 4),
            "is_value": e > threshold,
            "kelly_stake": round(k * bankroll, 2),
            "kelly_fraction": round(k, 4),
        })

    value_bets = [r for r in rows if r.get("is_value")]
    value_bets.sort(key=lambda r: r["edge"], reverse=True)
    rows.sort(key=lambda r: (r.get("edge") is not None, r.get("edge", -9)),
              reverse=True)
    return {
        "match": f"{pred['home_team']} v {pred['away_team']}",
        "all": rows,
        "value_bets": value_bets,
        "has_value": bool(value_bets),
        "params": {"threshold": threshold,
                   "kelly_fraction": kelly_fraction,
                   "bankroll": bankroll},
    }
