"""Téléchargement et nettoyage des données de matchs internationaux.

Source : dataset open-source `martj42/international_results` (licence MIT).
Colonnes brutes : date, home_team, away_team, home_score, away_score,
tournament, city, country, neutral.

Usage :
    python -m src.data.ingest
"""
from __future__ import annotations

import sys

import pandas as pd

from src import config


def download(force: bool = False) -> None:
    """Télécharge results.csv s'il n'est pas déjà présent."""
    config.ensure_dirs()
    if config.RAW_RESULTS.exists() and not force:
        print(f"[ingest] Fichier déjà présent : {config.RAW_RESULTS}")
        return
    try:
        import requests

        print(f"[ingest] Téléchargement depuis {config.RESULTS_URL} ...")
        resp = requests.get(config.RESULTS_URL, timeout=60)
        resp.raise_for_status()
        config.RAW_RESULTS.write_bytes(resp.content)
        print(f"[ingest] Enregistré -> {config.RAW_RESULTS}")
    except Exception as exc:  # pragma: no cover - dépend du réseau
        if config.RAW_RESULTS.exists():
            print(f"[ingest] Téléchargement impossible ({exc}), "
                  f"utilisation du cache local.")
        else:
            raise RuntimeError(
                "Impossible de télécharger les données et aucun cache local "
                f"trouvé. Placez manuellement results.csv dans "
                f"{config.DATA_RAW}. Erreur: {exc}"
            ) from exc


def clean() -> pd.DataFrame:
    """Charge, nettoie et enrichit les données brutes."""
    df = pd.read_csv(config.RAW_RESULTS)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "home_score", "away_score",
                           "home_team", "away_team"])
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    # neutral : booléen robuste
    df["neutral"] = (
        df["neutral"].astype(str).str.lower().isin(["true", "1", "yes"])
    )
    df["tournament"] = df["tournament"].fillna("Friendly")
    df = df.sort_values("date").reset_index(drop=True)

    # Résultat 1X2 du point de vue domicile.
    def outcome(r):
        if r.home_score > r.away_score:
            return "H"
        if r.home_score < r.away_score:
            return "A"
        return "D"

    df["result"] = df.apply(outcome, axis=1)
    return df


def build() -> pd.DataFrame:
    download()
    df = clean()
    config.ensure_dirs()
    df.to_csv(config.PROCESSED_MATCHES, index=False)
    print(f"[ingest] {len(df):,} matchs nettoyés -> {config.PROCESSED_MATCHES}")
    print(f"[ingest] Période : {df['date'].min().date()} "
          f"-> {df['date'].max().date()}")
    print(f"[ingest] Équipes distinctes : "
          f"{len(set(df.home_team) | set(df.away_team))}")
    return df


def load_matches() -> pd.DataFrame:
    """Charge les matchs traités (les construit si absents)."""
    if not config.PROCESSED_MATCHES.exists():
        return build()
    df = pd.read_csv(config.PROCESSED_MATCHES, parse_dates=["date"])
    return df


if __name__ == "__main__":
    sys.exit(0 if build() is not None else 1)
