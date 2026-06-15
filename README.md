# ⚽ ModelPrep — Prédiction de score (Coupe du Monde)

Application web qui prédit le **score exact** (et les marchés 1X2, +/-2.5 buts,
BTTS) des matchs de **Coupe du Monde**, à partir d'une **vraie analyse ML** :
ratings **Elo**, modèle statistique **Dixon-Coles** (Poisson bivarié) et
**gradient boosting** (XGBoost), le tout **backtesté** sur les éditions passées.

> ⚠️ **Honnêteté avant tout.** Le score exact est l'un des paris les plus
> durs : même un excellent modèle ne dépasse guère **12–18 %** de réussite sur
> le score exact. L'objectif ici est de fournir des **probabilités calibrées**
> et un avantage statistique, **pas une promesse de gains**. Pariez de façon
> responsable.

## 🧠 Comment ça marche

| Brique | Rôle |
|---|---|
| **Elo** (`src/features/elo.py`) | Force globale de chaque sélection, mise à jour match par match (pondérée par l'importance et l'écart de buts). |
| **Forme** (`src/features/form.py`) | Moyennes glissantes de buts marqués/encaissés (5 derniers matchs). |
| **Dixon-Coles** (`src/models/poisson_dixon_coles.py`) | Modèle de buts → **matrice complète des probabilités de score**. Régression de Poisson pondérée dans le temps + correction `rho` pour les petits scores. |
| **XGBoost** (`src/models/xgb_goals.py`) | Estimation ML des buts attendus à partir des features. Mélangé avec Dixon-Coles. |
| **Backtest** (`src/backtest.py`) | Validation *walk-forward* : on n'entraîne que sur le passé de chaque CDM. |
| **Value betting** (`src/value.py`) | Compare les probabilités du modèle aux cotes du bookmaker, repère les paris à valeur (edge > 0) et calcule la mise de Kelly fractionnée. C'est le vrai levier vers un ROI positif. |

Les buts attendus des deux modèles sont **mélangés** (`DC_BLEND_WEIGHT` dans
`src/config.py`), puis transformés en matrice de scores d'où l'on dérive tous
les marchés (`src/models/markets.py`).

## 📊 Données

Dataset open-source **[martj42/international_results](https://github.com/martj42/international_results)**
(matchs internationaux de 1872 à aujourd'hui, licence MIT). Téléchargé et
nettoyé automatiquement par `src/data/ingest.py`.

## 🚀 Installation & lancement

```bash
# 1. Dépendances (un virtualenv est recommandé)
pip install -r requirements.txt

# 2. Télécharger + nettoyer les données
python -m src.data.ingest

# 3. Entraîner le modèle (crée artifacts/model.pkl)
python -m src.train

# 4. (Optionnel) Backtester sur les CDM récentes
python -m src.backtest --since 2010

# 5. Lancer l'application web
uvicorn api.main:app --port 8000
# puis ouvrir http://localhost:8000
```

### Prédiction en ligne de commande

```bash
python -m src.predict France Brazil            # terrain neutre (défaut CDM)
python -m src.predict France Croatia --not-neutral
```

## 🌐 API

| Endpoint | Description |
|---|---|
| `GET /api/teams` | Liste des équipes connues. |
| `GET /api/predict?home=France&away=Brazil&neutral=true` | Prédiction complète. |
| `POST /api/value` | Value betting : envoie `{home, away, neutral, odds:{...}, bankroll}` → edges + mises Kelly. |
| `GET /api/report` | Dernier rapport de backtest (JSON). |
| `GET /` | Interface web. |

### Aperçu hors-ligne (partage mobile)

```bash
python -m scripts.build_preview   # génère preview.html (autonome, sans serveur)
```
Le fichier `preview.html` embarque de vraies prédictions pré-calculées et
s'ouvre dans n'importe quel navigateur (idéal pour une démo sur téléphone).

## 📈 Résultats du backtest

Mesures *walk-forward* réelles (entraînement strictement sur le passé de chaque
édition), Coupes du Monde 2010 → 2026, **268 matchs évalués** :

| Métrique | Valeur |
|---|---|
| Score exact (top-1) | **≈ 11,9 %** |
| Résultat 1X2 correct | **≈ 55,6 %** |
| Log-loss 1X2 | ≈ 0,98 |
| Brier 1X2 | ≈ 0,58 |

Ces chiffres sont **conformes à l'état de l'art** : ~12 % de score exact est un
bon résultat, et 55 % de 1X2 correct bat largement le hasard (~33 %). Reproduire
avec `python -m src.backtest --since 2010` (vos chiffres peuvent varier
légèrement selon la version des données).

Lancez `python -m src.backtest` pour (re)générer `artifacts/backtest_report.json`.
Le rapport contient, par édition et au global :

- `exact_acc` — taux de score exact,
- `result_acc` — taux de résultat 1X2 correct,
- `logloss` / `brier` — qualité de calibration,
- `roi_exact` — ROI simulé d'une mise fixe sur le score prédit (cote supposée).

> Le dataset international ne contient pas les cotes des bookmakers : le ROI est
> donc *simulé* à cote moyenne paramétrable (`--odds`). Pour un ROI réel,
> branchez un flux de cotes et comparez les probabilités du modèle aux cotes
> proposées (value betting).

## 🗂️ Structure

```
modelprep/
├── data/                  # données brutes + traitées (régénérables)
├── artifacts/             # modèle entraîné + rapport de backtest
├── src/
│   ├── config.py          # paramètres centraux
│   ├── data/ingest.py     # téléchargement + nettoyage
│   ├── features/          # elo.py, form.py
│   ├── models/            # poisson_dixon_coles.py, xgb_goals.py, markets.py
│   ├── train.py           # entraînement
│   ├── backtest.py        # validation walk-forward
│   └── predict.py         # prédiction d'un match
├── api/main.py            # FastAPI
└── web/                   # interface (HTML/CSS/JS)
```

## ⚖️ Avertissement

Outil **éducatif et statistique**. Les paris sportifs comportent un risque de
perte financière. Aucune prédiction ne garantit un résultat.
