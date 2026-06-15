"""API FastAPI : sert les prédictions et l'interface web.

Lancement :
    uvicorn api.main:app --reload --port 8000
puis ouvrir http://localhost:8000
"""
from __future__ import annotations

import json

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from src import config
from src.predict import load_bundle, predict_from_bundle

app = FastAPI(title="ModelPrep — Prédiction Coupe du Monde", version="1.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/teams")
def teams():
    try:
        bundle = load_bundle()
    except FileNotFoundError as exc:
        raise HTTPException(503, str(exc))
    return {"teams": bundle["teams"]}


@app.get("/api/predict")
def predict(home: str = Query(...), away: str = Query(...),
            neutral: bool = Query(True)):
    try:
        bundle = load_bundle()
    except FileNotFoundError as exc:
        raise HTTPException(503, str(exc))
    if home == away:
        raise HTTPException(400, "Choisissez deux équipes différentes.")
    return predict_from_bundle(bundle, home, away, neutral)


@app.get("/api/report")
def report():
    if not config.BACKTEST_REPORT.exists():
        raise HTTPException(404, "Aucun rapport de backtest. Lancez "
                                 "`python -m src.backtest`.")
    return JSONResponse(json.loads(config.BACKTEST_REPORT.read_text()))


@app.get("/")
def index():
    return FileResponse(config.WEB_DIR / "index.html")


# Fichiers statiques (app.js, style.css).
app.mount("/", StaticFiles(directory=str(config.WEB_DIR)), name="static")
