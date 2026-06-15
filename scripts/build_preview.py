"""Génère un aperçu HTML AUTONOME (hors-ligne) de l'application.

Calcule de vraies prédictions du modèle pour un panel d'équipes et les
embarque dans un seul fichier HTML interactif (aucun serveur requis).
Idéal pour partager une démo (mobile compris).

Usage :
    python -m scripts.build_preview
    -> écrit preview.html à la racine du projet.
"""
from __future__ import annotations

import itertools
import json

from src import config
from src.predict import load_bundle, predict_from_bundle

TEAMS = [
    "Argentina", "France", "Brazil", "England", "Spain", "Portugal",
    "Netherlands", "Germany", "Belgium", "Croatia", "Italy", "Morocco",
    "Uruguay", "Mexico", "United States", "Japan", "Senegal", "Switzerland",
]

STYLE = (config.WEB_DIR / "style.css").read_text(encoding="utf-8")

HTML = """<!DOCTYPE html>
<html lang="fr"><head><meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>ModelPrep — Aperçu (démo hors-ligne)</title>
<style>__STYLE__
select{background:#0d1119;border:1px solid #2a3346;color:var(--text);
border-radius:8px;padding:10px;font-size:1rem;width:100%;}
.badge{display:inline-block;background:#0d1119;color:var(--accent);
border:1px solid var(--accent);border-radius:6px;padding:2px 8px;
font-size:.7rem;margin-left:8px;}
</style></head>
<body>
<header><h1>⚽ ModelPrep <span class="badge">DÉMO HORS-LIGNE</span></h1>
<p class="subtitle">Aperçu interactif — prédictions réelles pré-calculées</p>
</header>
<main>
<section class="card form">
<div class="row">
<label>Équipe domicile<select id="home"></select></label>
<span class="vs">VS</span>
<label>Équipe extérieur<select id="away"></select></label>
</div>
<p class="status">Match sur terrain neutre (typique en Coupe du Monde).</p>
</section>
<section id="results" class="card">
<div class="headline">
<div class="score-box"><span class="label">Score le plus probable</span>
<span id="bigscore" class="bigscore">—</span>
<span id="bigprob" class="bigprob"></span></div>
<div class="xg"><span id="xg" class="muted"></span></div></div>
<h3>Top 5 des scores probables</h3><table id="topscores"></table>
<h3>Résultat (1X2)</h3><div id="bars1x2" class="bars"></div>
<div class="markets">
<div><h4>Plus / Moins 2.5 buts</h4><div id="ou" class="bars small"></div></div>
<div><h4>Les deux équipes marquent</h4><div id="btts" class="bars small"></div></div>
</div></section>
<p class="disclaimer">⚠️ Démo : prédictions réelles du modèle (Dixon-Coles + XGBoost)
pour un panel d'équipes. L'app complète permet toutes les équipes + value betting.
Le score exact reste difficile (~12% au mieux). Pariez responsable.</p>
</main>
<script>
const DATA = __DATA__;
const TEAMS = __TEAMS__;
const $=(id)=>document.getElementById(id);
const pct=(x)=>(x*100).toFixed(1)+"%";
function bar(l,v,c){const w=Math.max(2,v*100);
return `<div class="bar-row"><span class="barlabel">${l}</span>
<div class="bartrack"><div class="barfill" style="width:${w}%;background:${c}">${pct(v)}</div></div></div>`;}
function fill(sel,def){sel.innerHTML=TEAMS.map(t=>`<option ${t===def?"selected":""}>${t}</option>`).join("");}
function render(){const h=$("home").value,a=$("away").value;
if(h===a){$("bigscore").textContent="=";$("bigprob").textContent="choisissez 2 équipes différentes";return;}
const d=DATA[h+"|"+a];if(!d){return;}
$("bigscore").textContent=d.predicted_score;
$("bigprob").textContent="probabilité "+pct(d.predicted_prob);
$("xg").textContent=`Buts attendus : ${h} ${d.expected_goals.home} — ${d.expected_goals.away} ${a}`;
$("topscores").innerHTML="<tr><th>Score</th><th>Probabilité</th></tr>"+
d.top_scores.map(s=>`<tr><td>${s.score}</td><td>${pct(s.prob)}</td></tr>`).join("");
const o=d.outcome_1x2;
$("bars1x2").innerHTML=bar(h,o.home,"var(--home)")+bar("Nul",o.draw,"var(--draw)")+bar(a,o.away,"var(--away)");
$("ou").innerHTML=bar("+2.5",d.over_under_2_5.over,"var(--accent)")+bar("-2.5",d.over_under_2_5.under,"var(--muted)");
$("btts").innerHTML=bar("Oui",d.btts.yes,"var(--accent)")+bar("Non",d.btts.no,"var(--muted)");}
fill($("home"),TEAMS[0]);fill($("away"),TEAMS[1]);
$("home").addEventListener("change",render);$("away").addEventListener("change",render);
render();
</script></body></html>"""


def main() -> None:
    bundle = load_bundle()
    data = {}
    for h, a in itertools.permutations(TEAMS, 2):
        p = predict_from_bundle(bundle, h, a, neutral=True)
        data[f"{h}|{a}"] = {
            "predicted_score": p["predicted_score"],
            "predicted_prob": p["predicted_prob"],
            "top_scores": p["top_scores"],
            "outcome_1x2": p["outcome_1x2"],
            "over_under_2_5": p["over_under_2_5"],
            "btts": p["btts"],
            "expected_goals": p["expected_goals"],
        }
    html = (HTML
            .replace("__STYLE__", STYLE)
            .replace("__DATA__", json.dumps(data, ensure_ascii=False))
            .replace("__TEAMS__", json.dumps(TEAMS, ensure_ascii=False)))
    out = config.ROOT / "preview.html"
    out.write_text(html, encoding="utf-8")
    print(f"[preview] {len(data)} prédictions -> {out} "
          f"({out.stat().st_size // 1024} Ko)")


if __name__ == "__main__":
    main()
