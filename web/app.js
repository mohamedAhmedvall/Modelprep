const $ = (id) => document.getElementById(id);

async function loadTeams() {
  try {
    const res = await fetch("/api/teams");
    if (!res.ok) throw new Error("Modèle non entraîné");
    const data = await res.json();
    const dl = $("teamlist");
    dl.innerHTML = data.teams.map((t) => `<option value="${t}">`).join("");
    $("status").textContent = `${data.teams.length} équipes chargées.`;
  } catch (e) {
    $("status").textContent =
      "⚠️ Modèle non disponible. Lancez `python -m src.train` puis rechargez.";
  }
}

function pct(x) {
  return (x * 100).toFixed(1) + "%";
}

function bar(label, value, color) {
  const w = Math.max(2, value * 100);
  return `<div class="bar-row"><span class="barlabel">${label}</span>
    <div class="bartrack"><div class="barfill"
      style="width:${w}%;background:${color}">${pct(value)}</div></div></div>`;
}

async function predict() {
  const home = $("home").value.trim();
  const away = $("away").value.trim();
  const neutral = $("neutral").checked;
  if (!home || !away) {
    $("status").textContent = "Renseignez les deux équipes.";
    return;
  }
  $("status").textContent = "Calcul en cours...";
  const url = `/api/predict?home=${encodeURIComponent(home)}&away=${encodeURIComponent(
    away
  )}&neutral=${neutral}`;
  const res = await fetch(url);
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    $("status").textContent = "Erreur : " + (err.detail || res.status);
    return;
  }
  const d = await res.json();
  $("status").textContent = "";
  render(d);
}

function _render(d) {
  $("results").classList.remove("hidden");
  $("bigscore").textContent = d.predicted_score;
  $("bigprob").textContent = "probabilité " + pct(d.predicted_prob);
  $("xg").textContent =
    `Buts attendus : ${d.home_team} ${d.expected_goals.home} — ` +
    `${d.expected_goals.away} ${d.away_team}`;

  $("topscores").innerHTML =
    "<tr><th>Score</th><th>Probabilité</th></tr>" +
    d.top_scores
      .map((s) => `<tr><td>${s.score}</td><td>${pct(s.prob)}</td></tr>`)
      .join("");

  const o = d.outcome_1x2;
  $("bars1x2").innerHTML =
    bar(d.home_team, o.home, "var(--home)") +
    bar("Nul", o.draw, "var(--draw)") +
    bar(d.away_team, o.away, "var(--away)");

  $("ou").innerHTML =
    bar("+2.5", d.over_under_2_5.over, "var(--accent)") +
    bar("-2.5", d.over_under_2_5.under, "var(--muted)");
  $("btts").innerHTML =
    bar("Oui", d.btts.yes, "var(--accent)") +
    bar("Non", d.btts.no, "var(--muted)");

  const warn = $("warn");
  if (!d.known_teams.home || !d.known_teams.away) {
    warn.classList.remove("hidden");
    warn.textContent =
      "⚠️ Une équipe est inconnue de l'historique : prédiction moins fiable.";
  } else {
    warn.classList.add("hidden");
  }
}

function render(d) {
  window.__last = { home: d.home_team, away: d.away_team, neutral: d.neutral };
  $("valuecard").classList.remove("hidden");
  return _render(d);
}

const ODDS_FIELDS = [
  "home", "draw", "away", "over25", "under25", "btts_yes", "btts_no",
];
const ODDS_LABEL = {
  home: "1 (dom.)", draw: "Nul", away: "2 (ext.)",
  over25: "+2.5", under25: "-2.5", btts_yes: "BTTS oui", btts_no: "BTTS non",
};

async function analyzeValue() {
  if (!window.__last) {
    $("status").textContent = "Faites d'abord une prédiction.";
    return;
  }
  const odds = {};
  ODDS_FIELDS.forEach((f) => {
    const v = parseFloat($("o_" + f).value);
    if (v && v > 1) odds[f] = v;
  });
  if (Object.keys(odds).length === 0) {
    $("valuetable").innerHTML =
      "<tr><td>Saisis au moins une cote (> 1).</td></tr>";
    return;
  }
  const body = {
    ...window.__last,
    odds,
    bankroll: parseFloat($("bankroll").value) || 100,
  };
  const res = await fetch("/api/value", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    $("valuetable").innerHTML = `<tr><td>Erreur : ${err.detail || res.status}</td></tr>`;
    return;
  }
  const d = await res.json();
  renderValue(d);
}

function renderValue(d) {
  let html =
    "<tr><th>Sélection</th><th>Cote</th><th>Proba modèle</th>" +
    "<th>Edge</th><th>Mise Kelly</th></tr>";
  d.all.forEach((r) => {
    if (r.model_prob == null) return;
    const cls = r.is_value ? ' style="color:var(--accent);font-weight:700"' : "";
    html +=
      `<tr${cls}><td>${ODDS_LABEL[r.selection] || r.selection}</td>` +
      `<td>${r.odds}</td><td>${pct(r.model_prob)}</td>` +
      `<td>${(r.edge * 100).toFixed(1)}%</td>` +
      `<td>${r.is_value ? r.kelly_stake : "—"}</td></tr>`;
  });
  $("valuetable").innerHTML = html;
  if (!d.has_value) {
    $("valuetable").innerHTML +=
      '<tr><td colspan="5" class="muted">Aucun pari à valeur sur ces cotes ' +
      "(le bookmaker est correctement margé). Mieux vaut s'abstenir.</td></tr>";
  }
}

$("go").addEventListener("click", predict);
$("govalue").addEventListener("click", analyzeValue);
["home", "away"].forEach((id) =>
  $(id).addEventListener("keydown", (e) => e.key === "Enter" && predict())
);
loadTeams();
