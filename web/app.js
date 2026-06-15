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

function render(d) {
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

$("go").addEventListener("click", predict);
["home", "away"].forEach((id) =>
  $(id).addEventListener("keydown", (e) => e.key === "Enter" && predict())
);
loadTeams();
