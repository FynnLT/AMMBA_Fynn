/* AMMBA Market Simulator — vanilla JS, no build step.
 *
 * The UI plays the roles the PoC does not implement as separate services:
 * the Market Orchestrator (creates markets, triggers clearing) and the
 * participants (submit orders, report smart-meter readings).
 */

"use strict";

// ------------------------------------------------------------- endpoints

const qs = new URLSearchParams(location.search);
const EP = {
  db: (qs.get("db") || "http://localhost:8080").replace(/\/$/, ""),
  clearing: (qs.get("clearing") || "http://localhost:8081").replace(/\/$/, ""),
  execution: (qs.get("execution") || "http://localhost:8082").replace(/\/$/, ""),
};
const TIME_SLOT_SEC = 900;

// ------------------------------------------------------------------ state

let idSeq = 0;
const state = {
  config: { name: "Community 1", kUpper: 40.0, kLower: 8.0, theta: 1.0, steepness: 0.6 },
  // Numeric multiplier parameters sent with the trigger. The rules themselves
  // (allocation order, multiplier mode/sides) stay in the backend
  // configuration and are only *displayed* here — nothing about the
  // mechanism is computed or chosen in the browser.
  preferences: { greenMultiplier: 0.10, greyLevy: 0.10, levyCap: 0.20 },
  // Defaults reproduce the implementation guide's example:
  // supply 12.5 kWh vs demand 10 kWh -> ratio 1.25 -> ~15.15 ct/kWh
  // `type` = energy source (green/grey); `partner` = preferred trading
  // partner (id of an opposite-side participant, or null).
  producers: [
    { id: ++idSeq, name: "Rooftop PV A", energy: 5.0, type: "green", partner: null },
    { id: ++idSeq, name: "Rooftop PV B", energy: 3.5, type: "green", partner: null },
    { id: ++idSeq, name: "Community Battery", energy: 4.0, type: "grey", partner: null },
  ],
  consumers: [
    { id: ++idSeq, name: "Household 1", energy: 4.5, partner: null },
    { id: ++idSeq, name: "Household 2", energy: 3.0, partner: null },
    { id: ++idSeq, name: "Bakery", energy: 2.5, partner: null },
  ],
  lastClearing: null,   // response of /trigger-clearing
  lastMarket: null,     // {market_id, community_uuid, time_slot}
  areaByName: {},       // participant name -> area_uuid for the last run
};
// Demo default: PV A and Household 1 nominate each other -> a mutual pair,
// which the backend serves before the pro-rata residual (96.9 % vs 80 % fill
// for PV A in the guide's reference market). Clear one side to see the
// "no mutual pair" case — a match requires both.
state.producers[0].partner = state.consumers[0].id;
state.consumers[0].partner = state.producers[0].id;

// ---------------------------------------------------------------- helpers

const $ = (sel) => document.querySelector(sel);

function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function slug(s) {
  return String(s).toLowerCase().replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "") || "x";
}

const fmt = {
  kwh: (v) => `${(+v).toFixed(2)} kWh`,
  price: (v) => `${(+v).toFixed(2)} ct/kWh`,
  ct: (v) => `${(+v).toFixed(2)} ct`,
  pct: (v) => `${(100 * v).toFixed(1)} %`,
  ratio: (v) => (+v).toFixed(3),
  eur: (ct) => `€${(ct / 100).toFixed(2)}`,   // values are in ct -> euros
  slot: (ts) => new Date(ts * 1000).toLocaleString([], {
    dateStyle: "medium", timeStyle: "short" }),
};

function showError(message) {
  const toast = $("#toast");
  toast.innerHTML = `<b>Request failed</b>${esc(message)}`;
  toast.classList.add("show");
  clearTimeout(showError._t);
  showError._t = setTimeout(() => toast.classList.remove("show"), 9000);
}
document.addEventListener("DOMContentLoaded", () =>
  $("#toast").addEventListener("click", () => $("#toast").classList.remove("show")));

async function api(base, path, opts = {}) {
  const init = { method: opts.method || (opts.body ? "POST" : "GET") };
  if (opts.body !== undefined) {
    init.headers = { "Content-Type": "application/json" };
    init.body = JSON.stringify(opts.body);
  }
  let resp;
  try {
    resp = await fetch(base + path, init);
  } catch {
    throw new Error(`${base}${path} is unreachable — are the services up? (docker-compose up)`);
  }
  if (!resp.ok) {
    let detail = "";
    try { const j = await resp.json(); detail = j.detail || JSON.stringify(j); }
    catch { detail = resp.statusText; }
    throw new Error(`HTTP ${resp.status} on ${path}: ${detail}`);
  }
  return resp.json();
}

// ----------------------------------------------------------- sigmoid math

// Mirrors amm-clearing-node/src/sigmoid.py for live previews only —
// the authoritative price always comes from the Clearing Node.
function sigmoidPrice(ratio, c) {
  if (c.kUpper <= c.kLower) return c.kLower;
  const arg = -c.steepness * (ratio - c.theta);
  if (arg > 700) return c.kUpper;
  if (arg < -700) return c.kLower;
  return c.kUpper - (c.kUpper - c.kLower) / (1 + Math.exp(arg));
}

function totals() {
  const supply = state.producers.reduce((s, p) => s + (+p.energy || 0), 0);
  const demand = state.consumers.reduce((s, c) => s + (+c.energy || 0), 0);
  return { supply, demand, ratio: demand > 0 ? supply / demand : null };
}

// ------------------------------------------------------------ SVG charts

function sigmoidChartSVG(c, marker) {
  const W = 640, H = 330, m = { l: 58, r: 20, t: 20, b: 44 };
  const xMax = 4;
  const top = Math.max(c.kUpper, c.kLower, 1);
  const yMax = top * 1.15;
  const X = (r) => m.l + (r / xMax) * (W - m.l - m.r);
  const Y = (p) => H - m.b - (p / yMax) * (H - m.t - m.b);

  let g = "";
  // grid + axes ticks
  for (let i = 0; i <= 4; i++) {
    const y = Y((yMax / 4) * i);
    g += `<line x1="${m.l}" y1="${y}" x2="${W - m.r}" y2="${y}" stroke="#eef1f7"/>`;
    g += `<text x="${m.l - 8}" y="${y + 4}" text-anchor="end" font-size="11" fill="#5d6b85">${((yMax / 4) * i).toFixed(1)}</text>`;
  }
  for (let r = 0; r <= xMax; r += 0.5) {
    const x = X(r);
    g += `<line x1="${x}" y1="${H - m.b}" x2="${x}" y2="${H - m.b + (Number.isInteger(r) ? 6 : 3)}" stroke="#aab4c8"/>`;
    if (Number.isInteger(r)) g += `<text x="${x}" y="${H - m.b + 20}" text-anchor="middle" font-size="11" fill="#5d6b85">${r}</text>`;
  }
  g += `<line x1="${m.l}" y1="${H - m.b}" x2="${W - m.r}" y2="${H - m.b}" stroke="#aab4c8"/>`;
  g += `<line x1="${m.l}" y1="${m.t}" x2="${m.l}" y2="${H - m.b}" stroke="#aab4c8"/>`;
  g += `<text x="${(m.l + W - m.r) / 2}" y="${H - 6}" text-anchor="middle" font-size="12" fill="#5d6b85">supply / demand ratio</text>`;
  g += `<text x="14" y="${(m.t + H - m.b) / 2}" font-size="12" fill="#5d6b85" transform="rotate(-90 14 ${(m.t + H - m.b) / 2})" text-anchor="middle">price (ct/kWh)</text>`;

  // K bounds (dashed)
  g += `<line x1="${m.l}" y1="${Y(c.kUpper)}" x2="${W - m.r}" y2="${Y(c.kUpper)}" stroke="#b45309" stroke-dasharray="7 5" stroke-width="1.5"/>`;
  g += `<text x="${W - m.r}" y="${Y(c.kUpper) - 6}" text-anchor="end" font-size="11.5" fill="#b45309">K_upper = ${c.kUpper}</text>`;
  g += `<line x1="${m.l}" y1="${Y(c.kLower)}" x2="${W - m.r}" y2="${Y(c.kLower)}" stroke="#047857" stroke-dasharray="7 5" stroke-width="1.5"/>`;
  g += `<text x="${W - m.r}" y="${Y(c.kLower) + 15}" text-anchor="end" font-size="11.5" fill="#047857">K_lower = ${c.kLower}</text>`;

  // theta marker (vertical dotted)
  if (c.theta >= 0 && c.theta <= xMax) {
    g += `<line x1="${X(c.theta)}" y1="${m.t}" x2="${X(c.theta)}" y2="${H - m.b}" stroke="#aab4c8" stroke-dasharray="2 4"/>`;
    g += `<text x="${X(c.theta) + 4}" y="${m.t + 12}" font-size="11.5" fill="#5d6b85">θ = ${c.theta}</text>`;
  }

  // curve
  let d = "";
  for (let r = 0; r <= xMax + 1e-9; r += 0.04) {
    d += `${d ? "L" : "M"}${X(r).toFixed(1)},${Y(sigmoidPrice(r, c)).toFixed(1)} `;
  }
  g += `<path d="${d}" fill="none" stroke="#4f46e5" stroke-width="2.6"/>`;

  // current point marker
  if (marker && marker.ratio != null) {
    const rShown = Math.min(marker.ratio, xMax);
    const price = marker.price != null ? marker.price : sigmoidPrice(marker.ratio, c);
    const x = X(rShown), y = Y(price);
    g += `<line x1="${m.l}" y1="${y}" x2="${x}" y2="${y}" stroke="#dc2626" stroke-dasharray="4 4"/>`;
    g += `<line x1="${x}" y1="${y}" x2="${x}" y2="${H - m.b}" stroke="#dc2626" stroke-dasharray="4 4"/>`;
    g += `<circle cx="${x}" cy="${y}" r="6" fill="#dc2626" stroke="#fff" stroke-width="2.5"/>`;
    const label = `ratio ${fmt.ratio(marker.ratio)} → ${(+price).toFixed(2)} ct/kWh`;
    const anchor = rShown > 2.6 ? "end" : "start";
    const tx = rShown > 2.6 ? x - 12 : x + 12;
    g += `<text x="${tx}" y="${y - 12}" font-size="12.5" font-weight="700" fill="#dc2626" text-anchor="${anchor}">${label}${marker.ratio > xMax ? " (off-scale)" : ""}</text>`;
  }

  return `<svg viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg" font-family="Segoe UI, sans-serif">${g}</svg>`;
}

function hbar(label, value, max, cls, text) {
  const w = max > 0 ? Math.max(0, Math.min(100, (value / max) * 100)) : 0;
  return `<div class="hbar">
    <span class="hbar-label">${esc(label)}</span>
    <div class="hbar-track"><div class="hbar-fill ${cls}" style="width:${w.toFixed(2)}%"></div></div>
    <span class="hbar-val">${esc(text)}</span>
  </div>`;
}

function supplyDemandBars(supply, demand, traded) {
  const max = Math.max(supply, demand, 1e-9);
  let html = hbar("Supply", supply, max, "supply", fmt.kwh(supply)) +
             hbar("Demand", demand, max, "demand", fmt.kwh(demand));
  if (traded != null) html += hbar("Traded", traded, max, "traded", fmt.kwh(traded));
  return `<div class="hbars">${html}</div>`;
}

// --------------------------------------------------- panel A: configuration

function readConfigInputs() {
  state.config = {
    name: $("#cfg-name").value.trim() || "Community 1",
    kUpper: parseFloat($("#cfg-kupper").value) || 0,
    kLower: parseFloat($("#cfg-klower").value) || 0,
    theta: parseFloat($("#cfg-theta").value) || 0,
    steepness: parseFloat($("#cfg-steepness").value) || 0,
  };
}

let _chartRaf = 0;
function renderConfigChart() {
  if (_chartRaf) return;
  _chartRaf = requestAnimationFrame(() => {
    _chartRaf = 0;
    const { ratio } = totals();
    $("#cfg-chart").innerHTML = sigmoidChartSVG(state.config,
      ratio != null ? { ratio } : null);
    const warning = $("#cfg-warning");
    if (state.config.kUpper <= state.config.kLower) {
      warning.style.display = "";
      warning.textContent = "K_upper ≤ K_lower: degenerate price band — the clearing price is fixed at K_lower.";
    } else {
      warning.style.display = "none";
    }
  });
}

// --------------------------------------------------- panel B: participants

function partnerSelect(p, oppositeList) {
  // Preferred trading partner: options are the opposite side, value = its id.
  const opts = oppositeList.map((o) =>
    `<option value="${o.id}" data-opt-id="${o.id}"${p.partner === o.id ? " selected" : ""}>${esc(o.name || "(unnamed)")}</option>`).join("");
  return `<select data-field="partner"><option value=""${!p.partner ? " selected" : ""}>— none —</option>${opts}</select>`;
}

function participantRow(p, side, opposite) {
  const name = `<input type="text" data-field="name" value="${esc(p.name)}" placeholder="name">`;
  const energy = `<input type="number" data-field="energy" value="${p.energy}" min="0" step="0.1" placeholder="kWh">`;
  const remove = `<button class="remove" title="remove">✕</button>`;
  if (side === "producer") {
    // producers additionally pick their energy source (green/grey)
    const type = `<select data-field="type">
      <option value="green"${p.type === "green" ? " selected" : ""}>Green</option>
      <option value="grey"${p.type === "grey" ? " selected" : ""}>Grey</option>
    </select>`;
    return `<div class="prow prow-producer" data-id="${p.id}">${name}${energy}${type}${partnerSelect(p, opposite)}${remove}</div>`;
  }
  return `<div class="prow prow-consumer" data-id="${p.id}">${name}${energy}${partnerSelect(p, opposite)}${remove}</div>`;
}

// Update preferred-partner <option> labels in place when a name changes,
// so we don't re-render rows (which would drop input focus mid-typing).
function refreshPartnerLabels() {
  const byId = {};
  for (const x of [...state.producers, ...state.consumers]) byId[x.id] = x;
  document.querySelectorAll('select[data-field="partner"] option[data-opt-id]').forEach((opt) => {
    const m = byId[+opt.dataset.optId];
    if (m) opt.textContent = m.name || "(unnamed)";
  });
}

function renderParticipants() {
  $("#producer-rows").innerHTML =
    state.producers.map((p) => participantRow(p, "producer", state.consumers)).join("");
  $("#consumer-rows").innerHTML =
    state.consumers.map((c) => participantRow(c, "consumer", state.producers)).join("");
  renderLiveStats();
}

function renderLiveStats() {
  const { supply, demand, ratio } = totals();
  const price = ratio != null ? sigmoidPrice(ratio, state.config) : null;
  $("#live-stats").innerHTML = `
    <div class="stat supply"><div class="k">Total supply</div><div class="v">${supply.toFixed(2)} <small>kWh</small></div></div>
    <div class="stat demand"><div class="k">Total demand</div><div class="v">${demand.toFixed(2)} <small>kWh</small></div></div>
    <div class="stat"><div class="k">Ratio s/d</div><div class="v">${ratio != null ? fmt.ratio(ratio) : "—"}</div></div>
    <div class="stat price"><div class="k">Clearing price (preview)</div><div class="v">${price != null ? price.toFixed(2) : "—"} <small>ct/kWh</small></div></div>`;
  $("#live-bar").innerHTML = supplyDemandBars(supply, demand);
  renderConfigChart();
}

function bindParticipantEvents(containerSel, list) {
  const container = $(containerSel);
  container.addEventListener("input", (ev) => {
    const row = ev.target.closest(".prow");
    if (!row) return;
    const item = list().find((p) => p.id === +row.dataset.id);
    if (!item) return;
    const field = ev.target.dataset.field;
    if (field === "name") { item.name = ev.target.value; refreshPartnerLabels(); }
    else if (field === "energy") item.energy = parseFloat(ev.target.value) || 0;
    else if (field === "type") item.type = ev.target.value;
    else if (field === "partner") item.partner = ev.target.value ? +ev.target.value : null;
    renderLiveStats();
  });
  container.addEventListener("click", (ev) => {
    if (!ev.target.classList.contains("remove")) return;
    const id = +ev.target.closest(".prow").dataset.id;
    const arr = list();
    arr.splice(arr.findIndex((p) => p.id === id), 1);
    renderParticipants();
  });
}

// ------------------------------------------------------ panel C: clearing

function setBusy(btnSel, busy) {
  const btn = $(btnSel);
  btn.disabled = busy;
  btn.classList.toggle("busy", busy);
}

function uniqueAreas(participants) {
  // area_uuid per participant; numbered on name collisions
  const seen = {};
  const map = {};
  for (const p of participants) {
    let area = `area-${slug(p.name)}`;
    if (seen[area]) area += `-${++seen[area]}`; else seen[area] = 1;
    map[p.id] = area;
  }
  return map;
}

async function chooseTimeSlot(communityUuid) {
  // The mock generates market_id = blake2b("spot" + time_slot), so each demo
  // run needs a fresh slot: the next boundary after now or after the latest
  // existing market for this community.
  const next = (Math.floor(Date.now() / 1000 / TIME_SLOT_SEC) + 1) * TIME_SLOT_SEC;
  let latest = 0;
  try {
    const markets = await api(EP.db,
      `/community-market?community_uuid=${encodeURIComponent(communityUuid)}`);
    for (const m of markets) latest = Math.max(latest, m.time_slot || 0);
  } catch { /* DB down -> the POST below will surface the real error */ }
  return Math.max(next, latest + TIME_SLOT_SEC);
}

async function runClearing() {
  readConfigInputs();
  const producers = state.producers.filter((p) => p.name.trim() && p.energy > 0);
  const consumers = state.consumers.filter((c) => c.name.trim() && c.energy > 0);
  const status = $("#run-status");
  setBusy("#run-clearing", true);
  try {
    const c = state.config;
    const communityUuid = `community-${slug(c.name)}`;

    status.textContent = "choosing delivery slot…";
    const timeSlot = await chooseTimeSlot(communityUuid);

    status.textContent = "creating market in off-chain DB…";
    const producerAreas = uniqueAreas(producers);
    const consumerAreas = uniqueAreas(consumers);
    // avoid producer/consumer area collisions
    for (const id in consumerAreas) {
      if (Object.values(producerAreas).includes(consumerAreas[id])) {
        consumerAreas[id] += "-load";
      }
    }
    const market = await api(EP.db, "/market", { body: {
      community_uuid: communityUuid,
      community_name: c.name,
      time_slot: timeSlot,
      community_areas: [
        ...producers.map((p) => ({ area_uuid: producerAreas[p.id], name: p.name, area_type: "PV" })),
        ...consumers.map((x) => ({ area_uuid: consumerAreas[x.id], name: x.name, area_type: "Load" })),
      ],
    }});

    state.areaByName = {};
    producers.forEach((p) => { state.areaByName[p.name] = producerAreas[p.id]; });
    consumers.forEach((x) => { state.areaByName[x.name] = consumerAreas[x.id]; });

    // Preferred partners travel as `requirements.preferred_partner`, keyed by
    // area_uuid (the canonical identity — `created_by` is a display name).
    const requirements = (partnerId, areasOfOtherSide) => {
      const area = partnerId ? areasOfOtherSide[partnerId] : null;
      return area ? { requirements: { preferred_partner: area } } : {};
    };

    const orders = [
      // AMMBA is a uniform-price auction: order energy_rate limits are not
      // used by the clearing; offers carry the feed-in floor, bids the
      // retail cap.
      ...producers.map((p) => ({
        order_type: "Offer", created_by: p.name,
        area_uuid: producerAreas[p.id], market_id: market.market_id,
        time_slot: timeSlot, energy: +p.energy, energy_rate: c.kLower,
        attributes: { energy_type: p.type === "grey" ? "grey" : "green" },
        ...requirements(p.partner, consumerAreas),
      })),
      ...consumers.map((x) => ({
        order_type: "Bid", created_by: x.name,
        area_uuid: consumerAreas[x.id], market_id: market.market_id,
        time_slot: timeSlot, energy: +x.energy, energy_rate: c.kUpper,
        ...requirements(x.partner, producerAreas),
      })),
    ];
    if (orders.length) {
      status.textContent = `posting ${orders.length} order(s)…`;
      await api(EP.db, "/orders-normalized", { body: orders });
    }

    status.textContent = "triggering Clearing Node…";
    const pref = readPreferenceInputs();
    const result = await api(EP.clearing, "/trigger-clearing", { body: {
      market_id: market.market_id,
      community_uuid: communityUuid,
      time_slot: timeSlot,
      community_name: c.name,
      sigmoid_params: { k_upper: c.kUpper, k_lower: c.kLower,
                        theta: c.theta, steepness: c.steepness },
      // Only the three numeric parameters. Every field of PreferenceParams is
      // optional and resolve_preferences merges non-null values only, so the
      // omitted settings (enabled, order, mode, sides, multipliers_enabled)
      // fall back to the node's configuration.yaml / environment. Sending
      // nulls would be equivalent but noisier — omit them.
      preference_params: {
        green_multiplier: pref.greenMultiplier, grey_levy: pref.greyLevy,
        levy_cap: pref.levyCap },
    }});

    state.lastClearing = result;
    state.lastMarket = { market_id: market.market_id,
                         community_uuid: communityUuid, time_slot: timeSlot };
    status.textContent = `done — market ${market.market_id.slice(0, 12)}…`;
    renderBackendResults(result);
    renderActuals(result);
  } catch (err) {
    status.textContent = "failed";
    showError(err.message);
  } finally {
    setBusy("#run-clearing", false);
  }
}

// --------------------------------------------- panel: Clearing Results
//
// Overview of the backend round-trip (dashboard mockup layout). Preferred-
// partner priority and the energy-type multipliers are computed by the
// Clearing Node and read out of the response's `preferences` block — the
// browser does no market economics of its own.

function roundBadge(roundType) {
  const cls = (roundType || "").toLowerCase().replace("_", "-");
  return `<span class="badge ${cls}">${esc((roundType || "").replace("_", "-"))}</span>`;
}

const prefBadge = (matched) => matched
  ? ` <span class="badge ok" title="served before the pro-rata residual">paired</span>` : "";

function producersTable(r) {
  const rows = r.producers.map((p) => `<tr>
      <td>${esc(p.name)}${prefBadge(p.matched)}</td>
      <td><span class="badge ${p.type}">${p.type}</span></td>
      <td class="num">${p.requested.toFixed(2)}</td>
      <td class="num">${p.allocated.toFixed(3)}</td>
      <td>${fillCell(p.requested > 0 ? p.allocated / p.requested : 0, "supply")}</td>
      <td class="num">${p.finalRate.toFixed(2)}</td>
      <td class="num">${fmt.eur(p.allocated * p.finalRate)}</td></tr>`).join("");
  return `<table class="data"><thead><tr><th>Producer</th><th>Type</th><th class="num">Offered</th><th class="num">Allocated</th><th>Fill</th><th class="num">Final ct/kWh</th><th class="num">Revenue</th></tr></thead><tbody>${rows}</tbody></table>`;
}

function consumersTable(r) {
  const rows = r.consumers.map((c) => `<tr>
      <td>${esc(c.name)}${prefBadge(c.matched)}</td>
      <td class="num">${c.requested.toFixed(2)}</td>
      <td class="num">${c.allocated.toFixed(3)}</td>
      <td>${fillCell(c.requested > 0 ? c.allocated / c.requested : 0, "demand")}</td>
      <td class="num">${c.finalRate.toFixed(2)}</td>
      <td class="num">${fmt.eur(c.allocated * c.finalRate)}</td></tr>`).join("");
  return `<table class="data"><thead><tr><th>Consumer</th><th class="num">Demanded</th><th class="num">Allocated</th><th>Fill</th><th class="num">ct/kWh</th><th class="num">Cost</th></tr></thead><tbody>${rows}</tbody></table>`;
}

// What the Clearing Node actually did with the preferred partners.
function preferredMatchesHtml(r) {
  if (!r.mutualPairs.length) {
    return `<div class="notice info">No mutual preferred pairs found — the full volume was allocated pro-rata. (A match requires <b>both</b> sides to select each other.)</div>`;
  }
  const list = r.mutualPairs.map((p) =>
    `${esc(p.label)} — ${fmt.kwh(p.energy_kwh)}`).join(" · ");
  if (r.order === "pro_rata_first") {
    return `<div class="notice info">Mutual preferred pair(s): ${list}. Allocation order is <b>pro-rata first</b> (baseline), so the pairs are only flagged for routing — quantities are unchanged.</div>`;
  }
  const rationed = r.pairsRationed
    ? ` The pairs asked for more than the market cleared and were rationed proportionally.`
    : "";
  return `<div class="notice ok">Mutual preferred pair(s) served first at the clearing price: ${list} (${fmt.kwh(r.preferentialKwh)} of ${fmt.kwh(r.traded)} traded). The remainder was split pro-rata.${rationed}</div>`;
}

function scalingNoticeHtml(r) {
  if (r.greenAlloc > 0 && r.greyAlloc <= 0) {
    return `<div class="notice warn">No grey volume in this round — the green bonus has no funding source and was not paid.</div>`;
  }
  const notices = [];
  if (r.greenAlloc > 0 && r.scale < 1 - 1e-9) {
    notices.push(`<div class="notice warn">Grey levy revenue (${fmt.eur(r.levyCollected)}) does not fully cover the requested green bonus (${fmt.eur(r.bonusRequested)}). Dynamic subsidy scaling applied: green bonus reduced to ${(r.scale * 100).toFixed(0)}% so it stays self-funded.</div>`);
  } else if (r.greenAlloc > 0 && r.greyAlloc > 0) {
    notices.push(`<div class="notice info">Grey levy revenue (${fmt.eur(r.levyCollected)}) fully funds the green bonus (${fmt.eur(r.bonusPaid)}).</div>`);
  }
  if (r.poolSurplus > 1e-6) {
    notices.push(`<div class="notice warn">The levy over-collects: the pool retains <b>${fmt.eur(r.poolSurplus)}</b>. Buyers pay the uniform price while sellers receive less in total — the <b>${esc(r.mode)}</b> formulation is not zero-sum here. This is reported, not absorbed.</div>`);
  }
  return notices.join("");
}

function renderClearingResults(r) {
  const panel = $("#panel-results");
  panel.classList.remove("hidden");
  $("#results-sub").textContent =
    `Market ${r.marketId.slice(0, 18)}… · delivery slot ${fmt.slot(r.timeSlot)} · ${r.numTrades} trade objects`;

  const note = r.alreadyCleared
    ? `<div class="notice info">This market slot was already cleared — showing the stored result (idempotent re-trigger).</div>` : "";
  const anchor = `
        <div class="tablecap" style="margin-top:18px">On-chain anchor
          <span class="badge ${r.simulated ? "mock" : "live"}">${r.simulated ? "simulated" : "on-chain"}</span></div>
        <div class="txhash">clearMarket tx: ${esc(r.txHash || "—")}</div>`;
  const limited = r.roundType === "DEMAND_LIMITED" ? "demand-limited (sellers rationed)"
    : r.roundType === "SUPPLY_LIMITED" ? "supply-limited (buyers rationed)" : "balanced";
  const tradesJson = `
    <details class="json">
      <summary>Trade objects as written to the off-chain DB (${r.numTrades}) — blake2b-256 ids, JSON</summary>
      <pre>${esc(JSON.stringify(r.trades, null, 2))}</pre>
    </details>`;

  $("#results-body").innerHTML = `
    ${note}
    <div class="grid2-wide">
      <div>
        <div class="price-hero">
          <div class="amount">${r.clearing.toFixed(2)}<small> ct/kWh</small></div>
          <div class="caption">uniform clearing price · ratio ${fmt.ratio(r.ratio)} · ${roundBadge(r.roundType)}</div>
        </div>
        <div class="statgrid">
          <div class="stat supply"><div class="k">Supply</div><div class="v">${r.supply.toFixed(2)} <small>kWh</small></div></div>
          <div class="stat demand"><div class="k">Demand</div><div class="v">${r.demand.toFixed(2)} <small>kWh</small></div></div>
          <div class="stat"><div class="k">Traded</div><div class="v">${r.traded.toFixed(2)} <small>kWh</small></div></div>
        </div>
        ${supplyDemandBars(r.supply, r.demand, r.traded)}
        ${anchor}
      </div>
      <div>
        <div class="chart-wrap">${sigmoidChartSVG(r.chartCfg, { ratio: r.ratio, price: r.clearing })}</div>
        <div class="chart-note">Clearing price = intersection of the sigmoid with the realized supply/demand ratio.</div>
      </div>
    </div>
    <div class="notice info">Market is <b>${limited}</b>. Traded quantity = min(supply, demand) = ${r.traded.toFixed(2)} kWh. Every participant settles against the community pool at the uniform clearing price.</div>
    <div class="tablecap" style="margin-top:16px">Energy distribution flow</div>
    ${flowDiagram(r)}
    <div class="tablecap" style="margin-top:18px"><span class="swatch supply"></span>Producers — allocation &amp; revenue</div>
    ${producersTable(r)}
    <div class="tablecap" style="margin-top:18px"><span class="swatch demand"></span>Consumers — allocation &amp; cost</div>
    ${consumersTable(r)}
    <div class="tablecap" style="margin-top:18px">Preferred trading-partner matches <small style="font-weight:400;color:var(--muted)">— ${esc(r.order)}</small></div>
    ${preferredMatchesHtml(r)}
    <div class="tablecap" style="margin-top:18px">Energy-type economics <small style="font-weight:400;color:var(--muted)">— ${esc(r.mode)}, applied to ${esc(r.sides === "both" ? "sellers & buyers" : "sellers")}</small></div>
    <div class="statgrid">
      <div class="stat"><div class="k">Grey levy collected</div><div class="v">${fmt.eur(r.levyCollected)}</div></div>
      <div class="stat supply"><div class="k">Green bonus paid</div><div class="v">${fmt.eur(r.bonusPaid)}</div></div>
      <div class="stat"><div class="k">Green / grey volume</div><div class="v">${r.greenAlloc.toFixed(2)} / ${r.greyAlloc.toFixed(2)} <small>kWh</small></div></div>
    </div>
    ${scalingNoticeHtml(r)}
    <p class="balance-summary">Buyers pay <b>${fmt.eur(r.buyersPay)}</b> · Sellers receive <b>${fmt.eur(r.sellersReceive)}</b> · Pool levy surplus <b>${fmt.eur(Math.abs(r.poolSurplus) < 1e-4 ? 0 : r.poolSurplus)}</b> <small>(rates settle at 6 decimals; sub-0.0001 ct residuals are rounding, not economics)</small></p>
    ${tradesJson}`;
  panel.scrollIntoView({ behavior: "smooth", block: "start" });
}

function renderNoTrade(msgHtml) {
  const panel = $("#panel-results");
  panel.classList.remove("hidden");
  $("#results-sub").textContent = "";
  $("#results-body").innerHTML = `<div class="notice warn">${msgHtml}</div>`;
  panel.scrollIntoView({ behavior: "smooth", block: "start" });
}

// Normalize a /trigger-clearing response to the unified result shape.
// Everything about preferences and energy types comes from the backend's
// `preferences` block and the allocation rows — the UI state is only used to
// *configure* the run, never to recompute its economics.
function backendResult(res) {
  const sp = res.sigmoid_params || {};
  const prefs = res.preferences || {};
  const m = prefs.multipliers || {};
  const clearing = +res.clearing_price_ct_per_kwh;
  const row = (a) => ({
    name: a.name, requested: +a.requested_kwh, allocated: +a.allocated_kwh,
    finalRate: a.final_energy_rate != null ? +a.final_energy_rate : clearing,
    matched: !!a.preference_matched, area: a.area_uuid,
  });
  const producers = res.allocations.producers.map(
    (a) => ({ ...row(a), type: a.energy_type === "grey" ? "grey" : "green" }));
  const consumers = res.allocations.consumers.map(row);

  const nameByArea = {};
  for (const p of [...producers, ...consumers]) nameByArea[p.area] = p.name;
  const mutualPairs = (prefs.mutual_pairs || []).map((p) => ({
    ...p,
    label: `${nameByArea[p.offer_area] || p.offer_area} ↔ ${nameByArea[p.bid_area] || p.bid_area}`,
  }));

  return {
    producers, consumers,
    supply: +res.total_supply_kwh, demand: +res.total_demand_kwh,
    traded: +res.traded_quantity_kwh, ratio: +res.ratio,
    clearing, roundType: res.round_type,
    chartCfg: { kUpper: sp.k_upper, kLower: sp.k_lower,
                theta: sp.theta, steepness: sp.steepness },
    // preferred-partner priority
    order: prefs.order || "preferences_first",
    mutualPairs, pairsRationed: !!prefs.pairs_rationed,
    preferentialKwh: mutualPairs.reduce((s, p) => s + (+p.energy_kwh || 0), 0),
    // energy-type multipliers
    mode: m.mode || "multiplicative", sides: m.sides || "seller",
    greenAlloc: +m.green_alloc_kwh || 0, greyAlloc: +m.grey_alloc_kwh || 0,
    levyCollected: +m.levy_collected_ct || 0,
    bonusRequested: +m.bonus_requested_ct || 0,
    bonusPaid: +m.bonus_paid_ct || 0,
    scale: m.scale != null ? +m.scale : 1,
    buyersPay: +m.buyers_pay_ct || 0, sellersReceive: +m.sellers_receive_ct || 0,
    poolSurplus: +m.pool_surplus_ct || 0,
    marketId: res.market_id, timeSlot: res.time_slot, numTrades: res.num_trades,
    txHash: res.tx_hash, simulated: (res.blockchain_mode || "mock") !== "live",
    alreadyCleared: res.status === "already_cleared", trades: res.trades,
  };
}

function renderBackendResults(res) {
  if (res.status === "no_trade") {
    renderNoTrade(`<b>No trade.</b> ${esc(res.message || "")}
      (supply ${(+res.total_supply_kwh).toFixed(2)} kWh, demand ${(+res.total_demand_kwh).toFixed(2)} kWh)
      — all open orders were marked <i>Expired</i>.`);
    return;
  }
  const r = backendResult(res);
  // Report the rules the backend actually applied, not what the UI assumed.
  renderActiveRules(r);
  renderClearingResults(r);
}

// -------------------------------------------------- panel E: post-delivery

function renderActuals(res) {
  const panel = $("#panel-execution");
  if (res.status === "no_trade" || !res.allocations) {
    panel.classList.add("hidden");
    return;
  }
  panel.classList.remove("hidden");
  $("#penalties-body").innerHTML = "";
  $("#exec-status").textContent = "";

  const row = (r, role) => `<div class="prow actual" data-area="${esc(r.area_uuid)}">
      <span class="actual-name">${esc(r.name)}
        <span class="badge ${role === "seller" ? "supply-limited" : "demand-limited"}">${role}</span>
        <small>allocated ${fmt.kwh(r.allocated_kwh)}</small></span>
      <input type="number" min="0" step="0.1" value="${r.allocated_kwh}">
      <span></span>
    </div>`;

  $("#actuals-body").innerHTML = `
    <div class="grid2 participants">
      <div>
        <h3><span class="swatch supply"></span>Actual delivered (kWh)</h3>
        ${res.allocations.producers.map((r) => row(r, "seller")).join("")}
      </div>
      <div>
        <h3><span class="swatch demand"></span>Actual consumed (kWh)</h3>
        ${res.allocations.consumers.map((r) => row(r, "buyer")).join("")}
      </div>
    </div>`;
}

async function runExecution() {
  if (!state.lastMarket) return;
  const status = $("#exec-status");
  setBusy("#run-execution", true);
  try {
    const { market_id, community_uuid, time_slot } = state.lastMarket;
    const measurements = [...document.querySelectorAll(".prow.actual")].map((row) => ({
      community_uuid,
      area_uuid: row.dataset.area,
      time_slot,
      energy_kwh: parseFloat(row.querySelector("input").value) || 0,
    }));

    status.textContent = "posting smart-meter measurements…";
    await api(EP.db, "/measurements", { body: measurements });

    status.textContent = "triggering Execution Node…";
    const res = await api(EP.execution, "/trigger-execution", { body: {
      market_id, community_uuid, time_slot }});
    status.textContent = "done";
    renderPenalties(res);
  } catch (err) {
    status.textContent = "failed";
    showError(err.message);
  } finally {
    setBusy("#run-execution", false);
  }
}

function counterfactualChart(res) {
  const penalized = res.results.filter((r) => r.externality_penalty_ct > 0);
  if (!penalized.length) {
    return `<div class="chart-note" style="margin-top:14px">No externality penalties in this round — counterfactual prices equal the clearing price.</div>`;
  }
  const max = Math.max(res.clearing_price_ct_per_kwh,
    ...penalized.map((r) => r.counterfactual_price_ct_per_kwh)) * 1.1;
  const blocks = penalized.map((r) => {
    const kind = r.role === "seller" ? "withheld" : "underreported";
    return `<div class="cf-block">
      <div class="cf-title">${esc(r.name)} <small>(${kind} ${(+r.externality_kwh).toFixed(2)} kWh — Δ ${(Math.abs(r.counterfactual_price_ct_per_kwh - res.clearing_price_ct_per_kwh)).toFixed(2)} ct/kWh × ${(+res.traded_quantity_kwh).toFixed(2)} kWh traded)</small></div>
      ${hbar("cleared", res.clearing_price_ct_per_kwh, max, "traded", fmt.price(res.clearing_price_ct_per_kwh))}
      ${hbar("counterfactual", r.counterfactual_price_ct_per_kwh, max, "cf", fmt.price(r.counterfactual_price_ct_per_kwh))}
    </div>`;
  }).join("");
  return `<div class="tablecap" style="margin-top:20px">Counterfactual prices (had the deviation not occurred)</div>${blocks}`;
}

function renderPenalties(res) {
  const body = $("#penalties-body");
  if (res.status !== "executed") {
    body.innerHTML = `<div class="notice warn">${esc(res.message || res.status)}</div>`;
    return;
  }
  const rows = res.results.map((r) => {
    const penalized = r.total_penalty_ct > 0;
    const meterNote = r.measurement_found ? "" :
      ` <small title="no meter data — assumed delivered as traded">*</small>`;
    return `<tr class="${penalized ? "penalized" : ""}">
      <td>${esc(r.name)}${meterNote}</td>
      <td>${esc(r.role)}</td>
      <td class="num">${fmt.kwh(r.traded_kwh)}</td>
      <td class="num">${fmt.kwh(r.actual_kwh)}</td>
      <td class="num">${fmt.ct(r.shortfall_penalty_ct)}</td>
      <td class="num">${fmt.ct(r.externality_penalty_ct)}</td>
      <td class="num"><b>${fmt.ct(r.total_penalty_ct)}</b></td>
    </tr>`;
  }).join("");

  body.innerHTML = `
    <div class="statgrid" style="margin-top:20px">
      <div class="stat"><div class="k">Round type</div><div class="v" style="font-size:16px">${roundBadge(res.round_type)}</div></div>
      <div class="stat price"><div class="k">Clearing price</div><div class="v">${(+res.clearing_price_ct_per_kwh).toFixed(2)} <small>ct/kWh</small></div></div>
      <div class="stat"><div class="k">K_sho (γ·K_upper)</div><div class="v">${(+res.penalty_params.k_sho_ct_per_kwh).toFixed(2)} <small>ct/kWh</small></div></div>
      <div class="stat ${res.total_penalties_ct > 0 ? "penalty" : ""}"><div class="k">Total penalties</div><div class="v" style="color:${res.total_penalties_ct > 0 ? "#dc2626" : "inherit"}">${(+res.total_penalties_ct).toFixed(2)} <small>ct</small></div></div>
    </div>
    <table class="data">
      <thead><tr><th>Name</th><th>Role</th><th class="num">Traded</th><th class="num">Actual</th>
        <th class="num">Shortfall penalty</th><th class="num">Externality penalty</th><th class="num">Total penalty</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>
    ${counterfactualChart(res)}`;
}

// ------------------- panel C: preferences & energy-type multipliers
//
// Both mechanisms (Guide §4.4 step 6 / §6) live in the Clearing Node. The
// demo runs the fixed combination preferences_first / multiplicative /
// seller; the panel only collects the numeric parameters, which travel with
// the trigger next to `sigmoid_params`. The *rules* are deliberately not
// selectable here: supervisor question B-04 (allocation order, multiplier
// side) is unresolved, so answering it differently must stay a configuration
// change. The backend keeps every variant — see configuration.yaml and the
// PREFERENCE_ORDER / MULTIPLIER_MODE / MULTIPLIER_SIDES environment vars.

function readPreferenceInputs() {
  state.preferences = {
    greenMultiplier: parseFloat($("#pref-green").value) || 0,
    greyLevy: parseFloat($("#pref-greylevy").value) || 0,
    levyCap: parseFloat($("#pref-levycap").value) || 0,
  };
  return state.preferences;
}

// Human-readable labels for what the backend reports it actually did.
const RULE_LABELS = {
  order: { preferences_first: "preferred pairs first",
           pro_rata_first: "pro-rata first (pairs flagged only)" },
  mode: { multiplicative: "multiplicative multipliers",
          additive: "additive multipliers" },
  sides: { seller: "applied to sellers only",
           both: "applied to sellers &amp; buyers" },
};

const ruleLabel = (kind, value) =>
  RULE_LABELS[kind][value] || esc(String(value));

// Read-only status line in panel C: the rules the backend reported it
// applied. `r` is the normalized last clearing result, or null before the
// first run — then the line stays empty rather than asserting defaults the
// UI has not been told. (Which variants exist and how to switch them is
// documented in the README, not repeated in the panel.)
function renderActiveRules(r) {
  $("#pref-active").innerHTML = r
    ? `${ruleLabel("order", r.order)} · ${ruleLabel("mode", r.mode)} · ${ruleLabel("sides", r.sides)}`
    : "";
}

function trunc(s, n) { s = String(s); return s.length > n ? s.slice(0, n - 1) + "…" : s; }

function fillCell(rate, cls) {
  const pct = Math.max(0, Math.min(100, rate * 100));
  return `<div class="fillbar"><div class="fillbar-track"><div class="fillbar-fill ${cls}" style="width:${pct.toFixed(1)}%"></div></div><div class="fillbar-pct">${pct.toFixed(0)}%</div></div>`;
}

// Sankey-style flow: producers -> community pool -> consumers.
function flowDiagram(r) {
  const P = r.producers, C = r.consumers;
  const W = 1080, boxW = 248, boxH = 56, poolW = 120, poolH = 150, padTop = 16, gap = 24;
  const n = Math.max(P.length, C.length, 1);
  const H = Math.max(n * (boxH + gap) - gap + padTop * 2, poolH + 40);
  const leftX = 24, rightX = W - 24 - boxW;
  const poolX = (W - poolW) / 2, poolY = (H - poolH) / 2, centerY = H / 2;
  const maxA = Math.max(0.001,
    ...P.map((p) => p.allocated || 0), ...C.map((c) => c.allocated || 0));
  const ys = (count) => {
    if (count <= 1) return [(H - boxH) / 2];
    const usable = H - padTop * 2 - boxH;
    return Array.from({ length: count }, (_, i) => padTop + usable * i / (count - 1));
  };
  const pys = ys(P.length), cys = ys(C.length);
  const bandW = (a) => 3 + (a / maxA) * 22;
  let bands = "", boxes = "";

  P.forEach((p, i) => {
    const a = p.allocated || 0;
    const y0 = pys[i] + boxH / 2, x0 = leftX + boxW;
    const y1 = poolY + poolH * (i + 0.5) / P.length, x1 = poolX;
    const stroke = p.type === "grey" ? "#9ca3af" : "#34a877";
    bands += `<path d="M${x0},${y0} C${(x0 + x1) / 2},${y0} ${(x0 + x1) / 2},${y1} ${x1},${y1}" fill="none" stroke="${stroke}" stroke-width="${bandW(a)}" stroke-opacity="0.55"/>`;
    const bg = p.type === "grey" ? "#f3f4f6" : "#d6f5e8";
    const bd = p.type === "grey" ? "#9ca3af" : "#34a877";
    const ink = p.type === "grey" ? "#586173" : "#065f46";
    boxes += `<g><rect x="${leftX}" y="${pys[i]}" width="${boxW}" height="${boxH}" rx="9" fill="${bg}" stroke="${bd}" stroke-width="1.6"/>`
      + `<text x="${leftX + 14}" y="${pys[i] + 23}" font-size="14" font-weight="700" fill="${ink}">${esc(trunc(p.name, 22))}</text>`
      + `<text x="${leftX + 14}" y="${pys[i] + 42}" font-size="12" fill="${ink}" opacity="0.85">${a.toFixed(2)} kWh · ${p.type}</text></g>`;
  });
  C.forEach((c, i) => {
    const a = c.allocated || 0;
    const y0 = cys[i] + boxH / 2, x0 = rightX;
    const y1 = poolY + poolH * (i + 0.5) / C.length, x1 = poolX + poolW;
    bands += `<path d="M${x1},${y1} C${(x0 + x1) / 2},${y1} ${(x0 + x1) / 2},${y0} ${x0},${y0}" fill="none" stroke="#3b82c4" stroke-width="${bandW(a)}" stroke-opacity="0.5"/>`;
    boxes += `<g><rect x="${rightX}" y="${cys[i]}" width="${boxW}" height="${boxH}" rx="9" fill="#dbeffc" stroke="#3b82c4" stroke-width="1.6"/>`
      + `<text x="${rightX + 14}" y="${cys[i] + 23}" font-size="14" font-weight="700" fill="#0c4a6e">${esc(trunc(c.name, 22))}</text>`
      + `<text x="${rightX + 14}" y="${cys[i] + 42}" font-size="12" fill="#0c4a6e" opacity="0.85">${a.toFixed(2)} kWh</text></g>`;
  });
  const pool = `<rect x="${poolX}" y="${poolY}" width="${poolW}" height="${poolH}" rx="12" fill="#ece9fd" stroke="#7c6df2" stroke-width="2"/>`
    + `<text x="${poolX + poolW / 2}" y="${centerY - 4}" text-anchor="middle" font-size="16" font-weight="800" fill="#4f46e5">POOL</text>`
    + `<text x="${poolX + poolW / 2}" y="${centerY + 16}" text-anchor="middle" font-size="12" fill="#5b54b8">${r.traded.toFixed(2)} kWh</text>`;
  return `<div class="flow-wrap"><svg viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg" width="100%" font-family="Segoe UI, sans-serif">${bands}${pool}${boxes}</svg></div>`;
}

// ---------------------------------------------------------- dev auto-reload

// scripts/serve_ui.py exposes /__version (latest mtime of the ui/ files).
// Poll it and reload as soon as it changes, so an open tab can never keep
// showing stale code. Other servers (e.g. the nginx Docker image) don't
// have the endpoint — the watcher then disables itself. While the dev
// server is down the poll just fails silently and resumes on restart, so
// a stale tab also self-heals once the stack is started again.
function startDevReload() {
  if (!/^https?:$/.test(location.protocol)) return;
  let current = null;
  const timer = setInterval(async () => {
    let resp;
    try {
      resp = await fetch("__version", { cache: "no-store" });
    } catch { return; }                              // server down — retry later
    if (!resp.ok) { clearInterval(timer); return; }  // not the dev server
    const stamp = await resp.text();
    if (current === null) current = stamp;
    else if (stamp !== current) location.reload();
  }, 3000);
}

// ------------------------------------------------------------ service pings

async function pingServices() {
  const targets = [
    ["db", `${EP.db}/health_check`],
    ["clearing", `${EP.clearing}/health`],
    ["execution", `${EP.execution}/health`],
  ];
  await Promise.allSettled(targets.map(async ([svc, url]) => {
    const pill = document.querySelector(`.pill[data-svc="${svc}"]`);
    try {
      const resp = await fetch(url, { signal: AbortSignal.timeout(4000) });
      pill.classList.toggle("up", resp.ok);
      pill.classList.toggle("down", !resp.ok);
    } catch {
      pill.classList.remove("up");
      pill.classList.add("down");
    }
  }));
}

// ------------------------------------------------------------------- init

document.addEventListener("DOMContentLoaded", () => {
  $("#ep-db").textContent = EP.db;
  $("#ep-clearing").textContent = EP.clearing;
  $("#ep-execution").textContent = EP.execution;

  renderParticipants();
  renderConfigChart();

  for (const id of ["cfg-name", "cfg-kupper", "cfg-klower", "cfg-theta", "cfg-steepness"]) {
    document.getElementById(id).addEventListener("input", () => {
      readConfigInputs();
      renderLiveStats();
    });
  }

  bindParticipantEvents("#producer-rows", () => state.producers);
  bindParticipantEvents("#consumer-rows", () => state.consumers);
  $("#add-producer").addEventListener("click", () => {
    state.producers.push({ id: ++idSeq, name: `Producer ${state.producers.length + 1}`, energy: 1.0, type: "green", partner: null });
    renderParticipants();
  });
  $("#add-consumer").addEventListener("click", () => {
    state.consumers.push({ id: ++idSeq, name: `Consumer ${state.consumers.length + 1}`, energy: 1.0, partner: null });
    renderParticipants();
  });

  for (const id of ["pref-green", "pref-greylevy", "pref-levycap"]) {
    document.getElementById(id).addEventListener("input", readPreferenceInputs);
  }
  // Empty until the first clearing reports what the backend actually ran.
  renderActiveRules(null);

  $("#run-clearing").addEventListener("click", runClearing);
  $("#run-execution").addEventListener("click", runExecution);

  pingServices();
  setInterval(pingServices, 20000);
  startDevReload();
});
