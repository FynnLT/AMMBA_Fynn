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
  config: { name: "Community 1", kUpper: 28.5, kLower: 8.0, theta: 1.0, steepness: 2.5 },
  // Defaults reproduce the implementation guide's example:
  // supply 12.5 kWh vs demand 10 kWh -> ratio 1.25 -> ~15.15 ct/kWh
  producers: [
    { id: ++idSeq, name: "Rooftop PV A", energy: 5.0 },
    { id: ++idSeq, name: "Rooftop PV B", energy: 3.5 },
    { id: ++idSeq, name: "Community Battery", energy: 4.0 },
  ],
  consumers: [
    { id: ++idSeq, name: "Household 1", energy: 4.5 },
    { id: ++idSeq, name: "Household 2", energy: 3.0 },
    { id: ++idSeq, name: "Bakery", energy: 2.5 },
  ],
  lastClearing: null,   // response of /trigger-clearing
  lastMarket: null,     // {market_id, community_uuid, time_slot}
  areaByName: {},       // participant name -> area_uuid for the last run
};

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

function renderConfigChart() {
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
}

// --------------------------------------------------- panel B: participants

function participantRow(p) {
  return `<div class="prow" data-id="${p.id}">
    <input type="text" data-field="name" value="${esc(p.name)}" placeholder="name">
    <input type="number" data-field="energy" value="${p.energy}" min="0" step="0.1" placeholder="kWh">
    <button class="remove" title="remove">✕</button>
  </div>`;
}

function renderParticipants() {
  $("#producer-rows").innerHTML = state.producers.map(participantRow).join("");
  $("#consumer-rows").innerHTML = state.consumers.map(participantRow).join("");
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
    if (field === "name") item.name = ev.target.value;
    if (field === "energy") item.energy = parseFloat(ev.target.value) || 0;
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

    const orders = [
      // AMMBA is a uniform-price auction: order energy_rate limits are not
      // used by the clearing; offers carry the feed-in floor, bids the
      // retail cap.
      ...producers.map((p) => ({
        order_type: "Offer", created_by: p.name,
        area_uuid: producerAreas[p.id], market_id: market.market_id,
        time_slot: timeSlot, energy: +p.energy, energy_rate: c.kLower,
      })),
      ...consumers.map((x) => ({
        order_type: "Bid", created_by: x.name,
        area_uuid: consumerAreas[x.id], market_id: market.market_id,
        time_slot: timeSlot, energy: +x.energy, energy_rate: c.kUpper,
      })),
    ];
    if (orders.length) {
      status.textContent = `posting ${orders.length} order(s)…`;
      await api(EP.db, "/orders-normalized", { body: orders });
    }

    status.textContent = "triggering Clearing Node…";
    const result = await api(EP.clearing, "/trigger-clearing", { body: {
      market_id: market.market_id,
      community_uuid: communityUuid,
      time_slot: timeSlot,
      community_name: c.name,
      sigmoid_params: { k_upper: c.kUpper, k_lower: c.kLower,
                        theta: c.theta, steepness: c.steepness },
    }});

    state.lastClearing = result;
    state.lastMarket = { market_id: market.market_id,
                         community_uuid: communityUuid, time_slot: timeSlot };
    status.textContent = `done — market ${market.market_id.slice(0, 12)}…`;
    renderResults(result);
    renderActuals(result);
  } catch (err) {
    status.textContent = "failed";
    showError(err.message);
  } finally {
    setBusy("#run-clearing", false);
  }
}

// ------------------------------------------------------- panel D: results

function roundBadge(roundType) {
  const cls = (roundType || "").toLowerCase().replace("_", "-");
  return `<span class="badge ${cls}">${esc((roundType || "").replace("_", "-"))}</span>`;
}

function allocationTable(rows, kind) {
  const qtyHead = kind === "producer" ? "Offered" : "Demanded";
  const valHead = kind === "producer" ? "Revenue" : "Cost";
  const body = rows.map((r) => `<tr>
      <td>${esc(r.name)}</td>
      <td class="num">${fmt.kwh(r.requested_kwh)}</td>
      <td class="num">${fmt.kwh(r.allocated_kwh)}</td>
      <td class="num">${fmt.pct(r.fill_rate)}</td>
      <td class="num">${fmt.ct(r.value_ct)}</td>
    </tr>`).join("");
  const totals_ = rows.reduce((a, r) => ({
    req: a.req + r.requested_kwh, alloc: a.alloc + r.allocated_kwh,
    val: a.val + r.value_ct }), { req: 0, alloc: 0, val: 0 });
  return `<table class="data">
    <thead><tr><th>Name</th><th class="num">${qtyHead}</th><th class="num">Allocated</th>
      <th class="num">Fill rate</th><th class="num">${valHead}</th></tr></thead>
    <tbody>${body}</tbody>
    <tfoot><tr><td>Total</td><td class="num">${fmt.kwh(totals_.req)}</td>
      <td class="num">${fmt.kwh(totals_.alloc)}</td><td class="num"></td>
      <td class="num">${fmt.ct(totals_.val)}</td></tr></tfoot>
  </table>`;
}

function renderResults(res) {
  const panel = $("#panel-results");
  panel.classList.remove("hidden");
  const sub = $("#results-sub");
  const body = $("#results-body");

  if (res.status === "no_trade") {
    sub.textContent = "";
    body.innerHTML = `<div class="notice warn"><b>No trade.</b> ${esc(res.message || "")}
      (supply ${(+res.total_supply_kwh).toFixed(2)} kWh, demand ${(+res.total_demand_kwh).toFixed(2)} kWh)
      — all open orders were marked <i>Expired</i>.</div>`;
    $("#panel-execution").classList.add("hidden");
    panel.scrollIntoView({ behavior: "smooth", block: "start" });
    return;
  }

  const sp = res.sigmoid_params || {};
  const chartCfg = { kUpper: sp.k_upper, kLower: sp.k_lower,
                     theta: sp.theta, steepness: sp.steepness };
  const simulated = (res.blockchain_mode || "mock") !== "live";
  const note = res.status === "already_cleared"
    ? `<div class="notice info">This market slot was already cleared — showing the stored result (idempotent re-trigger).</div>` : "";

  sub.textContent = `Market ${res.market_id.slice(0, 18)}… · delivery slot ${fmt.slot(res.time_slot)} · ${res.num_trades} trade objects`;
  body.innerHTML = `
    ${note}
    <div class="grid2-wide">
      <div>
        <div class="price-hero">
          <div class="amount">${(+res.clearing_price_ct_per_kwh).toFixed(2)}<small> ct/kWh</small></div>
          <div class="caption">uniform clearing price · ratio ${fmt.ratio(res.ratio)} · ${roundBadge(res.round_type)}</div>
        </div>
        <div class="statgrid">
          <div class="stat supply"><div class="k">Supply</div><div class="v">${(+res.total_supply_kwh).toFixed(2)} <small>kWh</small></div></div>
          <div class="stat demand"><div class="k">Demand</div><div class="v">${(+res.total_demand_kwh).toFixed(2)} <small>kWh</small></div></div>
          <div class="stat"><div class="k">Traded</div><div class="v">${(+res.traded_quantity_kwh).toFixed(2)} <small>kWh</small></div></div>
        </div>
        ${supplyDemandBars(res.total_supply_kwh, res.total_demand_kwh, res.traded_quantity_kwh)}
        <div class="tablecap" style="margin-top:18px">On-chain anchor
          <span class="badge ${simulated ? "mock" : "live"}">${simulated ? "simulated" : "on-chain"}</span></div>
        <div class="txhash">clearMarket tx: ${esc(res.tx_hash || "—")}</div>
      </div>
      <div>
        <div class="chart-wrap">${sigmoidChartSVG(chartCfg, { ratio: res.ratio, price: res.clearing_price_ct_per_kwh })}</div>
        <div class="chart-note">Clearing price = intersection of the sigmoid with the realized supply/demand ratio.</div>
      </div>
    </div>
    <div class="grid2">
      <div>
        <div class="tablecap"><span class="swatch supply"></span>Producers</div>
        ${allocationTable(res.allocations.producers, "producer")}
      </div>
      <div>
        <div class="tablecap"><span class="swatch demand"></span>Consumers</div>
        ${allocationTable(res.allocations.consumers, "consumer")}
      </div>
    </div>
    <details class="json">
      <summary>Trade objects as written to the off-chain DB (${res.num_trades}) — blake2b-256 ids, JSON</summary>
      <pre>${esc(JSON.stringify(res.trades, null, 2))}</pre>
    </details>`;
  panel.scrollIntoView({ behavior: "smooth", block: "start" });
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
    await api(EP.db, "/asset_measurements", { body: measurements });

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

// ------------------------------------------------------------ service pings

async function pingServices() {
  const targets = [
    ["db", `${EP.db}/health_check`],
    ["clearing", `${EP.clearing}/health`],
    ["execution", `${EP.execution}/health`],
  ];
  for (const [svc, url] of targets) {
    const pill = document.querySelector(`.pill[data-svc="${svc}"]`);
    try {
      const resp = await fetch(url, { signal: AbortSignal.timeout(4000) });
      pill.classList.toggle("up", resp.ok);
      pill.classList.toggle("down", !resp.ok);
    } catch {
      pill.classList.remove("up");
      pill.classList.add("down");
    }
  }
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
    state.producers.push({ id: ++idSeq, name: `Producer ${state.producers.length + 1}`, energy: 1.0 });
    renderParticipants();
  });
  $("#add-consumer").addEventListener("click", () => {
    state.consumers.push({ id: ++idSeq, name: `Consumer ${state.consumers.length + 1}`, energy: 1.0 });
    renderParticipants();
  });

  $("#run-clearing").addEventListener("click", runClearing);
  $("#run-execution").addEventListener("click", runExecution);

  pingServices();
  setInterval(pingServices, 20000);
});
