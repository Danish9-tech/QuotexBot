"""
Flask blueprint for the live verification dashboard.

Routes:
  GET  /                    -> redirect to /dashboard
  GET  /dashboard           -> HTML page
  GET  /api/metrics         -> JSON snapshot of all metrics
  POST /api/kill_switch/pause   -> pause trading (token required)
  POST /api/kill_switch/resume  -> resume trading (token required)
  GET  /api/kill_switch/state   -> current pause/resume state
"""

import os
import asyncio
import logging
import threading
from functools import wraps
from typing import Any  # noqa: F401  (kept for future use)

from flask import Blueprint, jsonify, redirect, render_template_string, request, abort

import motor.motor_asyncio
from dotenv import load_dotenv

from . import metrics, alerts

load_dotenv()
logger = logging.getLogger("dashboard.web")

MONGO_URI = os.getenv("MONGO_URI", "")
DASHBOARD_TOKEN = os.getenv("DASHBOARD_TOKEN", "")

bp = Blueprint("dashboard", __name__)

# Background event loop runs in a dedicated thread. motor is async — calling
# `asyncio.run` from a sync Flask handler in the main thread will collide with
# any other event loop on the same thread. We submit coroutines to the worker
# thread instead, so handlers stay non-blocking and motor's connection pool is
# reused across requests.
_state_lock = threading.Lock()
_loop: asyncio.AbstractEventLoop | None = None
_loop_thread: threading.Thread | None = None
_client: motor.motor_asyncio.AsyncIOMotorClient | None = None
_loop_ready = threading.Event()


def _loop_main():
    global _client, _loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URI)
    with _state_lock:
        _client = client
        _loop = loop
    _loop_ready.set()
    loop.run_forever()


def _ensure_loop_started():
    global _loop_thread
    if _loop_thread is not None and _loop_thread.is_alive():
        return
    _loop_thread = threading.Thread(target=_loop_main, name="dashboard-async", daemon=True)
    _loop_thread.start()
    if not _loop_ready.wait(timeout=10):
        raise RuntimeError("dashboard async loop did not start within 10s")


def _submit(coro) -> Any:
    """Submit a coroutine to the background loop and wait for its result."""
    _ensure_loop_started()
    future = asyncio.run_coroutine_threadsafe(coro, _loop)
    return future.result(timeout=30)


def get_db():
    _ensure_loop_started()
    with _state_lock:
        client = _client
    if client is None:
        raise RuntimeError("dashboard async client not initialized")
    return client["quotexTraderBot"]


def require_token(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        provided = request.headers.get("X-Dashboard-Token", "")
        if not DASHBOARD_TOKEN:
            abort(503, description="DASHBOARD_TOKEN not configured")
        if provided != DASHBOARD_TOKEN:
            abort(401, description="bad token")
        return f(*args, **kwargs)
    return wrapper


# ----- Routes -----

@bp.route("/")
def root():
    return redirect("/dashboard", code=302)


@bp.route("/dashboard")
def dashboard_page():
    return render_template_string(_DASHBOARD_HTML)


@bp.route("/api/metrics")
def api_metrics():
    db = get_db()
    snapshot = _submit(metrics.build_full_metrics(db))
    return jsonify(snapshot)


@bp.route("/api/trades")
def api_trades():
    """Recent N trades for the dashboard table. Capped at 500."""
    try:
        limit = int(request.args.get("limit", "50"))
    except ValueError:
        limit = 50
    limit = max(1, min(500, limit))
    db = get_db()
    trades = _submit(metrics.rolling_window(db, n=limit))
    return jsonify(trades)


@bp.route("/api/kill_switch/state")
def api_kill_switch_state():
    db = get_db()
    snapshot = _submit(metrics.build_full_metrics(db))
    state = alerts.evaluate_kill_switch(snapshot)
    return jsonify({
        "triggered": state["triggered"],
        "reasons": state["reasons"],
        "values": state["values"],
        "thresholds": state["thresholds"],
        "accounts": snapshot.get("service_status", {}).get("accounts", []),
    })


@bp.route("/api/kill_switch/pause", methods=["POST"])
@require_token
def api_pause():
    db = get_db()
    n = _submit(alerts.write_kill_switch(db, service_status=False))
    logger.info(f"Manual pause: flipped {n} account(s) to service_status=False")
    return jsonify({"ok": True, "modified": n, "service_status": False})


@bp.route("/api/kill_switch/resume", methods=["POST"])
@require_token
def api_resume():
    db = get_db()
    n = _submit(alerts.write_kill_switch(db, service_status=True))
    logger.info(f"Manual resume: flipped {n} account(s) to service_status=True")
    return jsonify({"ok": True, "modified": n, "service_status": True})


# ----- HTML -----

_DASHBOARD_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Quotex Bot — Live Verification Dashboard</title>
<style>
  :root {
    --bg: #0f1115;
    --panel: #1a1d24;
    --panel-2: #232730;
    --fg: #e7e9ee;
    --muted: #8a92a3;
    --green: #2ecc71;
    --red: #e74c3c;
    --yellow: #f1c40f;
    --border: #2c313c;
    --accent: #5b9bff;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; padding: 16px;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", sans-serif;
    background: var(--bg); color: var(--fg);
    font-size: 14px; line-height: 1.4;
  }
  h1 { margin: 0 0 4px 0; font-size: 20px; font-weight: 600; }
  h2 { margin: 24px 0 8px 0; font-size: 14px; font-weight: 600;
       text-transform: uppercase; letter-spacing: 0.5px; color: var(--muted); }
  .subtitle { color: var(--muted); font-size: 12px; margin-bottom: 16px; }
  .banner {
    padding: 12px 16px; border-radius: 8px; margin-bottom: 16px;
    font-weight: 600; display: flex; align-items: center; justify-content: space-between;
    flex-wrap: wrap; gap: 8px;
  }
  .banner.armed { background: rgba(46, 204, 113, 0.12); border: 1px solid var(--green); color: var(--green); }
  .banner.triggered { background: rgba(231, 76, 60, 0.12); border: 1px solid var(--red); color: var(--red); }
  .banner .reasons { font-weight: 400; font-size: 12px; margin-top: 4px; }
  .btn {
    background: var(--panel-2); color: var(--fg);
    border: 1px solid var(--border); padding: 6px 12px;
    border-radius: 6px; cursor: pointer; font-size: 12px;
    font-weight: 500; font-family: inherit;
  }
  .btn:hover { background: var(--border); }
  .btn.danger { border-color: var(--red); color: var(--red); }
  .btn.success { border-color: var(--green); color: var(--green); }
  .grid {
    display: grid; gap: 12px;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    margin-bottom: 16px;
  }
  .tile {
    background: var(--panel); border: 1px solid var(--border);
    border-radius: 8px; padding: 14px;
  }
  .tile .label { font-size: 11px; text-transform: uppercase;
                 letter-spacing: 0.5px; color: var(--muted); margin-bottom: 6px; }
  .tile .value { font-size: 22px; font-weight: 700; font-variant-numeric: tabular-nums; }
  .tile .sub { font-size: 12px; color: var(--muted); margin-top: 4px; }
  .green { color: var(--green); }
  .red { color: var(--red); }
  .yellow { color: var(--yellow); }
  .muted { color: var(--muted); }
  table { width: 100%; border-collapse: collapse; font-size: 12px;
          background: var(--panel); border-radius: 8px; overflow: hidden; }
  th, td { padding: 6px 10px; text-align: left; border-bottom: 1px solid var(--border); }
  th { background: var(--panel-2); color: var(--muted);
       text-transform: uppercase; font-size: 10px; letter-spacing: 0.5px; }
  tr:last-child td { border-bottom: 0; }
  td.num { text-align: right; font-variant-numeric: tabular-nums; }
  .pill { display: inline-block; padding: 1px 8px; border-radius: 999px;
          font-size: 10px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }
  .pill.prime { background: rgba(46, 204, 113, 0.18); color: var(--green); }
  .pill.profitable { background: rgba(91, 155, 255, 0.18); color: var(--accent); }
  .pill.loss { background: rgba(231, 76, 60, 0.18); color: var(--red); }
  .pill.empty { background: rgba(138, 146, 163, 0.18); color: var(--muted); }
  .trades { max-height: 400px; overflow-y: auto; }
  .footer { margin-top: 24px; padding-top: 12px; border-top: 1px solid var(--border);
            color: var(--muted); font-size: 11px; }
  .pulse { display: inline-block; width: 8px; height: 8px; border-radius: 50%;
           background: var(--green); margin-right: 6px;
           animation: pulse 1.5s ease-in-out infinite; }
  @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }
  .inversion { display: flex; gap: 16px; align-items: baseline; }
  .inversion .pair { font-size: 13px; }
  .inversion .pair span { font-weight: 700; font-size: 18px; }
  .err { background: rgba(231, 76, 60, 0.12); border: 1px solid var(--red);
         color: var(--red); padding: 12px; border-radius: 8px; margin-bottom: 16px; }
</style>
</head>
<body>
  <h1>🩺 Quotex Bot — Live Verification Dashboard</h1>
  <div class="subtitle">
    <span class="pulse"></span>
    <span id="last-updated">loading…</span> · auto-refresh every 5s
  </div>

  <div id="banner" class="banner armed">
    <div>
      <div id="banner-title">ARMED — checking…</div>
      <div id="banner-reasons" class="reasons"></div>
    </div>
    <div>
      <button class="btn danger" id="btn-pause">⏸ Pause bot</button>
      <button class="btn success" id="btn-resume">▶ Resume bot</button>
    </div>
  </div>

  <div id="error"></div>

  <div class="grid">
    <div class="tile">
      <div class="label">Win Rate (last 50)</div>
      <div class="value" id="t-wr">—</div>
      <div class="sub" id="t-wr-sub">vs 54.05% breakeven</div>
    </div>
    <div class="tile">
      <div class="label">Daily PnL</div>
      <div class="value" id="t-daily">—</div>
      <div class="sub" id="t-daily-sub"></div>
    </div>
    <div class="tile">
      <div class="label">Drawdown</div>
      <div class="value" id="t-dd">—</div>
      <div class="sub" id="t-dd-sub">current / max</div>
    </div>
    <div class="tile">
      <div class="label">Last 10 trades</div>
      <div class="value" id="t-streak">—</div>
      <div class="sub" id="t-streak-sub"></div>
    </div>
    <div class="tile">
      <div class="label">Inversion test</div>
      <div class="inversion">
        <div class="pair"><span id="t-inv-orig">—</span> orig</div>
        <div class="pair muted">vs</div>
        <div class="pair"><span id="t-inv-inv">—</span> inv</div>
      </div>
      <div class="sub" id="t-inv-sub">edge vs flipped signal</div>
    </div>
  </div>

  <h2>Hourly edge (UTC)</h2>
  <table>
    <thead>
      <tr>
        <th>Hour</th><th class="num">Trades</th><th class="num">Wins</th>
        <th class="num">Losses</th><th class="num">WR%</th>
        <th class="num">Profit</th><th>Label</th>
      </tr>
    </thead>
    <tbody id="hourly-body"></tbody>
  </table>

  <h2>Per-asset kill candidates (≥20 trades, ≤45% WR)</h2>
  <table>
    <thead>
      <tr>
        <th>Asset</th><th class="num">Trades</th><th class="num">Wins</th>
        <th class="num">Losses</th><th class="num">WR%</th><th class="num">Profit</th>
      </tr>
    </thead>
    <tbody id="kill-body">
      <tr><td colspan="6" class="muted">none</td></tr>
    </tbody>
  </table>

  <h2>Linked accounts</h2>
  <table>
    <thead>
      <tr><th>Account</th><th>Mode</th><th>Status</th><th class="num">Stake</th></tr>
    </thead>
    <tbody id="accounts-body">
      <tr><td colspan="4" class="muted">none</td></tr>
    </tbody>
  </table>

  <h2>Recent trades (last 50)</h2>
  <div class="trades">
    <table>
      <thead>
        <tr>
          <th>Time (UTC)</th><th>Asset</th><th>Dir</th>
          <th>Result</th><th class="num">Profit</th><th class="num">Stake</th>
          <th>Reason</th>
        </tr>
      </thead>
      <tbody id="trades-body"></tbody>
    </table>
  </div>

  <div class="footer">
    Built from <code>trade_history</code> + <code>trade_settings</code> in MongoDB.
    Kill switch flips <code>service_status</code> on <code>trade_settings</code>;
    bot.py picks it up on its next asset iteration.
    Breakeven is 54.05% at an 85% payout. Inversion test: if "inv" ≈ "orig",
    the strategy is a coin flip and no indicator tweak will fix that.
  </div>

<script>
const TOKEN = "";  // Set via prompt() on first action; persisted in sessionStorage.

function fmtPnl(n) {
  if (n === null || n === undefined) return "—";
  const s = n >= 0 ? "+" : "−";
  return s + "$" + Math.abs(n).toFixed(2);
}
function fmtPct(n) { return (n === null || n === undefined) ? "—" : n.toFixed(2) + "%"; }
function fmtTime(iso) {
  if (!iso) return "—";
  const d = new Date(iso);
  if (isNaN(d)) return iso;
  return d.toISOString().slice(0, 19).replace("T", " ") + "Z";
}
function colorClass(n, threshold) {
  if (n === null || n === undefined) return "";
  return n >= threshold ? "green" : "red";
}
function pillClass(label) {
  if (label === "PRIME") return "prime";
  if (label === "PROFITABLE") return "profitable";
  if (label === "LOSS") return "loss";
  return "empty";
}
function escHtml(s) {
  if (s === null || s === undefined) return "";
  return String(s)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
}

async function fetchJSON(url, opts = {}) {
  const headers = Object.assign({ "Accept": "application/json" }, opts.headers || {});
  if (TOKEN) headers["X-Dashboard-Token"] = TOKEN;
  const r = await fetch(url, { ...opts, headers });
  if (!r.ok) {
    if (r.status === 401) throw new Error("bad token");
    throw new Error("HTTP " + r.status);
  }
  return await r.json();
}

function ensureToken() {
  if (TOKEN) return TOKEN;
  let t = sessionStorage.getItem("dashboard_token");
  if (!t) {
    t = window.prompt("Dashboard token (X-Dashboard-Token header):");
    if (!t) return null;
    sessionStorage.setItem("dashboard_token", t);
  }
  window.TOKEN = t;
  return t;
}

async function refresh() {
  const errEl = document.getElementById("error");
  errEl.innerHTML = "";
  try {
    const m = await fetchJSON("/api/metrics");
    renderMetrics(m);
  } catch (e) {
    errEl.innerHTML = `<div class="err">Failed to load metrics: ${escHtml(e.message)}</div>`;
  }
  try {
    const s = await fetchJSON("/api/kill_switch/state");
    renderKillState(s);
  } catch (e) { /* state endpoint is best-effort */ }
}

function renderMetrics(m) {
  document.getElementById("last-updated").textContent =
    "last updated " + fmtTime(m.generated_at);

  // Win rate tile
  const r = m.rolling_50 || {};
  const wr = r.win_rate;
  const wrEl = document.getElementById("t-wr");
  wrEl.textContent = fmtPct(wr);
  wrEl.className = "value " + colorClass(wr, r.breakeven);
  const gap = r.gap;
  document.getElementById("t-wr-sub").innerHTML =
    `breakeven ${r.breakeven}% · gap <span class="${gap >= 0 ? 'green' : 'red'}">${gap >= 0 ? '+' : ''}${gap.toFixed(2)}pp</span>`;

  // Daily PnL
  const d = m.daily_pnl || {};
  const dEl = document.getElementById("t-daily");
  dEl.textContent = fmtPnl(d.net_pnl);
  dEl.className = "value " + (d.net_pnl >= 0 ? "green" : "red");
  document.getElementById("t-daily-sub").textContent =
    `since ${(d.since_utc || "").slice(0, 10)} UTC`;

  // Drawdown
  const dd = m.drawdown || {};
  const ddEl = document.getElementById("t-dd");
  ddEl.textContent = "$" + dd.current_drawdown.toFixed(2);
  ddEl.className = "value " + (dd.current_drawdown > 0 ? "red" : "muted");
  document.getElementById("t-dd-sub").textContent =
    `peak $${dd.peak_equity.toFixed(2)} · max $${dd.max_drawdown.toFixed(2)}`;

  // Streak
  const s = m.recent_10 || {};
  document.getElementById("t-streak").textContent = `${s.wins} / ${s.n}`;
  document.getElementById("t-streak").className =
    "value " + (s.wins >= 4 ? "green" : "red");
  document.getElementById("t-streak-sub").textContent = "wins in last 10";

  // Inversion
  const inv = m.inversion || {};
  document.getElementById("t-inv-orig").textContent =
    inv.n ? inv.original_winrate.toFixed(2) + "%" : "—";
  document.getElementById("t-inv-inv").textContent =
    inv.n ? inv.inverted_winrate.toFixed(2) + "%" : "—";
  const edge = inv.edge_pct || 0;
  const edgeEl = document.getElementById("t-inv-sub");
  if (inv.n < 20) {
    edgeEl.textContent = `need ≥20 trades (have ${inv.n})`;
  } else if (Math.abs(edge) < 2) {
    edgeEl.innerHTML = `<span class="yellow">≈ no edge — coin flip</span>`;
  } else if (edge > 0) {
    edgeEl.innerHTML = `+${edge.toFixed(2)}pp edge over inversion`;
  } else {
    edgeEl.innerHTML = `<span class="red">${edge.toFixed(2)}pp — strategy worse than flipping a coin</span>`;
  }

  // Hourly edge
  const hb = document.getElementById("hourly-body");
  hb.innerHTML = "";
  for (const h of m.hourly_edge || []) {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${String(h.hour_utc).padStart(2, "0")}:00</td>
      <td class="num">${h.trades}</td>
      <td class="num">${h.wins}</td>
      <td class="num">${h.losses}</td>
      <td class="num ${h.label === "PRIME" ? "green" : (h.label === "LOSS" ? "red" : "")}">${h.win_rate.toFixed(1)}%</td>
      <td class="num ${h.profit >= 0 ? "green" : "red"}">${fmtPnl(h.profit)}</td>
      <td><span class="pill ${pillClass(h.label)}">${h.label}</span></td>
    `;
    hb.appendChild(tr);
  }

  // Per-asset kill
  const kb = document.getElementById("kill-body");
  if (!m.per_asset_kill || m.per_asset_kill.length === 0) {
    kb.innerHTML = `<tr><td colspan="6" class="muted">none — no asset has ≥20 trades and ≤45% WR</td></tr>`;
  } else {
    kb.innerHTML = "";
    for (const a of m.per_asset_kill) {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td>${escHtml(a.asset)}</td>
        <td class="num">${a.trades}</td>
        <td class="num">${a.wins}</td>
        <td class="num">${a.losses}</td>
        <td class="num red">${a.win_rate.toFixed(2)}%</td>
        <td class="num ${a.profit >= 0 ? "green" : "red"}">${fmtPnl(a.profit)}</td>
      `;
      kb.appendChild(tr);
    }
  }

  // Accounts
  const ab = document.getElementById("accounts-body");
  const accounts = (m.service_status && m.service_status.accounts) || [];
  if (accounts.length === 0) {
    ab.innerHTML = `<tr><td colspan="4" class="muted">none</td></tr>`;
  } else {
    ab.innerHTML = "";
    for (const a of accounts) {
      const tr = document.createElement("tr");
      const status = a.service_status
        ? '<span class="pill profitable">LIVE</span>'
        : '<span class="pill loss">PAUSED</span>';
      tr.innerHTML = `
        <td>${escHtml(a.account_doc_id.slice(-8))}</td>
        <td>${escHtml(a.account_mode)}</td>
        <td>${status}</td>
        <td class="num">$${(a.trade_amount || 0).toFixed(2)}</td>
      `;
      ab.appendChild(tr);
    }
  }

  // Recent trades — last 50 from rolling window
  const tb = document.getElementById("trades-body");
  tb.innerHTML = "";
  const trades = (m.rolling_500_count && m._trades) || [];
  // We don't fetch the full 500 here; the API gives us a count. Fetch separately.
  fetchTrades();
}

function renderKillState(s) {
  const banner = document.getElementById("banner");
  const title = document.getElementById("banner-title");
  const reasonsEl = document.getElementById("banner-reasons");
  if (s.triggered) {
    banner.className = "banner triggered";
    title.textContent = "🛑 KILL SWITCH TRIGGERED — bot paused";
    reasonsEl.innerHTML = (s.reasons || []).map(r => "• " + escHtml(r)).join("<br>");
  } else {
    banner.className = "banner armed";
    title.textContent = "✅ ARMED — bot live, all thresholds within range";
    reasonsEl.textContent = "";
  }
}

async function fetchTrades() {
  try {
    // We piggyback on /api/metrics but it doesn't return the full 500. Add a fetch here.
    const r = await fetch("/api/trades?limit=50", { headers: { "Accept": "application/json" } });
    if (!r.ok) return;
    const trades = await r.json();
    const tb = document.getElementById("trades-body");
    tb.innerHTML = "";
    for (const t of trades) {
      const tr = document.createElement("tr");
      const cls = t.result === "WIN" ? "green" : (t.result === "LOSS" ? "red" : "yellow");
      tr.innerHTML = `
        <td>${fmtTime(t.timestamp)}</td>
        <td>${escHtml(t.asset)}</td>
        <td>${escHtml(t.direction)}</td>
        <td class="${cls}">${escHtml(t.result)}</td>
        <td class="num ${t.profit >= 0 ? "green" : "red"}">${fmtPnl(t.profit)}</td>
        <td class="num">$${t.amount.toFixed(2)}</td>
        <td class="muted">${escHtml(t.reason)}</td>
      `;
      tb.appendChild(tr);
    }
  } catch (e) { /* best-effort */ }
}

async function postKillSwitch(action) {
  if (!ensureToken()) return;
  try {
    const r = await fetchJSON("/api/kill_switch/" + action, { method: "POST" });
    if (r.ok) {
      await refresh();
    } else {
      alert("Failed: " + JSON.stringify(r));
    }
  } catch (e) {
    alert("Error: " + e.message);
  }
}

document.getElementById("btn-pause").addEventListener("click", () => postKillSwitch("pause"));
document.getElementById("btn-resume").addEventListener("click", () => postKillSwitch("resume"));

refresh();
setInterval(refresh, 5000);
</script>
</body>
</html>
"""
