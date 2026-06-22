"""
gui_app.py  —  Web GUI for the Roman Urdu RAG System
Run with:  python gui_app.py
Opens at:  http://127.0.0.1:5000
All original logic in main.py is imported unchanged.
"""

import os, sys, json, threading
from datetime import datetime
from flask import Flask, render_template_string, request, jsonify, Response

# ── Make sure the project root is on the path ─────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from normalization.normalize import normalize_query, transliterate_query, load_normalization_map
from embeddings.embed import load_index, load_model
from retrieval.retrieve import baseline_retrieve, improved_retrieve
from generation.generate import get_client, generate_answer

# ── Globals (loaded once at startup) ──────────────────────────────────────────
_model = _index = _chunks = _norm_map = _client = None
_gen_available = False
RESPONSES_PATH = "output/responses.json"

app = Flask(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# HTML TEMPLATE
# ─────────────────────────────────────────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>RAG — Roman Urdu</title>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@400;500&family=Noto+Nastaliq+Urdu&display=swap" rel="stylesheet"/>
<style>
:root {
  --bg:         #0a0a0f;
  --surface:    #111118;
  --border:     #1e1e2e;
  --border2:    #2a2a3e;
  --accent:     #7b5ea7;
  --accent2:    #a87fd4;
  --gold:       #c9a84c;
  --gold2:      #e8c97a;
  --text:       #e8e6f0;
  --muted:      #7a788a;
  --success:    #4caf7d;
  --danger:     #cf6679;
  --r:          12px;
  --r2:         6px;
}

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

body {
  background: var(--bg);
  color: var(--text);
  font-family: 'Syne', sans-serif;
  min-height: 100vh;
  overflow-x: hidden;
}

/* ── Background decoration ── */
body::before {
  content: '';
  position: fixed; inset: 0; z-index: 0; pointer-events: none;
  background:
    radial-gradient(ellipse 60% 50% at 80% 10%, rgba(123,94,167,.18) 0%, transparent 60%),
    radial-gradient(ellipse 40% 40% at 10% 90%, rgba(201,168,76,.10) 0%, transparent 55%);
}

/* ── Layout ── */
.shell {
  position: relative; z-index: 1;
  max-width: 1100px;
  margin: 0 auto;
  padding: 0 24px 80px;
}

/* ── Header ── */
header {
  padding: 48px 0 40px;
  display: flex; align-items: flex-end; gap: 20px;
}
.logo-mark {
  width: 48px; height: 48px; flex-shrink: 0;
  background: linear-gradient(135deg, var(--accent) 0%, var(--gold) 100%);
  border-radius: 14px;
  display: grid; place-items: center;
  font-size: 22px; font-weight: 800; color: #fff;
  box-shadow: 0 0 28px rgba(123,94,167,.5);
}
.header-text h1 {
  font-size: clamp(22px,4vw,32px);
  font-weight: 800;
  letter-spacing: -.5px;
  background: linear-gradient(90deg, var(--text) 40%, var(--accent2));
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.header-text p { font-size: 13px; color: var(--muted); margin-top: 4px; letter-spacing:.3px; }
.status-pill {
  margin-left: auto; flex-shrink: 0;
  background: var(--surface); border: 1px solid var(--border2);
  border-radius: 99px; padding: 6px 14px;
  font-size: 12px; font-family: 'JetBrains Mono', monospace;
  color: var(--muted); display: flex; align-items: center; gap: 7px;
}
.dot { width:7px; height:7px; border-radius:50%; background: var(--success); box-shadow: 0 0 6px var(--success); }

/* ── Two-col ── */
.cols { display: grid; grid-template-columns: 1fr 340px; gap: 24px; }
@media(max-width:760px){ .cols { grid-template-columns: 1fr; } }

/* ── Card ── */
.card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--r);
  padding: 24px;
}
.card-title {
  font-size: 11px; font-weight: 700; letter-spacing: 1.5px;
  text-transform: uppercase; color: var(--muted);
  margin-bottom: 16px; display: flex; align-items: center; gap: 8px;
}
.card-title::before {
  content: ''; display: block;
  width: 3px; height: 14px; border-radius: 2px;
  background: linear-gradient(var(--accent), var(--gold));
}

/* ── Input ── */
.query-wrap { position: relative; margin-bottom: 16px; }
textarea#query {
  width: 100%;
  background: #0d0d16;
  border: 1px solid var(--border2);
  border-radius: var(--r2);
  color: var(--text);
  font-family: 'JetBrains Mono', monospace;
  font-size: 15px;
  padding: 14px 16px;
  resize: vertical;
  min-height: 90px;
  outline: none;
  transition: border-color .2s;
  line-height: 1.6;
}
textarea#query:focus { border-color: var(--accent); }
textarea#query::placeholder { color: var(--muted); }

/* ── Mode selector ── */
.mode-row { display: flex; gap: 10px; margin-bottom: 20px; }
.mode-btn {
  flex: 1; padding: 9px 8px; font-family: 'Syne', sans-serif;
  font-size: 12px; font-weight: 700; letter-spacing: .5px;
  border-radius: var(--r2); cursor: pointer;
  border: 1px solid var(--border2);
  background: transparent; color: var(--muted);
  transition: all .2s;
}
.mode-btn.active, .mode-btn:hover {
  border-color: var(--accent);
  color: var(--text);
  background: rgba(123,94,167,.12);
}
.mode-btn.active { background: rgba(123,94,167,.22); color: var(--accent2); }

/* ── Run button ── */
#run-btn {
  width: 100%; padding: 13px;
  background: linear-gradient(90deg, var(--accent), var(--gold));
  color: #fff; font-family: 'Syne', sans-serif;
  font-size: 14px; font-weight: 700; letter-spacing: .5px;
  border: none; border-radius: var(--r2); cursor: pointer;
  position: relative; overflow: hidden;
  transition: opacity .2s, transform .1s;
}
#run-btn:hover { opacity: .9; }
#run-btn:active { transform: scale(.99); }
#run-btn.loading { opacity: .7; pointer-events: none; }
#run-btn::after {
  content: ''; position: absolute; inset: 0;
  background: linear-gradient(90deg,transparent 0%,rgba(255,255,255,.15) 50%,transparent 100%);
  transform: translateX(-100%);
  transition: transform 0s;
}
#run-btn.loading::after { transform: translateX(100%); transition: transform 1.2s ease; }

/* ── Stats sidebar ── */
.stat-block { margin-bottom: 20px; }
.stat-label { font-size: 11px; color: var(--muted); letter-spacing:.5px; margin-bottom: 5px; }
.stat-val {
  font-family: 'JetBrains Mono', monospace;
  font-size: 22px; font-weight: 500;
  background: linear-gradient(90deg, var(--accent2), var(--gold2));
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.stat-sub { font-size: 11px; color: var(--muted); margin-top: 3px; }

.history-list { list-style: none; }
.history-item {
  padding: 8px 10px; border-radius: var(--r2);
  font-size: 12px; font-family: 'JetBrains Mono', monospace;
  color: var(--muted); cursor: pointer;
  border: 1px solid transparent;
  margin-bottom: 6px;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  transition: all .15s;
}
.history-item:hover { border-color: var(--border2); color: var(--text); background: rgba(255,255,255,.03); }

/* ── Results ── */
#results { margin-top: 24px; }
.result-section { margin-bottom: 24px; }

.answer-box {
  background: linear-gradient(135deg, rgba(123,94,167,.08), rgba(201,168,76,.06));
  border: 1px solid rgba(123,94,167,.3);
  border-radius: var(--r);
  padding: 22px 24px;
}
.answer-text {
  font-family: 'Noto Nastaliq Urdu', serif;
  font-size: 18px; line-height: 2.2;
  direction: rtl; text-align: right;
  color: var(--text);
}
.answer-text.roman {
  font-family: 'JetBrains Mono', monospace;
  font-size: 14px; line-height: 1.8;
  direction: ltr; text-align: left;
}

.passages-grid { display: flex; flex-direction: column; gap: 12px; }
.passage-card {
  background: #0d0d16;
  border: 1px solid var(--border);
  border-radius: var(--r2);
  padding: 16px 18px;
  transition: border-color .2s;
  position: relative; overflow: hidden;
}
.passage-card:hover { border-color: var(--border2); }
.passage-card::before {
  content: '';
  position: absolute; left: 0; top: 0; bottom: 0;
  width: 3px;
  background: linear-gradient(var(--accent), var(--gold));
  opacity: 0; transition: opacity .2s;
}
.passage-card:hover::before { opacity: 1; }
.passage-meta {
  display: flex; align-items: center; gap: 10px;
  margin-bottom: 10px;
}
.passage-rank {
  width: 24px; height: 24px; border-radius: 6px;
  background: rgba(123,94,167,.18);
  border: 1px solid rgba(123,94,167,.3);
  display: grid; place-items: center;
  font-size: 11px; font-weight: 700; color: var(--accent2);
}
.passage-title { font-size: 13px; font-weight: 700; color: var(--text); flex: 1; }
.score-badge {
  font-family: 'JetBrains Mono', monospace; font-size: 11px;
  padding: 3px 8px; border-radius: 99px;
  background: rgba(201,168,76,.12); color: var(--gold2);
  border: 1px solid rgba(201,168,76,.2);
}
.passage-text {
  font-size: 13px; line-height: 1.7; color: var(--muted);
  display: -webkit-box; -webkit-line-clamp: 4; -webkit-box-orient: vertical;
  overflow: hidden;
}
.passage-id {
  margin-top: 8px; font-size: 10px;
  font-family: 'JetBrains Mono', monospace; color: #3a3a50;
}

/* ── Normalization preview ── */
.norm-row {
  display: flex; gap: 12px; flex-wrap: wrap;
  margin-bottom: 16px;
}
.norm-chip {
  background: #0d0d16; border: 1px solid var(--border2);
  border-radius: var(--r2); padding: 8px 14px;
}
.norm-chip-label { font-size: 10px; color: var(--muted); letter-spacing:.5px; text-transform:uppercase; margin-bottom: 4px; }
.norm-chip-val { font-size: 13px; font-family: 'JetBrains Mono', monospace; color: var(--text); }
.norm-chip-val.urdu { font-family: 'Noto Nastaliq Urdu', serif; font-size: 16px; direction: rtl; }

/* ── Mode tag ── */
.mode-tag {
  display: inline-flex; align-items: center; gap: 5px;
  font-size: 11px; font-weight: 700; letter-spacing:.5px;
  padding: 3px 10px; border-radius: 99px;
  text-transform: uppercase;
}
.mode-tag.baseline { background: rgba(76,175,125,.1); color: var(--success); border: 1px solid rgba(76,175,125,.25); }
.mode-tag.improved { background: rgba(123,94,167,.1); color: var(--accent2); border: 1px solid rgba(123,94,167,.25); }
.mode-tag.both { background: rgba(201,168,76,.1); color: var(--gold2); border: 1px solid rgba(201,168,76,.25); }

/* ── Spinner ── */
.spinner {
  display: inline-block; width: 14px; height: 14px;
  border: 2px solid rgba(255,255,255,.3); border-top-color: #fff;
  border-radius: 50%; animation: spin .7s linear infinite; vertical-align: middle;
}
@keyframes spin { to { transform: rotate(360deg); } }

/* ── Divider ── */
.divider { height: 1px; background: var(--border); margin: 20px 0; }

/* ── Scroll indicator ── */
#loading-bar {
  position: fixed; top: 0; left: 0; right: 0; height: 3px; z-index: 999;
  background: linear-gradient(90deg, var(--accent), var(--gold));
  transform: scaleX(0); transform-origin: left;
  transition: transform .3s; border-radius: 0 2px 2px 0;
}
#loading-bar.active { transform: scaleX(0.7); }
#loading-bar.done   { transform: scaleX(1); transition: transform .15s; }

/* ── No results ── */
.empty-state {
  text-align: center; padding: 60px 20px; color: var(--muted);
}
.empty-state .big { font-size: 48px; margin-bottom: 14px; opacity: .4; }
.empty-state p { font-size: 14px; line-height: 1.7; }

/* ── Fade in ── */
@keyframes fadeUp {
  from { opacity:0; transform: translateY(12px); }
  to   { opacity:1; transform: translateY(0); }
}
.fade-in { animation: fadeUp .35s ease both; }
.fade-in-d { animation: fadeUp .35s .1s ease both; }
</style>
</head>
<body>
<div id="loading-bar"></div>
<div class="shell">

  <!-- Header -->
  <header>
    <div class="logo-mark">ر</div>
    <div class="header-text">
      <h1>Roman Urdu RAG System</h1>
      <p>Retrieval-Augmented Generation · Semantic Search · Lexical Normalization</p>
    </div>
    <div class="status-pill"><span class="dot"></span> Pipeline Ready</div>
  </header>

  <div class="cols">

    <!-- LEFT: Query + Results -->
    <div>
      <div class="card fade-in">
        <div class="card-title">Query Interface</div>

        <div class="query-wrap">
          <textarea id="query" placeholder="Type your Roman Urdu query here…
e.g.  mujhe pakistan ki tarikh ke baare mein batao"></textarea>
        </div>

        <div class="mode-row">
          <button class="mode-btn" data-mode="baseline" onclick="setMode('baseline')">Baseline</button>
          <button class="mode-btn active" data-mode="improved" onclick="setMode('improved')">Improved</button>
          <button class="mode-btn" data-mode="both" onclick="setMode('both')">Both</button>
        </div>

        <button id="run-btn" onclick="runQuery()">
          <span id="btn-text">Run Query</span>
        </button>
      </div>

      <div id="results">
        <div class="empty-state fade-in-d">
          <div class="big">◈</div>
          <p>Enter a Roman Urdu query above and click <strong>Run Query</strong>.<br/>
          Results and generated answers will appear here.</p>
        </div>
      </div>
    </div>

    <!-- RIGHT: Sidebar -->
    <div>
      <div class="card fade-in" style="margin-bottom:20px;">
        <div class="card-title">Session Stats</div>
        <div class="stat-block">
          <div class="stat-label">Total Queries Run</div>
          <div class="stat-val" id="stat-total">0</div>
        </div>
        <div class="stat-block">
          <div class="stat-label">Last Query Time</div>
          <div class="stat-val" id="stat-time">—</div>
          <div class="stat-sub" id="stat-ts"></div>
        </div>
        <div class="stat-block">
          <div class="stat-label">Passages Retrieved</div>
          <div class="stat-val" id="stat-passages">—</div>
        </div>
      </div>

      <div class="card fade-in">
        <div class="card-title">Recent Queries</div>
        <ul class="history-list" id="history-list">
          <li style="font-size:12px;color:var(--muted);font-family:'JetBrains Mono',monospace;">No queries yet…</li>
        </ul>
      </div>
    </div>

  </div><!-- /.cols -->
</div><!-- /.shell -->

<script>
let selectedMode = 'improved';
let sessionCount = 0;
let history = [];

function setMode(m) {
  selectedMode = m;
  document.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('active'));
  document.querySelector(`[data-mode="${m}"]`).classList.add('active');
}

function scoreColor(s) {
  if (s >= 0.7) return 'var(--success)';
  if (s >= 0.4) return 'var(--gold2)';
  return 'var(--muted)';
}

function renderResult(data, container) {
  // Mode tag
  const modeClass = data.mode === 'both' ? 'both' : data.mode;
  let html = `
  <div class="result-section fade-in" style="margin-bottom:8px;">
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:16px;">
      <span class="mode-tag ${data.mode}">${data.mode.toUpperCase()}</span>
      <span style="font-size:11px;color:var(--muted);font-family:'JetBrains Mono',monospace;">${data.timestamp}</span>
    </div>`;

  // Normalization chips (improved only)
  if (data.query.normalized && data.mode === 'improved') {
    html += `<div class="norm-row">
      <div class="norm-chip">
        <div class="norm-chip-label">Original</div>
        <div class="norm-chip-val">${escHtml(data.query.original)}</div>
      </div>
      <div class="norm-chip">
        <div class="norm-chip-label">Normalized</div>
        <div class="norm-chip-val">${escHtml(data.query.normalized)}</div>
      </div>`;
    if (data.query.urdu_script && data.query.urdu_script !== data.query.original) {
      html += `
      <div class="norm-chip">
        <div class="norm-chip-label">Urdu Script</div>
        <div class="norm-chip-val urdu">${escHtml(data.query.urdu_script)}</div>
      </div>`;
    }
    html += `</div>`;
  }

  // Answer
  if (data.answer) {
    const isUrdu = /[\u0600-\u06FF]/.test(data.answer);
    html += `
    <div class="card-title" style="margin-bottom:12px;">Generated Answer</div>
    <div class="answer-box" style="margin-bottom:20px;">
      <div class="answer-text ${isUrdu ? '' : 'roman'}">${escHtml(data.answer)}</div>
    </div>`;
  }

  // Passages
  if (data.retrieved_passages && data.retrieved_passages.length) {
    html += `<div class="card-title" style="margin-bottom:12px;">Retrieved Passages <span style="font-size:11px;font-weight:400;color:var(--muted);">(top ${data.retrieved_passages.length})</span></div>
    <div class="passages-grid">`;
    for (const p of data.retrieved_passages) {
      const sc = (p.score * 100).toFixed(1);
      html += `
      <div class="passage-card">
        <div class="passage-meta">
          <div class="passage-rank">${p.rank}</div>
          <div class="passage-title">${escHtml(p.title || 'Untitled Passage')}</div>
          <div class="score-badge" style="color:${scoreColor(p.score)}">${sc}%</div>
        </div>
        <div class="passage-text">${escHtml(p.text || '(no text)')}</div>
        <div class="passage-id">${escHtml(p.chunk_id || '')}</div>
      </div>`;
    }
    html += `</div>`;
  }

  html += `</div><div class="divider"></div>`;
  const div = document.createElement('div');
  div.innerHTML = html;
  container.prepend(div);
}

function escHtml(s) {
  if (!s) return '';
  return String(s)
    .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
    .replace(/"/g,'&quot;').replace(/'/g,'&#39;');
}

async function runQuery() {
  const query = document.getElementById('query').value.trim();
  if (!query) { document.getElementById('query').focus(); return; }

  const btn = document.getElementById('run-btn');
  const btnText = document.getElementById('btn-text');
  const bar = document.getElementById('loading-bar');

  btn.classList.add('loading');
  btnText.innerHTML = '<span class="spinner"></span>  Running…';
  bar.classList.add('active');

  const t0 = Date.now();
  try {
    const res = await fetch('/api/query', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ query, mode: selectedMode })
    });
    const data = await res.json();
    const elapsed = ((Date.now() - t0) / 1000).toFixed(2);

    bar.classList.remove('active');
    bar.classList.add('done');
    setTimeout(() => bar.classList.remove('done'), 600);

    // update stats
    sessionCount++;
    document.getElementById('stat-total').textContent = sessionCount;
    document.getElementById('stat-time').textContent = elapsed + 's';
    document.getElementById('stat-ts').textContent = new Date().toLocaleTimeString();

    const passages = Array.isArray(data) ? data.reduce((a,d) => a + (d.retrieved_passages?.length||0),0)
                                         : (data.retrieved_passages?.length || 0);
    document.getElementById('stat-passages').textContent = passages;

    // history
    history.unshift(query);
    const hl = document.getElementById('history-list');
    hl.innerHTML = history.slice(0,8).map(q =>
      `<li class="history-item" onclick="document.getElementById('query').value=this.dataset.q" data-q="${escHtml(q)}">${escHtml(q.length>40?q.slice(0,40)+'…':q)}</li>`
    ).join('');

    // render
    const resultsEl = document.getElementById('results');
    resultsEl.innerHTML = ''; // clear empty state
    if (Array.isArray(data)) {
      data.forEach(d => renderResult(d, resultsEl));
    } else {
      renderResult(data, resultsEl);
    }

  } catch(e) {
    bar.classList.remove('active');
    document.getElementById('results').innerHTML = `
      <div class="card fade-in" style="border-color:var(--danger);">
        <div style="color:var(--danger);font-size:14px;font-family:'JetBrains Mono',monospace;">
          Error: ${escHtml(String(e))}
        </div>
      </div>`;
  } finally {
    btn.classList.remove('loading');
    btnText.textContent = 'Run Query';
  }
}

document.getElementById('query').addEventListener('keydown', e => {
  if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) runQuery();
});
</script>
</body>
</html>"""


# ─────────────────────────────────────────────────────────────────────────────
# Startup — load pipeline
# ─────────────────────────────────────────────────────────────────────────────

def _load():
    global _model, _index, _chunks, _norm_map, _client, _gen_available
    print("Loading RAG pipeline…")
    _model = load_model()
    _index, _chunks = load_index()
    _norm_map = load_normalization_map()
    try:
        _client = get_client()
        _gen_available = True
        print("OpenAI client ready.")
    except ValueError as e:
        print(f"WARNING: {e} — running retrieval-only mode.")
        _client = None
        _gen_available = False
    print("Pipeline loaded. GUI ready at http://127.0.0.1:5000\n")


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/api/query", methods=["POST"])
def api_query():
    body  = request.get_json(force=True)
    query = body.get("query", "").strip()
    mode  = body.get("mode", "improved")

    if not query:
        return jsonify({"error": "empty query"}), 400

    modes = ["baseline", "improved"] if mode == "both" else [mode]
    records = []
    for m in modes:
        records.append(_run_one(query, m))

    # persist
    _persist_all(records)

    return jsonify(records if len(records) > 1 else records[0])


def _run_one(query: str, mode: str) -> dict:
    normalized = query
    urdu_script = query

    if mode == "baseline":
        retrieved = baseline_retrieve(query, _model, _index, _chunks, 5)
    else:
        retrieved, normalized, urdu_script = improved_retrieve(
            query, _model, _index, _chunks, _norm_map, 5
        )

    answer = None
    if _gen_available and _client:
        display_q = urdu_script if mode == "improved" else query
        answer = generate_answer(display_q, retrieved[:3], _client)

    return {
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "mode": mode,
        "query": {
            "original": query,
            "normalized": normalized,
            "urdu_script": urdu_script,
        },
        "retrieved_passages": [
            {
                "rank": i + 1,
                "score": round(r["score"], 4),
                "chunk_id": r.get("chunk_id", ""),
                "title": r.get("title", ""),
                "text": r.get("text", ""),
            }
            for i, r in enumerate(retrieved)
        ],
        "answer": answer,
    }


def _persist_all(records):
    os.makedirs("output", exist_ok=True)
    existing = []
    if os.path.exists(RESPONSES_PATH):
        with open(RESPONSES_PATH, "r", encoding="utf-8") as f:
            try:
                existing = json.load(f)
            except Exception:
                existing = []
    existing.extend(records)
    with open(RESPONSES_PATH, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    _load()
    # Open browser automatically after a short delay
    import webbrowser, threading
    threading.Timer(1.2, lambda: webbrowser.open("http://127.0.0.1:5000")).start()
    app.run(debug=False, port=5000)
