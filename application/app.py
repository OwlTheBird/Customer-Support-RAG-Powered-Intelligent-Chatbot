import time
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from llm import chain, extract_text
from retriever import retrieve, build_context
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from errors import register_error_handlers
import db

app = Flask(__name__)

# ── CORS ───────────────────────────────────────────────────────────────────────
# Allow the React dashboard (any origin for now; lock down in production)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# ── Rate limiter ───────────────────────────────────────────────────────────────
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://",
)

# ── Error handlers ─────────────────────────────────────────────────────────────
register_error_handlers(app)

# ── DB init ────────────────────────────────────────────────────────────────────
with app.app_context():
    db.init_db()


# ══════════════════════════════════════════════════════════════════════════════
# Chatbot routes
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
@limiter.limit("5 per minute")
def ask():
    user_input = request.form.get("user_input")

    if not user_input:
        return jsonify({"error": "Bad Request", "message": "No input provided"}), 400

    t_start = time.monotonic()

    chunks  = retrieve(user_input, top_k=2)
    context = build_context(chunks)
    answer  = extract_text(
        chain.invoke({"context": context, "question": user_input}).content
    )

    latency_ms = int((time.monotonic() - t_start) * 1000)

    # ── Persist to SQLite ──────────────────────────────────────────────────────
    log_id = db.log_query(
        user_input=user_input,
        answer=answer,
        chunks_used=len(chunks),
        latency_ms=latency_ms,
        model="gemini",
    )

    return jsonify({
        "log_id":   log_id,
        "question": user_input,
        "answer":   answer,
        "chunks":   chunks,
    })


# ══════════════════════════════════════════════════════════════════════════════
# Dashboard API routes  (all prefixed /api/)
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/api/health")
def api_health():
    """Quick liveness check used by the dashboard header badge."""
    return jsonify({"status": "ok", "model": "gemini"})


@app.route("/api/logs")
def api_logs():
    """
    GET /api/logs?limit=50
    Returns recent query/answer logs from the SQLite DB.
    """
    limit = min(int(request.args.get("limit", 50)), 200)
    logs  = db.get_recent_logs(limit=limit)
    return jsonify(logs)


@app.route("/api/metrics")
def api_metrics():
    """
    GET /api/metrics
    Returns aggregated KPIs: query count, avg latency, satisfaction rate, etc.
    """
    return jsonify(db.get_metrics())


@app.route("/api/rate", methods=["POST"])
def api_rate():
    """
    POST /api/rate   body: { log_id: int, rating: 'positive'|'negative' }
    Lets the dashboard thumbs-up/down a specific answer.
    """
    data   = request.get_json(silent=True) or {}
    log_id = data.get("log_id")
    rating = data.get("rating")

    if not log_id or rating not in ("positive", "negative"):
        return jsonify({"error": "Bad Request", "message": "log_id and rating are required"}), 400

    db.set_rating(log_id, rating)
    return jsonify({"ok": True})
