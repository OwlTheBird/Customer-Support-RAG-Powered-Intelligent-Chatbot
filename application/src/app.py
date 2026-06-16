import time
from flask import Flask, render_template, request, jsonify  # , Response
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from src.errors import register_error_handlers
from src.llm import chain, extract_text
from src.retriever import retrieve, build_context
# from flask_cors import CORS
from src.db import init_db, log_query, get_recent_logs, get_metrics, set_rating
from src.config import AI_MODEL

app = Flask(__name__)
# CORS(app, resources={r"/api/*": {"origins": "http://localhost:5173/"}})
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://",
)

register_error_handlers(app)


with app.app_context():
    init_db()


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

    chunks = retrieve(user_input, top_k=2)
    context = build_context(chunks)

    answer = extract_text(
        chain.invoke({"context": context, "question": user_input}).content
    )

    latency_ms = int((time.monotonic() - t_start) * 1000)

    log_id = log_query(
        user_input=user_input,
        answer=answer,
        chunks_used=len(chunks),
        latency_ms=latency_ms,
        model=AI_MODEL,
    )

    return jsonify(
        {
            "log_id": log_id,
            "question": user_input,
            "answer": answer,
            "chunks": chunks,
        }
    )


# * It drains the free quota
# @app.route("/stream")
# @limiter.limit("1 per minute")
# def stream():
#     user_input = request.args.get("user_input")

#     def generate():
#         chunks = retrieve(user_input, top_k=2)
#         context = build_context(chunks)

#         for chunk in chain.stream({"context": context, "question": user_input}):
#             text = extract_text(chunk.content)

#             if text:
#                 safe = text.replace("\n", "\\n")
#                 yield f"data: {safe}\n\n"

#     return Response(
#         generate(), mimetype="text/event-stream"
#     )  # Server‑Sent Events (SSE)


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
    logs = get_recent_logs(limit=limit)
    return jsonify(logs)


@app.route("/api/metrics")
def api_metrics():
    """
    GET /api/metrics
    Returns aggregated KPIs: query count, avg latency, satisfaction rate, etc.
    """
    return jsonify(get_metrics())


@app.route("/api/rate", methods=["POST"])
def api_rate():
    """
    POST /api/rate   body: { log_id: int, rating: 'positive'|'negative' }
    Lets the dashboard thumbs-up/down a specific answer.
    """
    data = request.get_json(silent=True) or {}
    log_id = data.get("log_id")
    rating = data.get("rating")

    if not log_id or rating not in ("positive", "negative"):
        return jsonify(
            {"error": "Bad Request", "message": "log_id and rating are required"}
        ), 400

    set_rating(log_id, rating)
    return jsonify({"ok": True})
