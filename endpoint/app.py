from flask import Flask, render_template, request, jsonify  # , Response
from llm import chain, extract_text
from retriever import retrieve, build_context
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from errors import register_error_handlers

app = Flask(__name__)
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://",
)

register_error_handlers(app)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
@limiter.limit("5 per minute")
def ask():
    user_input = request.form.get("user_input")

    if not user_input:
        return jsonify({"error": "Bad Request", "message": "No input provided"}), 400

    chunks = retrieve(user_input, top_k=2)
    context = build_context(chunks)

    answer = extract_text(
        chain.invoke({"context": context, "question": user_input}).content
    )

    return jsonify(
        {
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
