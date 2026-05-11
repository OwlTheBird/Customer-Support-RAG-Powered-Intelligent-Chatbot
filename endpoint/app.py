from flask import Flask, render_template, request, Response
from llm import chain
from retriever import retrieve, build_context

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/stream")
def stream():
    user_input = request.args.get("user_input")

    def generate():
        chunks = retrieve(user_input, top_k=2)
        context = build_context(chunks)

        for chunk in chain.stream({"context": context, "question": user_input}):
            content = chunk.content
            if isinstance(content, list):
                text = "".join(
                    block.get("text", "")
                    for block in content
                    if isinstance(block, dict)
                )
            else:
                text = content  # already a plain string (doesn't happen in our case)
            if text:
                safe = text.replace("\n", "\\n")
                yield f"data: {safe}\n\n"

    return Response(
        generate(), mimetype="text/event-stream"
    )  # Server‑Sent Events (SSE)
