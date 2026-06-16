from pinecone import Pinecone
from src.config import PINECONE_API_KEY, TOP_K

pc = Pinecone(api_key=PINECONE_API_KEY)
pc_index = pc.Index("customer-support")


def retrieve(question: str, top_k: int = TOP_K) -> list[dict]:
    results = pc_index.search(
        namespace="faq",
        top_k=top_k,
        inputs={"text": question},
        fields=[
            "text"
            # "question",
            # "answer",
            # "category",
        ],  # keep only what you actually use
    )

    return [
        {
            "score": hit.score,
            "text": hit.fields.get("text"),  # Q + A combined
            # "question": hit.fields.get("question"),
            # "answer": hit.fields.get("answer"),
            # "category": hit.fields.get("category"),
        }
        for hit in results.result["hits"]
    ]


def build_context(chunks: list[dict]) -> str:
    return "\n\n".join(chunk["text"] for chunk in chunks if chunk.get("text"))
