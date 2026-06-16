"""BLEU and ROUGE-L evaluation of the RAG chatbot against the golden dataset.

This module:
1. Loads the golden dataset from evaluation/golden_dataset.json
2. Queries the RAG pipeline for each question (via /test endpoint or direct import)
3. Computes ROUGE-L (F1) and BLEU scores per question
4. Aggregates results with per-category breakdowns
5. Saves structured results to evaluation/results.json

Usage:
    python -m evaluation.evaluate                      # Queries live /test endpoint
    python -m evaluation.evaluate --mode direct        # Imports retriever+llm directly
    python -m evaluation.evaluate --endpoint http://localhost:5000  # Custom endpoint URL
"""

import argparse
import json
import logging
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Any

import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import requests
from rouge_score import rouge_scorer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

EVAL_DIR = Path(__file__).resolve().parent
GOLDEN_DATASET_PATH = EVAL_DIR / "golden_dataset.json"
RESULTS_PATH = EVAL_DIR / "results.json"


def load_golden_dataset(path: Path = GOLDEN_DATASET_PATH) -> list[dict[str, Any]]:
    """Load, validate, and sample the golden dataset.

    Args:
        path: Path to golden_dataset.json.

    Returns:
        List of 45 entry dicts (30 in-domain, 15 OOD) with keys:
        id, question, expected_answer, category, is_ood.

    Raises:
        FileNotFoundError: If the golden dataset file does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
        ValueError: If required keys are missing from any entry.
    """
    if not path.exists():
        raise FileNotFoundError(f"Golden dataset not found at {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    entries = data.get("entries", [])
    required_keys = {"id", "question", "expected_answer", "category", "is_ood"}
    for entry in entries:
        missing = required_keys - set(entry.keys())
        if missing:
            raise ValueError(f"Entry {entry.get('id', '?')} missing keys: {missing}")

    # Separate in-domain and out-of-domain
    in_domain = [e for e in entries if not e["is_ood"]]
    ood = [e for e in entries if e["is_ood"]]

    # Sort to ensure deterministic order before sampling
    in_domain.sort(key=lambda x: x["id"])
    ood.sort(key=lambda x: x["id"])

    # Sample exactly 30 in-domain and 15 OOD
    import random
    rng = random.Random(42)
    in_domain_sample = rng.sample(in_domain, min(len(in_domain), 30))
    ood_sample = rng.sample(ood, min(len(ood), 15))

    sampled_entries = in_domain_sample + ood_sample
    sampled_entries.sort(key=lambda x: x["id"])

    logger.info(
        "Loaded and sampled %d entries (30 in-domain, 15 OOD) from golden dataset of %d total entries",
        len(sampled_entries),
        len(entries)
    )
    return sampled_entries


def query_rag_endpoint(
    question: str,
    endpoint_url: str = "http://localhost:5000",
    max_retries: int = 3,
    retry_delay: float = 2.0,
) -> str:
    """Query the RAG system via the /test HTTP endpoint.

    Args:
        question: The question to ask.
        endpoint_url: Base URL of the Flask app.
        max_retries: Maximum number of retry attempts.
        retry_delay: Seconds to wait between retries (doubles each attempt).

    Returns:
        The answer string from the RAG system.

    Raises:
        requests.RequestException: If all retries are exhausted.
    """
    url = f"{endpoint_url}/test"
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params={"user_input": question}, timeout=30)
            resp.raise_for_status()
            return resp.json()["answer"]
        except requests.RequestException as e:
            if attempt < max_retries - 1:
                wait = retry_delay * (2 ** attempt)
                logger.warning(
                    "Retry %d/%d after %.1fs: %s",
                    attempt + 1,
                    max_retries,
                    wait,
                    e,
                )
                time.sleep(wait)
            else:
                raise


def query_rag_direct(
    question: str,
    max_retries: int = 5,
    retry_delay: float = 2.0,
) -> str:
    """Query the RAG system by directly importing retriever and LLM chain.

    This bypasses the Flask endpoint and calls the chain directly.
    Useful for testing without running the Flask server.

    Args:
        question: The question to ask.
        max_retries: Maximum number of retry attempts for API calls.
        retry_delay: Seconds to wait between retries.

    Returns:
        The answer string from the RAG system.
    """
    # Import from parent package
    parent_dir = str(Path(__file__).resolve().parent.parent)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

    from retriever import retrieve, build_context
    from llm import chain, llm, extract_text

    # Dynamically override the model to bypass daily request limits on gemini-3.1-flash-lite
    llm.model = "gemma-4-31b-it"

    # Disable LangChain's internal retry loop to prevent quota-consuming retry storms
    if hasattr(llm, "max_retries"):
        llm.max_retries = 1

    chunks = retrieve(question, top_k=2)
    context = build_context(chunks)

    for attempt in range(max_retries):
        try:
            result = chain.invoke({"context": context, "question": question})
            return extract_text(result.content)
        except Exception as e:
            err_str = str(e)
            is_rate_limit = (
                "429" in err_str
                or "RESOURCE_EXHAUSTED" in err_str
                or "Quota exceeded" in err_str
            )
            if attempt < max_retries - 1:
                if is_rate_limit:
                    wait = 65.0
                    logger.warning(
                        "Rate limit hit. Waiting %.1fs to reset quota (attempt %d/%d): %s",
                        wait,
                        attempt + 1,
                        max_retries,
                        e,
                    )
                else:
                    wait = retry_delay * (2 ** attempt)
                    logger.warning(
                        "Direct query retry %d/%d after %.1fs: %s",
                        attempt + 1,
                        max_retries,
                        wait,
                        e,
                    )
                time.sleep(wait)
            else:
                raise


def compute_rouge_l(prediction: str, reference: str) -> dict[str, float]:
    """Compute ROUGE-L precision, recall, and F1 between prediction and reference.

    Args:
        prediction: The generated answer.
        reference: The expected answer.

    Returns:
        Dict with keys: precision, recall, fmeasure.
    """
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = scorer.score(reference, prediction)
    rl = scores["rougeL"]
    return {
        "precision": round(rl.precision, 4),
        "recall": round(rl.recall, 4),
        "fmeasure": round(rl.fmeasure, 4),
    }


def compute_bleu(prediction: str, reference: str) -> float:
    """Compute BLEU score between prediction and reference.

    Uses smoothing to handle short sequences and zero n-gram overlaps.

    Args:
        prediction: The generated answer.
        reference: The expected answer.

    Returns:
        BLEU score as a float in [0.0, 1.0].
    """
    ref_tokens = reference.lower().split()
    pred_tokens = prediction.lower().split()

    if not pred_tokens or not ref_tokens:
        return 0.0

    smoothing = SmoothingFunction().method1
    return round(
        sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoothing),
        4,
    )


def evaluate_single(
    entry: dict[str, Any],
    get_answer_fn: Callable[[str], str],
) -> dict[str, Any]:
    """Evaluate a single golden dataset entry.

    Args:
        entry: A golden dataset entry dict.
        get_answer_fn: Function that takes a question string and returns an answer string.

    Returns:
        Result dict with question, expected, generated, rouge_l, bleu, category, is_ood.
    """
    question = entry["question"]
    expected = entry["expected_answer"]
    is_ood = entry["is_ood"]

    try:
        generated = get_answer_fn(question)
    except Exception as e:
        logger.error("Failed to get answer for '%s': %s", question[:50], e)
        generated = ""

    rouge_l = compute_rouge_l(generated, expected)
    bleu = compute_bleu(generated, expected)

    # OOD accuracy: did the system correctly refuse?
    ood_correct = None
    if is_ood:
        refusal_phrases = [
            "i don't have that information",
            "i do not have that information",
            "i don't have enough information",
        ]
        ood_correct = any(phrase in generated.lower() for phrase in refusal_phrases)

    return {
        "id": entry["id"],
        "question": question,
        "expected_answer": expected,
        "generated_answer": generated,
        "category": entry["category"],
        "is_ood": is_ood,
        "rouge_l": rouge_l,
        "bleu": bleu,
        "ood_correct": ood_correct,
    }


def aggregate_results(per_question: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute aggregate metrics from per-question results.

    Computes:
    - Global mean/median/std for ROUGE-L F1 and BLEU
    - Per-category mean ROUGE-L F1 and BLEU
    - OOD accuracy (% of OOD questions correctly refused)

    Args:
        per_question: List of per-question result dicts.

    Returns:
        Aggregated results dict.
    """
    in_domain = [r for r in per_question if not r["is_ood"]]
    ood = [r for r in per_question if r["is_ood"]]

    rouge_scores = [r["rouge_l"]["fmeasure"] for r in in_domain]
    bleu_scores = [r["bleu"] for r in in_domain]

    # Per-category breakdown
    categories = set(r["category"] for r in in_domain)
    per_category = {}
    for cat in sorted(categories):
        cat_results = [r for r in in_domain if r["category"] == cat]
        per_category[cat] = {
            "count": len(cat_results),
            "rouge_l_mean": round(
                statistics.mean([r["rouge_l"]["fmeasure"] for r in cat_results]),
                4,
            ) if cat_results else 0.0,
            "bleu_mean": round(
                statistics.mean([r["bleu"] for r in cat_results]),
                4,
            ) if cat_results else 0.0,
        }

    ood_accuracy = None
    if ood:
        ood_correct_count = sum(1 for r in ood if r["ood_correct"])
        ood_accuracy = round(ood_correct_count / len(ood), 4)

    return {
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_evaluated": len(per_question),
            "in_domain_count": len(in_domain),
            "ood_count": len(ood),
        },
        "aggregate": {
            "rouge_l_fmeasure": {
                "mean": round(statistics.mean(rouge_scores), 4) if rouge_scores else None,
                "median": round(statistics.median(rouge_scores), 4) if rouge_scores else None,
                "std": round(statistics.stdev(rouge_scores), 4) if len(rouge_scores) > 1 else None,
                "min": round(min(rouge_scores), 4) if rouge_scores else None,
                "max": round(max(rouge_scores), 4) if rouge_scores else None,
            },
            "bleu": {
                "mean": round(statistics.mean(bleu_scores), 4) if bleu_scores else None,
                "median": round(statistics.median(bleu_scores), 4) if bleu_scores else None,
                "std": round(statistics.stdev(bleu_scores), 4) if len(bleu_scores) > 1 else None,
                "min": round(min(bleu_scores), 4) if bleu_scores else None,
                "max": round(max(bleu_scores), 4) if bleu_scores else None,
            },
            "ood_accuracy": ood_accuracy,
        },
        "per_category": per_category,
        "per_question": per_question,
    }


def main() -> None:
    """Run the BLEU/ROUGE evaluation pipeline."""
    parser = argparse.ArgumentParser(description="BLEU/ROUGE evaluation for RAG chatbot")
    parser.add_argument(
        "--mode",
        choices=["endpoint", "direct"],
        default="direct",
        help="How to query the RAG system",
    )
    parser.add_argument(
        "--endpoint",
        default="http://localhost:5000",
        help="Base URL for the Flask endpoint (endpoint mode only)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=4.5,
        help="Seconds to wait between queries (rate limiting/throttling prevention)",
    )
    args = parser.parse_args()

    # Ensure NLTK data is available
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)

    # Select query function
    if args.mode == "endpoint":
        get_answer = lambda q: query_rag_endpoint(q, args.endpoint)
    else:
        get_answer = query_rag_direct

    entries = load_golden_dataset()
    logger.info("Starting evaluation of %d entries in '%s' mode", len(entries), args.mode)

    results = []
    for i, entry in enumerate(entries):
        logger.info("[%d/%d] Evaluating: %s", i + 1, len(entries), entry["id"])
        result = evaluate_single(entry, get_answer)
        results.append(result)
        # Apply delay in both modes to avoid API rate limits
        if args.delay > 0:
            time.sleep(args.delay)

    aggregated = aggregate_results(results)

    # Save results
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(aggregated, f, indent=2, ensure_ascii=False)

    logger.info("Results saved to %s", RESULTS_PATH)

    # Print summary
    agg = aggregated["aggregate"]
    print("\n" + "=" * 60)
    print("BLEU/ROUGE EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total evaluated:    {aggregated['metadata']['total_evaluated']}")
    print(f"ROUGE-L F1 (mean):  {agg['rouge_l_fmeasure']['mean']}")
    print(f"ROUGE-L F1 (median):{agg['rouge_l_fmeasure']['median']}")
    print(f"BLEU (mean):        {agg['bleu']['mean']}")
    print(f"BLEU (median):      {agg['bleu']['median']}")
    if agg["ood_accuracy"] is not None:
        print(f"OOD Accuracy:       {agg['ood_accuracy']:.1%}")
    is_pass = (agg['rouge_l_fmeasure']['mean'] or 0) > 0.45
    print(f"\nTarget: ROUGE-L > 0.45 → {'PASS ✓' if is_pass else 'BELOW TARGET ✗'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
