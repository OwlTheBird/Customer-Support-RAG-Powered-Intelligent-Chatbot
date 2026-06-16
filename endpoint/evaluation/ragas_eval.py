"""RAGAS semantic evaluation of the RAG chatbot.

Evaluates faithfulness, answer relevance, and context precision using
the RAGAS framework with Gemini as the LLM-judge.

IMPORTANT LIMITATION:
    Using Gemini to judge Gemini-generated answers introduces circular bias.
    The judge model may systematically rate its own outputs more favorably.
    See metadata.limitations in the output for details.

Usage:
    python -m evaluation.ragas_eval
    python -m evaluation.ragas_eval --endpoint http://localhost:5000
    python -m evaluation.ragas_eval --sample 10  # Evaluate a random sample
"""

import argparse
import json
import logging
import math
import os
import random
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List
import requests
from datasets import Dataset

# Add parent to path for config access
EVAL_DIR = Path(__file__).resolve().parent
PROJECT_DIR = EVAL_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from config import AI_API_KEY

logger = logging.getLogger(__name__)

GOLDEN_DATASET_PATH = EVAL_DIR / "golden_dataset.json"
RESULTS_PATH = EVAL_DIR / "ragas_results.json"


def load_golden_dataset(path: Path = GOLDEN_DATASET_PATH) -> List[Dict[str, Any]]:
    """Load the golden dataset from the JSON file.

    Args:
        path: Path to the golden dataset JSON file.

    Returns:
        List of entries.
    """
    logger.info("Loading golden dataset from %s", path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("entries", [])


def collect_rag_responses(
    entries: List[Dict[str, Any]],
    endpoint_url: str = "http://localhost:5000",
    delay: float = 4.5,
    max_retries: int = 5,
) -> List[Dict[str, Any]]:
    """Query the RAG endpoint and collect responses with retrieved contexts.

    Args:
        entries: Golden dataset entries (in-domain only — skip OOD for RAGAS).
        endpoint_url: Base URL for the Flask app.
        delay: Seconds between queries for rate limiting.
        max_retries: Maximum number of retries for rate limits or network issues.

    Returns:
        List of dicts with keys: question, answer, contexts, ground_truth, category, id.
    """
    responses = []
    in_domain_entries = [e for e in entries if not e.get("is_ood")]

    for i, entry in enumerate(in_domain_entries):
        logger.info("[%d/%d] Querying RAG endpoint for: %s", i + 1, len(in_domain_entries), entry["id"])
        
        backoff = 2.0
        success = False
        data = {}
        error_msg = ""
        
        for attempt in range(max_retries):
            try:
                resp = requests.get(
                    f"{endpoint_url}/test",
                    params={"user_input": entry["question"]},
                    timeout=30,
                )
                if resp.status_code in (429, 503):
                    sleep_time = backoff + random.uniform(0, 1)
                    logger.warning(
                        "Attempt %d got status %d. Retrying in %.2fs...",
                        attempt + 1,
                        resp.status_code,
                        sleep_time,
                    )
                    time.sleep(sleep_time)
                    backoff *= 2
                    continue
                
                resp.raise_for_status()
                data = resp.json()
                success = True
                break
            except requests.RequestException as e:
                error_msg = str(e)
                sleep_time = backoff + random.uniform(0, 1)
                logger.warning(
                    "Attempt %d failed: %s. Retrying in %.2fs...",
                    attempt + 1,
                    e,
                    sleep_time,
                )
                time.sleep(sleep_time)
                backoff *= 2

        if success:
            # Extract context texts from chunks
            contexts = [
                chunk.get("text", "")
                for chunk in data.get("chunks", [])
                if chunk.get("text")
            ]
            responses.append({
                "id": entry["id"],
                "question": entry["question"],
                "answer": data.get("answer", ""),
                "contexts": contexts,
                "ground_truth": entry["expected_answer"],
                "category": entry["category"],
            })
        else:
            logger.error("Failed to retrieve response for '%s' after %d attempts: %s", entry["id"], max_retries, error_msg)
            responses.append({
                "id": entry["id"],
                "question": entry["question"],
                "answer": "",
                "contexts": [],
                "ground_truth": entry["expected_answer"],
                "category": entry["category"],
                "error": error_msg,
            })

        time.sleep(delay)

    return responses


def build_ragas_dataset(responses: List[Dict[str, Any]]) -> Dataset:
    """Convert collected responses into a HuggingFace Dataset for RAGAS.

    RAGAS expects a Dataset with columns:
    - question (str)
    - answer (str)
    - contexts (list[str])
    - ground_truth (str)

    Args:
        responses: List of response dicts from collect_rag_responses.

    Returns:
        HuggingFace Dataset ready for RAGAS evaluation.
    """
    # Filter out any entries that had errors
    valid = [r for r in responses if "error" not in r]

    return Dataset.from_dict({
        "question": [r["question"] for r in valid],
        "answer": [r["answer"] for r in valid],
        "contexts": [r["contexts"] for r in valid],
        "ground_truth": [r["ground_truth"] for r in valid],
    })


from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.outputs import ChatResult

import asyncio
import re

GEMINI_DELAY_BETWEEN_REQUESTS = 4.5  # About 13 RPM to be safe

class AsyncRateLimiter:
    def __init__(self, delay: float):
        self.delay = delay
        self._lock = None
        self.last_request_time = 0.0

    @property
    def lock(self):
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def wait(self):
        async with self.lock:
            now = time.time()
            elapsed = now - self.last_request_time
            if elapsed < self.delay:
                await asyncio.sleep(self.delay - elapsed)
            self.last_request_time = time.time()

GEMINI_ASYNC_LIMITER = AsyncRateLimiter(GEMINI_DELAY_BETWEEN_REQUESTS)

class GeminiRagasWrapper(ChatGoogleGenerativeAI):
    """Custom wrapper for ChatGoogleGenerativeAI to handle strict rate limiting."""
    
    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        for attempt in range(15):
            await GEMINI_ASYNC_LIMITER.wait()
            try:
                res = await super(GeminiRagasWrapper, self)._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)
                return res
            except Exception as e:
                msg = str(e)
                if "429" in msg or "Quota" in msg or "exhausted" in msg.lower():
                    logger.warning(f"Caught 429 Rate Limit error. Sleeping for 20s. Error: {e}")
                    await asyncio.sleep(20.0)
                else:
                    raise e
        
        await GEMINI_ASYNC_LIMITER.wait()
        return await super(GeminiRagasWrapper, self)._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)


def run_ragas_evaluation(dataset: Dataset) -> Any:
    """Run RAGAS evaluation metrics on the dataset.

    Configures Gemini as the LLM-judge via LangChain.

    Args:
        dataset: HuggingFace Dataset with RAGAS-compatible columns.

    Returns:
        RAGAS result dict with metric scores.

    Raises:
        ValueError: If dataset is empty.
    """
    if len(dataset) == 0:
        raise ValueError("Cannot evaluate an empty dataset")

    # Import RAGAS components
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_precision
    from ragas.run_config import RunConfig
    
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    # Read API key explicitly from the environment
    gemini_api_key = os.getenv("GEMINI_EVAL_API_KEY", os.getenv("AI_API_KEY_2", os.getenv("AI_API_KEY")))

    llm = GeminiRagasWrapper(
        model="gemini-3.1-flash-lite",
        google_api_key=gemini_api_key,
        temperature=0,
        max_retries=0,
    )

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=gemini_api_key,
        max_retries=10,
    )

    run_config = RunConfig(
        max_workers=1,      # Limit concurrency to 1 to prevent rate limits
        timeout=600,        # Increase timeout to allow backoff retries to succeed
        max_retries=20,
        max_wait=90,
    )

    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=llm,
        embeddings=embeddings,
        run_config=run_config,
    )

    return result


def sanitize_score(val: Any) -> float:
    """Sanitize score value from Ragas result.
    
    Converts to float, replaces NaN/None with 0.0.

    Args:
        val: Raw metric score.

    Returns:
        Float value of the metric score.
    """
    if val is None:
        return 0.0
    try:
        f_val = float(val)
        if math.isnan(f_val) or math.isinf(f_val):
            return 0.0
        return f_val
    except (ValueError, TypeError):
        return 0.0


def aggregate_ragas_results(
    ragas_result: Any,
    responses: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Aggregate RAGAS results with per-category breakdowns and metadata.

    Args:
        ragas_result: Raw RAGAS evaluation result.
        responses: Original response list (for category information).

    Returns:
        Structured results dict with metadata, aggregate, and per-category breakdowns.
    """
    # Extract per-question scores from RAGAS result DataFrame
    df = ragas_result.to_pandas()
    valid_responses = [r for r in responses if "error" not in r]

    per_question = []
    for i, row in df.iterrows():
        resp = valid_responses[i] if i < len(valid_responses) else {}
        per_question.append({
            "id": resp.get("id", f"Q-{i}"),
            "question": row.get("question", ""),
            "category": resp.get("category", "unknown"),
            "faithfulness": round(sanitize_score(row.get("faithfulness")), 4),
            "answer_relevancy": round(sanitize_score(row.get("answer_relevancy")), 4),
            "context_precision": round(sanitize_score(row.get("context_precision")), 4),
        })

    # Per-category breakdown
    categories = set(pq["category"] for pq in per_question)
    per_category = {}
    for cat in sorted(categories):
        cat_items = [pq for pq in per_question if pq["category"] == cat]
        per_category[cat] = {
            "count": len(cat_items),
            "faithfulness_mean": round(statistics.mean([x["faithfulness"] for x in cat_items]), 4) if cat_items else 0.0,
            "answer_relevancy_mean": round(statistics.mean([x["answer_relevancy"] for x in cat_items]), 4) if cat_items else 0.0,
            "context_precision_mean": round(statistics.mean([x["context_precision"] for x in cat_items]), 4) if cat_items else 0.0,
        }

    all_faith = [pq["faithfulness"] for pq in per_question]
    all_rel = [pq["answer_relevancy"] for pq in per_question]
    all_prec = [pq["context_precision"] for pq in per_question]

    mean_faith = statistics.mean(all_faith) if all_faith else 0.0
    mean_rel = statistics.mean(all_rel) if all_rel else 0.0
    mean_prec = statistics.mean(all_prec) if all_prec else 0.0

    return {
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_evaluated": len(per_question),
            "judge_model": "gemini-3.1-flash-lite",
            "ragas_version": "0.4.3",
            "limitations": [
                "Circular bias: Gemini is used as the judge to evaluate Gemini-generated answers.",
                "Small sample size may not be statistically representative.",
            ],
        },
        "aggregate": {
            "faithfulness": {
                "mean": round(mean_faith, 4),
                "median": round(statistics.median(all_faith) if all_faith else 0.0, 4),
                "std": round(statistics.stdev(all_faith), 4) if len(all_faith) > 1 else None,
            },
            "answer_relevancy": {
                "mean": round(mean_rel, 4),
                "median": round(statistics.median(all_rel) if all_rel else 0.0, 4),
                "std": round(statistics.stdev(all_rel), 4) if len(all_rel) > 1 else None,
            },
            "context_precision": {
                "mean": round(mean_prec, 4),
                "median": round(statistics.median(all_prec) if all_prec else 0.0, 4),
                "std": round(statistics.stdev(all_prec), 4) if len(all_prec) > 1 else None,
            },
        },
        "success_criteria": {
            "context_precision_target": 0.8,
            "context_precision_pass": (mean_prec > 0.8),
            "answer_relevancy_target": 0.8,
            "answer_relevancy_pass": (mean_rel > 0.8),
            "faithfulness_target": 0.7,
            "faithfulness_pass": (mean_faith > 0.7),
        },
        "per_category": per_category,
        "per_question": per_question,
    }


def main() -> None:
    """Run the RAGAS evaluation pipeline."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="RAGAS evaluation for RAG chatbot")
    parser.add_argument("--endpoint", default="http://localhost:5000",
                        help="Base URL for the Flask endpoint")
    parser.add_argument("--delay", type=float, default=4.5,
                        help="Seconds between queries (rate limiting)")
    parser.add_argument("--sample", type=int, default=None,
                        help="Evaluate a random sample of N entries (for faster testing)")
    args = parser.parse_args()

    # Load golden dataset
    entries = load_golden_dataset()

    # Skip OOD questions for Ragas evaluation (it only evaluates in-domain)
    in_domain = [e for e in entries if not e.get("is_ood")]
    logger.info("Found %d in-domain entries in golden dataset (skipped OOD)", len(in_domain))

    # Optional sampling for faster iteration
    if args.sample and args.sample < len(in_domain):
        random.seed(42)
        entries_to_eval = random.sample(in_domain, args.sample)
        logger.info("Sampled %d entries for evaluation", len(entries_to_eval))
    else:
        entries_to_eval = in_domain
        logger.info("Evaluating all %d in-domain entries", len(entries_to_eval))

    # Collect RAG responses
    logger.info("Collecting RAG responses from %s", args.endpoint)
    responses = collect_rag_responses(entries_to_eval, args.endpoint, args.delay)

    valid_count = sum(1 for r in responses if "error" not in r)
    logger.info("Collected %d valid responses out of %d", valid_count, len(responses))

    if valid_count == 0:
        logger.error("No valid responses collected. Cannot run RAGAS evaluation.")
        return

    # Build RAGAS dataset
    dataset = build_ragas_dataset(responses)

    # Run RAGAS evaluation
    logger.info("Running RAGAS evaluation on %d samples...", len(dataset))
    ragas_result = run_ragas_evaluation(dataset)

    # Aggregate and save
    results = aggregate_ragas_results(ragas_result, responses)

    # Ensure output directory exists
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info("RAGAS results saved to %s", RESULTS_PATH)

    # Print summary
    agg = results["aggregate"]
    sc = results["success_criteria"]
    print("\n" + "=" * 60)
    print("RAGAS EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total evaluated:      {results['metadata']['total_evaluated']}")
    print(f"Faithfulness (mean):  {agg['faithfulness']['mean']:.4f}  "
          f"(target > 0.7: {'PASS' if sc['faithfulness_pass'] else 'BELOW TARGET'})")
    print(f"Answer Relevancy:     {agg['answer_relevancy']['mean']:.4f}  "
          f"(target > 0.8: {'PASS' if sc['answer_relevancy_pass'] else 'BELOW TARGET'})")
    print(f"Context Precision:    {agg['context_precision']['mean']:.4f}  "
          f"(target > 0.8: {'PASS' if sc['context_precision_pass'] else 'BELOW TARGET'})")
    print(f"\n[INFO] JUDGE LLM: Gemini 3.1 Flash Lite evaluates Gemini — contains circular bias.")
    print("=" * 60)


if __name__ == "__main__":
    main()
