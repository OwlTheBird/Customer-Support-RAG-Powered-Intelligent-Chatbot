"""Prompt engineering and hyperparameter tuning for the RAG chatbot.

Tests multiple prompt templates and LLM hyperparameter combinations
against the golden dataset to find the optimal configuration for
minimizing hallucinations and maximizing answer quality.

Usage:
    python -m evaluation.prompt_tuning
"""

import os
import json
import logging
import random
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from rouge_score import rouge_scorer

# Load environment variables
load_dotenv()

# Add parent directory for absolute/direct imports of config, retriever, llm
EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR.parent))

from llm import extract_text
from retriever import build_context, retrieve

# Configure logging to console using standard format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

GOLDEN_DATASET_PATH = EVAL_DIR / "golden_dataset.json"
RESULTS_PATH = EVAL_DIR / "prompt_tuning_results.json"

# Load API key 2 with fallback to primary API key
AI_API_KEY_2 = os.getenv("AI_API_KEY_2", os.getenv("AI_API_KEY"))

# Defined prompt variants
PROMPT_VARIANTS: dict[str, str] = {
    "baseline": (
        "You are a helpful customer support assistant. Use the context below to answer "
        "the customer's question.\n\n"
        "Guidelines:\n"
        "- Answer in a clear, friendly, and complete way — don't just copy the context word for word.\n"
        "- If the context covers the topic, expand on it naturally and helpfully.\n"
        "- If multiple context pieces are relevant, synthesize them into one coherent answer.\n"
        "- If the answer truly cannot be found in the context, reply exactly with: "
        "\"I don't have that information.\"\n\n"
        "Context:\n{context}\n\n"
        "Customer question: {question}\n"
        "Answer:"
    ),
    "strict_refusal": (
        "You are a customer support assistant. You MUST ONLY answer using the provided context.\n\n"
        "CRITICAL RULES:\n"
        "1. If the context does not contain information to answer the question, you MUST respond "
        "with EXACTLY: \"I don't have that information.\" — nothing else.\n"
        "2. Do NOT use any knowledge beyond the provided context.\n"
        "3. Do NOT guess, speculate, or infer information not explicitly stated in the context.\n"
        "4. Do NOT add disclaimers like 'based on the context' — just answer directly.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n"
        "Answer:"
    ),
    "concise": (
        "You are a concise customer support assistant. Answer the question using ONLY the "
        "provided context.\n\n"
        "Rules:\n"
        "- Keep answers under 3 sentences.\n"
        "- Be direct and factual.\n"
        "- If the context doesn't contain the answer, reply: \"I don't have that information.\"\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n"
        "Answer:"
    ),
}

# Hyperparameter values to test
TEMPERATURE_VALUES: list[float] = [0.0, 0.7]

PHASE1_GRID: list[dict] = [
    {"prompt_variant": pv, "temperature": t, "top_p": 0.95, "max_tokens": 512}
    for pv in PROMPT_VARIANTS.keys()
    for t in TEMPERATURE_VALUES
]


@dataclass
class TuningConfig:
    """A single prompt + hyperparameter configuration to test."""

    prompt_variant: str
    temperature: float
    top_p: float
    max_tokens: int
    label: str = ""

    def __post_init__(self):
        if not self.label:
            self.label = f"{self.prompt_variant}_t{self.temperature}_p{self.top_p}_m{self.max_tokens}"


def load_golden_dataset() -> List[dict]:
    """Loads the golden dataset from the JSON file.

    Returns:
        List[dict]: A list of dictionary entries from the golden dataset.

    Raises:
        FileNotFoundError: If the golden dataset file does not exist.
    """
    logger.info("Loading golden dataset from %s", GOLDEN_DATASET_PATH)
    if not GOLDEN_DATASET_PATH.exists():
        raise FileNotFoundError(f"Golden dataset not found at {GOLDEN_DATASET_PATH}")
    with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["entries"]


def load_and_sample_dataset() -> List[dict]:
    """Loads the golden dataset and samples exactly 3 in-domain and 15 out-of-domain questions.

    Returns:
        List[dict]: Sampled entries containing exactly 18 questions.
    """
    entries = load_golden_dataset()
    in_domain = [e for e in entries if not e["is_ood"]]
    ood = [e for e in entries if e["is_ood"]]

    random.seed(42)
    sampled_in = random.sample(in_domain, min(3, len(in_domain)))
    sampled_ood = random.sample(ood, min(15, len(ood)))

    logger.info(
        "Sampled %d in-domain and %d out-of-domain questions (total %d)",
        len(sampled_in),
        len(sampled_ood),
        len(sampled_in) + len(sampled_ood)
    )
    return sampled_in + sampled_ood


def create_chain(config: TuningConfig) -> Any:
    """Creates a LangChain chain with the specified configuration and AI_API_KEY_2.

    Args:
        config: The tuning configuration specifying prompt template and LLM parameters.

    Returns:
        Any: A LangChain runnable chain (prompt | llm).
    """
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite",
        google_api_key=AI_API_KEY_2,
        temperature=config.temperature,
        top_p=config.top_p,
        max_output_tokens=config.max_tokens,
    )

    prompt_template = PROMPT_VARIANTS[config.prompt_variant]
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant."),
            ("user", prompt_template),
        ]
    )

    return prompt | llm


def invoke_chain_with_retry(
    chain: Any,
    inputs: Dict[str, Any],
    max_retries: int = 5,
    initial_delay: float = 2.0
) -> Any:
    """Invokes the chain with exponential backoff retry for rate limits.

    Args:
        chain: The LangChain runnable to invoke.
        inputs: Dict of inputs to pass to the chain.
        max_retries: Maximum number of retries before raising.
        initial_delay: Initial sleep time in seconds.

    Returns:
        Any: The result of the chain invocation.

    Raises:
        Exception: If chain invocation fails after maximum retries.
    """
    delay = initial_delay
    for attempt in range(1, max_retries + 1):
        try:
            return chain.invoke(inputs)
        except Exception as e:
            if attempt == max_retries:
                logger.error(
                    "Failed to invoke chain after %d attempts: %s", max_retries, e
                )
                raise
            logger.warning(
                "Chain invocation attempt %d failed: %s. Retrying in %.1f seconds...",
                attempt,
                e,
                delay,
            )
            time.sleep(delay)
            delay *= 2.0


def evaluate_config(
    config: TuningConfig,
    entries: List[dict],
    delay: float = 4.5
) -> Dict[str, Any]:
    """Evaluates a single prompt+hyperparameter configuration against the golden dataset.

    Args:
        config: The configuration to test.
        entries: Golden dataset entries to evaluate.
        delay: Seconds to sleep between queries for rate limiting.

    Returns:
        Dict[str, Any]: Result dict with per-question scores and aggregate metrics.
    """
    chain = create_chain(config)
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    in_domain = [e for e in entries if not e["is_ood"]]
    ood = [e for e in entries if e["is_ood"]]

    per_question = []

    # Evaluate in-domain questions
    for entry in in_domain:
        try:
            chunks = retrieve(entry["question"], top_k=2)
            context = build_context(chunks)
            result = invoke_chain_with_retry(
                chain, {"context": context, "question": entry["question"]}
            )
            answer = extract_text(result.content)
        except Exception as e:
            logger.error("Error for in-domain question %s: %s", entry["id"], e)
            answer = ""

        rl = scorer.score(entry["expected_answer"], answer)["rougeL"]

        per_question.append(
            {
                "id": entry["id"],
                "is_ood": False,
                "category": entry["category"],
                "generated_answer": answer,
                "rouge_l_f1": round(rl.fmeasure, 4),
            }
        )

        time.sleep(delay)

    # Evaluate OOD questions (hallucination detection)
    ood_results = []
    for entry in ood:
        try:
            # For OOD, we provide retrieved context (which should be irrelevant)
            chunks = retrieve(entry["question"], top_k=2)
            context = build_context(chunks)
            result = invoke_chain_with_retry(
                chain, {"context": context, "question": entry["question"]}
            )
            answer = extract_text(result.content)
        except Exception as e:
            logger.error("Error for OOD question %s: %s", entry["id"], e)
            answer = ""

        refusal_phrases = [
            "i don't have that information",
            "i do not have that information",
            "i don't have enough information",
        ]
        is_refusal = any(phrase in answer.lower() for phrase in refusal_phrases)

        ood_results.append(
            {
                "id": entry["id"],
                "is_ood": True,
                "category": "out_of_domain",
                "generated_answer": answer,
                "correctly_refused": is_refusal,
            }
        )

        time.sleep(delay)

    per_question.extend(ood_results)

    # Aggregate
    rouge_scores = [pq["rouge_l_f1"] for pq in per_question if not pq.get("is_ood")]
    ood_correct = sum(1 for r in ood_results if r["correctly_refused"])
    hallucination_rate = 1 - (ood_correct / len(ood_results)) if ood_results else 0.0

    return {
        "config": asdict(config),
        "summary": {
            "rouge_l_mean": round(statistics.mean(rouge_scores), 4)
            if rouge_scores
            else 0.0,
            "rouge_l_median": round(statistics.median(rouge_scores), 4)
            if rouge_scores
            else 0.0,
            "ood_total": len(ood_results),
            "ood_correct_refusals": ood_correct,
            "hallucination_rate": round(hallucination_rate, 4),
            "in_domain_count": len(in_domain),
        },
        "per_question": per_question,
    }


def main() -> None:
    """Runs the prompt tuning experiment pipeline."""
    # Ensure random seed is set for loading/sampling dataset
    entries = load_and_sample_dataset()

    configs = [TuningConfig(**cfg) for cfg in PHASE1_GRID]
    logger.info(
        "Testing %d configurations against %d sampled entries", len(configs), len(entries)
    )

    all_results = []
    for i, config in enumerate(configs):
        logger.info("[%d/%d] Testing: %s", i + 1, len(configs), config.label)
        result = evaluate_config(config, entries, delay=4.5)
        all_results.append(result)
        logger.info(
            "  ROUGE-L=%.4f, Hallucination=%.2f%%",
            result["summary"]["rouge_l_mean"],
            result["summary"]["hallucination_rate"] * 100,
        )

    # Rank configurations: Primary (lowest hallucination rate), Secondary (highest ROUGE-L)
    ranked = sorted(
        all_results,
        key=lambda r: (
            r["summary"]["hallucination_rate"],
            -r["summary"]["rouge_l_mean"],
        ),
    )

    best = ranked[0]

    output = {
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_configs_tested": len(all_results),
            "entries_evaluated": len(entries),
            "in_domain_count": sum(1 for e in entries if not e.get("is_ood")),
            "ood_count": sum(1 for e in entries if e.get("is_ood")),
            "current_production_config": {
                "prompt": "baseline (single template in config.py)",
                "temperature": "default (unset)",
                "top_p": "default (unset)",
                "max_tokens": "default (unset)",
            },
        },
        "best_configuration": {
            "config": best["config"],
            "rouge_l_mean": best["summary"]["rouge_l_mean"],
            "hallucination_rate": best["summary"]["hallucination_rate"],
            "recommendation": (
                f"Use prompt variant '{best['config']['prompt_variant']}' with "
                f"temperature={best['config']['temperature']}, "
                f"top_p={best['config']['top_p']}, "
                f"max_tokens={best['config']['max_tokens']}"
            ),
        },
        "ranking": [
            {
                "rank": i + 1,
                "config_label": r["config"]["label"],
                "prompt_variant": r["config"]["prompt_variant"],
                "temperature": r["config"]["temperature"],
                "rouge_l_mean": r["summary"]["rouge_l_mean"],
                "hallucination_rate": r["summary"]["hallucination_rate"],
                "ood_correct": f"{r['summary']['ood_correct_refusals']}/{r['summary']['ood_total']}",
            }
            for i, r in enumerate(ranked)
        ],
        "detailed_results": all_results,
    }

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    logger.info("Results successfully saved to %s", RESULTS_PATH)

    # Print summary table using logger to comply with quality standards
    logger.info("=" * 90)
    logger.info("PROMPT TUNING SUMMARY")
    logger.info("=" * 90)
    logger.info(
        f"{'Rank':<5} {'Config':<45} {'ROUGE-L':>8} {'Halluc.%':>9} {'OOD':>8}"
    )
    logger.info("-" * 90)
    for r in output["ranking"]:
        logger.info(
            f"{r['rank']:<5} {r['config_label']:<45} "
            f"{r['rouge_l_mean']:>8.4f} {r['hallucination_rate']*100:>8.1f}% "
            f"{r['ood_correct']:>8}"
        )
    logger.info("-" * 90)
    logger.info(f"Best Configuration: {best['config']['label']}")
    logger.info(f"  ROUGE-L Mean: {best['summary']['rouge_l_mean']:.4f}")
    logger.info(f"  Hallucination Rate: {best['summary']['hallucination_rate']:.1%}")
    logger.info(
        f"  Target (0% hallucination): "
        f"{'PASS ✓' if best['summary']['hallucination_rate'] == 0 else 'FAIL ✗'}"
    )
    logger.info("=" * 90)


if __name__ == "__main__":
    main()
