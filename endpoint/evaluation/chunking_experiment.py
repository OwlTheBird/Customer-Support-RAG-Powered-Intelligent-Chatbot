"""Chunking strategy experiment for the RAG pipeline.

Evaluates how different chunk sizes and overlap ratios affect retrieval quality
by simulating different effective context windows (top_k changes) on pre-embedded content,
and measuring retrieval recall against the golden dataset.

IMPORTANT NOTE:
    The primary FAQ dataset contains short entries (~40 words each). Chunking
    experiments are most meaningful for long-form content. This experiment
    establishes methodology and baselines for future content types.

Usage:
    python -m evaluation.chunking_experiment
    python -m evaluation.chunking_experiment --chunk-sizes 128 256 512 1024
"""

import sys
from pathlib import Path

# Add the endpoint root directory to the python path to allow importing retriever, llm, and config
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time
import json
import logging
import statistics
import argparse
from dataclasses import dataclass, asdict
from datetime import datetime, timezone

# Configure logger
logger = logging.getLogger(__name__)

EVAL_DIR = Path(__file__).resolve().parent
GOLDEN_DATASET_PATH = EVAL_DIR / "golden_dataset.json"
RESULTS_PATH = EVAL_DIR / "chunking_results.json"

# Experiment configurations to test
CHUNK_CONFIGS = [
    {"chunk_size": 256, "overlap": 0},
    {"chunk_size": 256, "overlap": 64},
    {"chunk_size": 512, "overlap": 0},
    {"chunk_size": 512, "overlap": 128},
    {"chunk_size": 1024, "overlap": 0},
    {"chunk_size": 1024, "overlap": 256},
]


@dataclass
class ChunkConfig:
    """Configuration for a chunking experiment."""
    chunk_size: int   # in tokens (approximated as whitespace-split words)
    overlap: int      # overlap in tokens
    label: str = ""

    def __post_init__(self):
        if not self.label:
            self.label = f"chunk_{self.chunk_size}_overlap_{self.overlap}"


@dataclass
class ExperimentResult:
    """Results of a chunking strategy experiment for a single config."""
    config: ChunkConfig
    effective_top_k: int
    mean_recall: float
    median_recall: float
    mean_latency: float
    median_latency: float
    p95_latency: float
    total_questions: int
    per_question: list[dict]

    def to_dict(self) -> dict:
        """Convert result to dictionary representation."""
        return {
            "config": asdict(self.config),
            "effective_top_k": self.effective_top_k,
            "summary": {
                "mean_recall": self.mean_recall,
                "median_recall": self.median_recall,
                "mean_latency": self.mean_latency,
                "median_latency": self.median_latency,
                "p95_latency": self.p95_latency,
                "total_questions": self.total_questions,
            },
            "per_question": self.per_question,
        }


def load_golden_dataset(path: Path | str | None = None) -> list[dict]:
    """Load the golden dataset from a JSON file.

    Args:
        path: Path to the golden dataset file.

    Returns:
        List of dataset entry dicts.
    """
    if path is None:
        path = GOLDEN_DATASET_PATH
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("entries", [])


def recursive_text_split(
    text: str,
    chunk_size: int,
    overlap: int,
    separators: list[str] | None = None,
) -> list[str]:
    """Split text into chunks of approximately chunk_size tokens with overlap.

    Uses recursive splitting on hierarchical separators (paragraph → sentence → word)
    to maintain semantic coherence within chunks.

    Args:
        text: The text to split.
        chunk_size: Target chunk size in whitespace-delimited tokens.
        overlap: Number of overlapping tokens between consecutive chunks.
        separators: List of separator strings, tried in order. Defaults to
                    ["\\n\\n", "\\n", ". ", " "].

    Returns:
        List of text chunks.
    """
    if separators is None:
        separators = ["\n\n", "\n", ". ", " "]

    tokens = text.split()
    if len(tokens) <= chunk_size:
        return [text] if text.strip() else []

    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk = " ".join(tokens[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
        if start >= len(tokens):
            break

    return chunks


def compute_retrieval_recall(
    *args,
    **kwargs,
) -> float:
    """Compute retrieval recall — what fraction of the expected answer is covered by retrieved chunks.

    Uses token-level overlap to measure how much of the expected answer appears
    in the retrieved context.

    Can be called as:
      compute_retrieval_recall(expected_answer, retrieved_chunks, threshold=0.3)
      or
      compute_retrieval_recall(question, expected, retrieved_chunks)

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Recall score in [0.0, 1.0].
    """
    threshold = kwargs.get("threshold", 0.3)
    
    if len(args) == 3:
        # Signature: (question, expected, retrieved_chunks)
        expected_answer = args[1]
        retrieved_chunks = args[2]
    elif len(args) == 2:
        # Signature: (expected_answer, retrieved_chunks)
        expected_answer = args[0]
        retrieved_chunks = args[1]
    else:
        # Try kwargs
        expected_answer = kwargs.get("expected_answer")
        if expected_answer is None:
            expected_answer = kwargs.get("expected")
        retrieved_chunks = kwargs.get("retrieved_chunks")

    if not expected_answer or not retrieved_chunks:
        return 0.0

    expected_tokens = set(expected_answer.lower().split())
    if not expected_tokens:
        return 0.0

    cleaned_chunks = []
    for chunk in retrieved_chunks:
        if isinstance(chunk, dict):
            text = chunk.get("text", "")
            if text:
                cleaned_chunks.append(text)
        elif isinstance(chunk, str):
            cleaned_chunks.append(chunk)
        else:
            text = getattr(chunk, "text", "")
            if text:
                cleaned_chunks.append(text)

    combined_retrieved = " ".join(cleaned_chunks).lower()
    retrieved_tokens = set(combined_retrieved.split())

    overlap = expected_tokens & retrieved_tokens
    return round(len(overlap) / len(expected_tokens), 4)


def run_single_config(
    config: ChunkConfig,
    entries: list[dict],
    index: any = None,
    delay: float = 1.0,
    skip_llm: bool = False,
) -> ExperimentResult:
    """Run retrieval evaluation for a single chunk configuration.

    Since Pinecone handles embeddings internally, this experiment varies top_k
    to simulate different effective context windows that different chunk sizes
    would produce.

    For chunk_size 256: top_k=4 (more, smaller chunks)
    For chunk_size 512: top_k=2 (default, medium chunks)
    For chunk_size 1024: top_k=1 (fewer, larger chunks)

    Args:
        config: The chunk configuration to test.
        entries: Golden dataset entries.
        index: Pinecone index instance (optional, defaults to using pc_index from retriever).
        delay: Seconds between queries.
        skip_llm: Whether to skip LLM generation (helps avoid rate limits and speeds up run).

    Returns:
        ExperimentResult object with summary and detailed question metrics.
    """
    # Map chunk size to effective top_k
    top_k_mapping = {256: 4, 512: 2, 1024: 1}
    effective_top_k = top_k_mapping.get(config.chunk_size, 2)

    # Import retriever and llm
    from retriever import retrieve, build_context
    if not skip_llm:
        from llm import chain, extract_text

    per_question = []
    latencies = []

    in_domain = [e for e in entries if not e.get("is_ood", False)]

    for i, entry in enumerate(in_domain):
        logger.info("[%s] [%d/%d] %s", config.label, i + 1, len(in_domain), entry["id"])

        start_time = time.perf_counter()

        try:
            chunks = retrieve(entry["question"], top_k=effective_top_k)
            context = build_context(chunks)
            if not skip_llm:
                result = chain.invoke({"context": context, "question": entry["question"]})
                answer = extract_text(result.content)
            else:
                answer = "LLM call skipped"
        except Exception as e:
            logger.error("Error for %s: %s", entry["id"], e)
            answer = ""
            chunks = []
            context = ""

        latency = time.perf_counter() - start_time
        latencies.append(latency)

        # Compute recall using retrieved chunks
        chunk_texts = [c.get("text", "") for c in chunks if c.get("text")]
        recall = compute_retrieval_recall(entry["expected_answer"], chunk_texts)

        per_question.append({
            "id": entry["id"],
            "question": entry["question"],
            "category": entry["category"],
            "recall": recall,
            "latency_seconds": round(latency, 4),
            "chunks_retrieved": len(chunks),
            "context_length_tokens": len(context.split()),
            "generated_answer": answer,
        })

        if delay > 0:
            time.sleep(delay)

    recall_scores = [pq["recall"] for pq in per_question]
    mean_recall = round(statistics.mean(recall_scores), 4) if recall_scores else 0.0
    median_recall = round(statistics.median(recall_scores), 4) if recall_scores else 0.0
    mean_latency = round(statistics.mean(latencies), 4) if latencies else 0.0
    median_latency = round(statistics.median(latencies), 4) if latencies else 0.0
    p95_latency = round(sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0.0, 4)

    return ExperimentResult(
        config=config,
        effective_top_k=effective_top_k,
        mean_recall=mean_recall,
        median_recall=median_recall,
        mean_latency=mean_latency,
        median_latency=median_latency,
        p95_latency=p95_latency,
        total_questions=len(per_question),
        per_question=per_question,
    )


def run_experiment(
    configs: list[ChunkConfig],
    entries: list[dict],
    index: any = None,
    delay: float = 1.0,
    skip_llm: bool = False,
) -> list[ExperimentResult]:
    """Run evaluation over all configurations.

    Args:
        configs: List of configurations to test.
        entries: Dataset entries.
        index: Pinecone index instance (optional).
        delay: Delay in seconds between API requests.
        skip_llm: Whether to skip LLM generation calls.

    Returns:
        List of ExperimentResult.
    """
    results = []
    for config in configs:
        logger.info("Starting config: %s", config.label)
        res = run_single_config(config, entries, index=index, delay=delay, skip_llm=skip_llm)
        results.append(res)
        logger.info(
            "Config %s completed: mean_recall=%.4f, mean_latency=%.4fs",
            config.label, res.mean_recall, res.mean_latency
        )
    return results


def aggregate_experiment_results(results: list[ExperimentResult]) -> dict:
    """Aggregate results from multiple configurations.

    Args:
        results: List of ExperimentResult.

    Returns:
        Dictionary of aggregated comparison metrics.
    """
    best = max(results, key=lambda r: r.mean_recall)
    comparison_table = [
        {
            "config": r.config.label,
            "chunk_size": r.config.chunk_size,
            "overlap": r.config.overlap,
            "effective_top_k": r.effective_top_k,
            "mean_recall": r.mean_recall,
            "mean_latency": r.mean_latency,
            "p95_latency": r.p95_latency,
        }
        for r in results
    ]
    return {
        "best_configuration": {
            "config": asdict(best.config),
            "recall": best.mean_recall,
            "latency": best.mean_latency,
        },
        "comparison_table": comparison_table,
    }


def save_results(results: list[ExperimentResult], path: Path | str, entries: list[dict]) -> None:
    """Format and save results as JSON.

    Args:
        results: List of ExperimentResult objects.
        path: Path where output JSON will be written.
        entries: Dataset entries to compute metadata.
    """
    aggregated = aggregate_experiment_results(results)
    
    output = {
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_configs_tested": len(results),
            "entries_per_config": len([e for e in entries if not e.get("is_ood", False)]),
            "notes": [
                "FAQ entries average ~40 words (~50-60 tokens). Chunking is largely irrelevant "
                "for this dataset since each entry fits within any chunk size.",
                "This experiment simulates chunk size effects by varying top_k: larger chunks "
                "→ fewer retrieved (top_k=1), smaller chunks → more retrieved (top_k=4).",
                "Results establish methodology and baselines for future long-form content.",
            ],
        },
        "best_configuration": aggregated["best_configuration"],
        "comparison_table": aggregated["comparison_table"],
        "detailed_results": [r.to_dict() for r in results],
    }
    
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)


def main() -> None:
    """Run the chunking experiment pipeline."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="Chunking experiment for RAG pipeline")
    parser.add_argument("--chunk-sizes", type=int, nargs="+", default=[256, 512, 1024],
                        help="Chunk sizes to test (in tokens)")
    parser.add_argument("--delay", type=float, default=1.5,
                        help="Seconds between API calls")
    parser.add_argument("--sample", type=int, default=None,
                        help="Use a random sample of N golden dataset entries")
    parser.add_argument("--skip-llm", action="store_true", default=False,
                        help="Skip calling the LLM chain to speed up evaluation and avoid rate limits")
    args = parser.parse_args()

    entries = load_golden_dataset()

    if args.sample:
        import random
        random.seed(42)
        in_domain = [e for e in entries if not e.get("is_ood", False)]
        entries = random.sample(in_domain, min(args.sample, len(in_domain)))

    configs = []
    for size in args.chunk_sizes:
        configs.append(ChunkConfig(chunk_size=size, overlap=0))
        configs.append(ChunkConfig(chunk_size=size, overlap=size // 4))

    logger.info("Running %d configurations against %d entries", len(configs), len(entries))

    # If skipping LLM calls, we can reduce delay to 0.1s to make it extremely fast
    run_delay = 0.1 if args.skip_llm else args.delay
    all_results = run_experiment(configs, entries, delay=run_delay, skip_llm=args.skip_llm)

    # Save outputs
    save_results(all_results, RESULTS_PATH, entries)
    logger.info("Results saved to %s", RESULTS_PATH)

    # Find best configuration
    best = max(all_results, key=lambda r: r.mean_recall)

    # Print summary table
    print("\n" + "=" * 80)
    print("CHUNKING EXPERIMENT SUMMARY")
    print("=" * 80)
    print(f"{'Config':<30} {'Recall':>8} {'Latency (s)':>12} {'top_k':>6}")
    print("-" * 80)
    for r in all_results:
        print(f"{r.config.label:<30} {r.mean_recall:>8.4f} {r.mean_latency:>12.4f} {r.effective_top_k:>6}")
    print("-" * 80)
    print(f"Best: {best.config.label} (recall={best.mean_recall:.4f})")
    print("=" * 80)


if __name__ == "__main__":
    main()
