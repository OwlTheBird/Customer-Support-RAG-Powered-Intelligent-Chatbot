"""Curation script to generate the golden dataset from Hugging Face.

This script loads the customer support FAQs dataset, categorizes each question
into one of several predefined domains using keyword matching, samples the
records to generate a representative benchmark, appends out-of-domain refusal
examples, validates the output schema, and saves the final golden dataset.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Set

from datasets import load_dataset  # type: ignore

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Predefined categories and keyword matching rules
CATEGORY_KEYWORDS: Dict[str, List[str]] = {
    "account_management": ["account", "password", "login", "sign up", "register", "profile", "username", "email", "verify"],
    "orders_and_shipping": ["order", "ship", "deliver", "track", "package", "dispatch", "transit"],
    "returns_and_refunds": ["return", "refund", "exchange", "cancel", "money back", "replacement"],
    "payments_and_billing": ["pay", "billing", "invoice", "charge", "credit card", "debit", "transaction", "price", "cost"],
    "product_information": ["product", "item", "feature", "specification", "size", "color", "availability", "stock"],
    "technical_support": ["technical", "error", "bug", "crash", "not working", "issue", "problem", "fix", "troubleshoot"],
    "subscription_and_membership": ["subscription", "membership", "plan", "upgrade", "downgrade", "renew", "cancel subscription"],
    "general_inquiries": ["how", "what", "where", "when", "who", "policy", "contact", "hours", "support"],
}

# 36 Out-of-Domain (OOD) questions covering 6 distinct subcategories
OOD_QUESTIONS: List[str] = [
    # Geography/History
    "What is the capital of France?",
    "Who was the first president of the United States?",
    "In which year did World War II end?",
    "What is the longest river in the world?",
    "Which country has the largest population?",
    "Where are the Pyramids of Giza located?",
    
    # Science/Math
    "What is the speed of light?",
    "How many planets are in the solar system?",
    "What is the chemical symbol for gold?",
    "What is the value of pi to 5 decimal places?",
    "How do you calculate the area of a circle?",
    "What is the state of matter of water at room temperature?",
    
    # Entertainment/Sports
    "Who won the 2024 Super Bowl?",
    "Who directed the movie Inception?",
    "Which actor played Iron Man in the Marvel Cinematic Universe?",
    "How many World Cup titles has Brazil won?",
    "Who wrote the play Hamlet?",
    "What is the name of the lead singer of U2?",
    
    # Personal/Opinion
    "What is the meaning of life?",
    "Do you think artificial intelligence will replace humans?",
    "What is your favorite color?",
    "How do you feel today?",
    "What is the best movie of all time?",
    "Do you believe in ghosts?",
    
    # Unrelated Business
    "How do I file a tax return?",
    "What are the current mortgage interest rates?",
    "How can I start a small business in California?",
    "What is the stock price of Apple today?",
    "How do I apply for a passport?",
    "What is the process for registering a trademark?",
    
    # Medical/Legal
    "What are the symptoms of diabetes?",
    "How is high blood pressure treated?",
    "What is the statute of limitations for breach of contract?",
    "Can you give me medical advice for a sore throat?",
    "How do I write a legally binding will?",
    "What are the side effects of aspirin?"
]


def classify_question(question: str) -> str:
    """Classify a question into a category based on keyword matching.

    Args:
        question: The question text to classify.

    Returns:
        The snake_case category string. Falls back to "general_inquiries".
    """
    q_lower = question.lower()
    scores: Dict[str, int] = {}
    for category, keywords in CATEGORY_KEYWORDS.items():
        scores[category] = sum(1 for kw in keywords if kw in q_lower)
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "general_inquiries"


def build_in_domain_entries(ds: Any, target_count: int = 75) -> List[Dict[str, Any]]:
    """Extract stratified in-domain Q&A pairs from the dataset.

    Args:
        ds: The HuggingFace dataset split.
        target_count: The target number of in-domain Q&A pairs to sample.

    Returns:
        List of golden dataset entries.
    """
    # 1. Deduplicate while keeping first occurrence
    seen: Set[str] = set()
    unique_records: List[Dict[str, str]] = []
    for row in ds:
        q = row["question"].strip()
        ans = row["answer"].strip()
        if q not in seen:
            seen.add(q)
            unique_records.append({"question": q, "answer": ans})

    # 2. Group by category
    category_groups: Dict[str, List[Dict[str, str]]] = {}
    for rec in unique_records:
        cat = classify_question(rec["question"])
        category_groups.setdefault(cat, []).append(rec)

    # Sort each group alphabetically by question to guarantee deterministic sampling
    for cat in category_groups:
        category_groups[cat].sort(key=lambda x: x["question"])

    categories = list(CATEGORY_KEYWORDS.keys())
    group_sizes = {cat: len(category_groups.get(cat, [])) for cat in categories}

    # 3. Stratified allocation
    # First assign minimum of 3 (or size if smaller)
    allocations = {cat: min(3, group_sizes[cat]) for cat in categories}
    current_total = sum(allocations.values())

    if current_total < target_count:
        # Determine remaining slots
        remaining_slots = target_count - current_total
        unsampled_sizes = {cat: group_sizes[cat] - allocations[cat] for cat in categories}
        total_unsampled = sum(unsampled_sizes.values())

        if total_unsampled > 0:
            # Proportional allocation of remaining slots
            float_allocations = {
                cat: (remaining_slots * unsampled_sizes[cat] / total_unsampled)
                for cat in categories
            }
            # Floor assignment
            floor_allocations = {cat: int(float_allocations[cat]) for cat in categories}
            allocated_sum = sum(floor_allocations.values())
            remainder_slots = remaining_slots - allocated_sum

            # Sort by remainder descending
            sorted_categories = sorted(
                categories,
                key=lambda cat: float_allocations[cat] - floor_allocations[cat],
                reverse=True
            )

            # Distribute remainder
            for i in range(remainder_slots):
                cat = sorted_categories[i]
                floor_allocations[cat] += 1

            # Update allocations
            for cat in categories:
                allocations[cat] += floor_allocations[cat]

    # 4. Extract entries
    selected_records: List[Dict[str, Any]] = []
    sorted_categories_for_output = sorted(categories)
    for cat in sorted_categories_for_output:
        count_to_take = allocations.get(cat, 0)
        records_in_cat = category_groups.get(cat, [])
        for rec in records_in_cat[:count_to_take]:
            selected_records.append({
                "question": rec["question"],
                "expected_answer": rec["answer"],
                "category": cat,
                "is_ood": False
            })

    # Sort alphabetically by question to guarantee final sorting order
    selected_records.sort(key=lambda x: x["question"])

    # Assign IDs
    final_entries: List[Dict[str, Any]] = []
    for idx, rec in enumerate(selected_records, 1):
        final_entries.append({
            "id": f"GD-{idx:03d}",
            "question": rec["question"],
            "expected_answer": rec["expected_answer"],
            "category": rec["category"],
            "is_ood": rec["is_ood"]
        })

    return final_entries


def build_ood_entries() -> List[Dict[str, Any]]:
    """Build the out-of-domain entries.

    Returns:
        List of out-of-domain entries.
    """
    entries: List[Dict[str, Any]] = []
    for idx, q in enumerate(OOD_QUESTIONS, 1):
        entries.append({
            "id": f"OOD-{idx:03d}",
            "question": q,
            "expected_answer": "I don't have that information.",
            "category": "out_of_domain",
            "is_ood": True
        })
    return entries


def validate_dataset(entries: List[Dict[str, Any]]) -> List[str]:
    """Validate the golden dataset for schema compliance.

    Args:
        entries: List of dataset entries.

    Returns:
        List of validation error strings. Empty list = valid.
    """
    errors: List[str] = []
    ids_seen: Set[str] = set()
    questions_seen: Set[str] = set()

    for i, entry in enumerate(entries):
        # Check required keys
        for key in ("id", "question", "expected_answer", "category", "is_ood"):
            if key not in entry:
                errors.append(f"Entry {i}: missing key '{key}'")

        # Check uniqueness of ID
        entry_id = entry.get("id")
        if entry_id in ids_seen:
            errors.append(f"Entry {i}: duplicate id '{entry_id}'")
        if entry_id:
            ids_seen.add(entry_id)

        # Check uniqueness of question
        q = entry.get("question")
        if q in questions_seen:
            errors.append(f"Entry {i}: duplicate question '{q}'")
        if q:
            questions_seen.add(q)

        # Check non-empty
        if not entry.get("question", "").strip():
            errors.append(f"Entry {i}: empty question")
        if not entry.get("expected_answer", "").strip():
            errors.append(f"Entry {i}: empty expected_answer")

        # Check OOD consistency
        if entry.get("is_ood"):
            if entry.get("expected_answer") != "I don't have that information.":
                errors.append(f"Entry {i}: OOD entry missing correct refusal phrase")
            if entry.get("category") != "out_of_domain":
                errors.append(f"Entry {i}: OOD entry category must be 'out_of_domain'")
        else:
            if entry.get("category") == "out_of_domain":
                errors.append(f"Entry {i}: In-domain entry cannot have category 'out_of_domain'")

    return errors


def main() -> None:
    """Main function to run curation and save output."""
    logger.info("Loading dataset from Hugging Face...")
    ds = load_dataset("MakTek/Customer_support_faqs_dataset", split="train")

    logger.info("Extracting and sampling in-domain entries...")
    in_domain_entries = build_in_domain_entries(ds, target_count=75)

    logger.info("Generating OOD entries...")
    ood_entries = build_ood_entries()

    all_entries = in_domain_entries + ood_entries

    logger.info("Validating entries...")
    errors = validate_dataset(all_entries)
    if errors:
        logger.error(f"Validation failed with {len(errors)} errors:")
        for err in errors:
            logger.error(f" - {err}")
        raise ValueError("Dataset validation failed.")

    logger.info("Validation passed.")

    output_path = Path(__file__).resolve().parent / "golden_dataset.json"
    output_data = {
        "metadata": {
            "version": "1.0.0",
            "created_at": datetime.utcnow().isoformat() + "Z",
            "source": "MakTek/Customer_support_faqs_dataset",
            "total_entries": len(all_entries),
            "in_domain_count": len(in_domain_entries),
            "ood_count": len(ood_entries),
            "categories": sorted(list(set(e["category"] for e in all_entries))),
        },
        "entries": all_entries
    }

    logger.info(f"Saving golden dataset to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    logger.info("Done.")


if __name__ == "__main__":
    main()
