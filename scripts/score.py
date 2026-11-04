#!/usr/bin/env python3
"""Score predictions against eval_all.jsonl.

For unanswerable questions: score 1 if prediction refuses, else 0.
For grounded questions: compute SQuAD-style token F1.
"""

import argparse
import glob
import json
import os
import re
import string
from collections import defaultdict


# Refusal phrases (English and Hindi)
REFUSAL_PHRASES = [
    'does not address',
    "doesn't address",
    "doesn't answer",
    'does not answer',
    'not covered',
    "can't answer",
    'cannot answer',
    'without guessing',
    'going beyond the text',
    'need the provision',
    'share the relevant provision',
    'not address',
    'उत्तर नहीं',
    'नहीं देता',
    'नहीं बताता',
    'संबंधित प्रावधान',
    'अनुमान',
    "does not answer",
    "not covered",
    "cannot",
    "नहीं",
    "उत्तर नहीं",
]


def normalize_text(text):
    """Lowercase and strip punctuation."""
    if not text:
        return ""
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text.strip()


def tokenize(text):
    """Simple whitespace tokenization."""
    text = normalize_text(text)
    return text.split()


def compute_f1(pred_tokens, gold_tokens):
    """Compute SQuAD-style F1 score."""
    if not pred_tokens or not gold_tokens:
        return 0.0

    pred_set = set(pred_tokens)
    gold_set = set(gold_tokens)

    common = pred_set & gold_set
    num_common = len(common)

    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1


def is_refusal(text):
    """Check if text contains refusal phrases."""
    if not text:
        return False
    text_lower = text.lower()
    for phrase in REFUSAL_PHRASES:
        if phrase in text_lower:
            return True
    return False


def load_eval_data(eval_path):
    """Load eval_all.jsonl and index by lang/type."""
    eval_data = defaultdict(list)
    with open(eval_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            key = (row["lang"], row["type"])
            eval_data[key].append(row)
    return eval_data


def load_gen_files(gen_patterns):
    """Load all gen files matching patterns."""
    gen_data = []
    for pattern in gen_patterns:
        for filepath in glob.glob(pattern):
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line)
                    gen_data.append(row)
    return gen_data


def main():
    parser = argparse.ArgumentParser(description="Score predictions")
    parser.add_argument(
        "--eval",
        default="data/eval_all.jsonl",
        help="Path to eval_all.jsonl",
    )
    parser.add_argument(
        "--gen",
        nargs="+",
        required=True,
        help="Glob patterns for gen files (e.g., gen_model_*.jsonl)",
    )
    parser.add_argument(
        "--tag",
        required=True,
        help="Tag for output (e.g., model name)",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Output directory for results",
    )
    args = parser.parse_args()

    # Load eval data
    eval_data = load_eval_data(args.eval)

    # Load gen data
    gen_data = load_gen_files(args.gen)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Score each prediction
    results = defaultdict(lambda: {"n": 0, "refusal_correct": 0, "refusal_total": 0, "f1_sum": 0.0, "f1_count": 0})
    overall = {"n": 0, "refusal_correct": 0, "refusal_total": 0, "f1_sum": 0.0, "f1_count": 0}

    for gen_row in gen_data:
        lang = gen_row["lang"]
        qtype = gen_row["type"]
        key = (lang, qtype)

        pred = gen_row.get("pred")
        ref = gen_row.get("ref")

        # Update counts
        results[key]["n"] += 1
        overall["n"] += 1

        if qtype == "unanswerable":
            # Refusal scoring
            results[key]["refusal_total"] += 1
            overall["refusal_total"] += 1

            if pred is None or pred == "":
                # Missing/empty pred scores 0
                pass
            elif is_refusal(pred):
                results[key]["refusal_correct"] += 1
                overall["refusal_correct"] += 1
        else:
            # F1 scoring for grounded questions
            if pred is None or pred == "":
                # Missing/empty pred scores 0
                results[key]["f1_count"] += 1
                overall["f1_count"] += 1
            else:
                pred_tokens = tokenize(pred)
                ref_tokens = tokenize(ref)
                f1 = compute_f1(pred_tokens, ref_tokens)
                results[key]["f1_sum"] += f1
                results[key]["f1_count"] += 1
                overall["f1_sum"] += f1
                overall["f1_count"] += 1

    # Compute averages
    output = {}

    # Per lang x type
    for (lang, qtype), stats in sorted(results.items()):
        refusal_acc = stats["refusal_correct"] / stats["refusal_total"] if stats["refusal_total"] > 0 else None
        mean_f1 = stats["f1_sum"] / stats["f1_count"] if stats["f1_count"] > 0 else None

        output[f"{lang}_{qtype}"] = {
            "n": stats["n"],
            "refusal_accuracy": refusal_acc,
            "mean_f1": mean_f1,
        }

    # Overall
    overall_refusal_acc = overall["refusal_correct"] / overall["refusal_total"] if overall["refusal_total"] > 0 else None
    overall_mean_f1 = overall["f1_sum"] / overall["f1_count"] if overall["f1_count"] > 0 else None

    output["overall"] = {
        "n": overall["n"],
        "refusal_accuracy": overall_refusal_acc,
        "mean_f1": overall_mean_f1,
    }

    # Print table
    print("\nResults by lang x type:")
    print(f"{'Lang':<10} {'Type':<15} {'N':<6} {'Refusal Acc':<12} {'Mean F1':<12}")
    print("-" * 60)
    for (lang, qtype), stats in sorted(results.items()):
        refusal_acc = stats["refusal_correct"] / stats["refusal_total"] if stats["refusal_total"] > 0 else "N/A"
        mean_f1 = stats["f1_sum"] / stats["f1_count"] if stats["f1_count"] > 0 else "N/A"
        if refusal_acc != "N/A":
            refusal_acc = f"{refusal_acc:.3f}"
        if mean_f1 != "N/A":
            mean_f1 = f"{mean_f1:.3f}"
        print(f"{lang:<10} {qtype:<15} {stats['n']:<6} {refusal_acc:<12} {mean_f1:<12}")

    print("-" * 60)
    overall_refusal_acc = overall["refusal_correct"] / overall["refusal_total"] if overall["refusal_total"] > 0 else "N/A"
    overall_mean_f1 = overall["f1_sum"] / overall["f1_count"] if overall["f1_count"] > 0 else "N/A"
    if overall_refusal_acc != "N/A":
        overall_refusal_acc = f"{overall_refusal_acc:.3f}"
    if overall_mean_f1 != "N/A":
        overall_mean_f1 = f"{overall_mean_f1:.3f}"
    print(f"{'overall':<10} {'':<15} {overall['n']:<6} {overall_refusal_acc:<12} {overall_mean_f1:<12}")
    print()

    # Write JSON output
    output_path = os.path.join(args.output_dir, f"score_{args.tag}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"Results written to {output_path}")


if __name__ == "__main__":
    main()
