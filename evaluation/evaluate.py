# Step 6: evaluation/evaluate.py
#
# Metrics:
#   Precision@k — of top-k results, fraction that are relevant
#   Recall@k    — whether the relevant chunk appears in top-k (0 or 1)
#   MRR         — Mean Reciprocal Rank (rewards finding correct answer higher up)
#
# Run: python evaluation/evaluate.py

import json
import os


# ─────────────────────────────────────────────
# Metric Functions
# ─────────────────────────────────────────────

def precision_at_k(retrieved_ids, relevant_id, k):
    """Fraction of top-k results that are relevant. Max is 1/k for single-relevant."""
    return sum(1 for r in retrieved_ids[:k] if r == relevant_id) / k


def recall_at_k(retrieved_ids, relevant_id, k):
    """1.0 if relevant chunk is anywhere in top-k, else 0.0."""
    return 1.0 if relevant_id in retrieved_ids[:k] else 0.0


def reciprocal_rank(retrieved_ids, relevant_id):
    """
    1/rank of first correct result.
    rank 1 → 1.0, rank 2 → 0.5, rank 3 → 0.33, not found → 0.0
    """
    for i, cid in enumerate(retrieved_ids):
        if cid == relevant_id:
            return 1.0 / (i + 1)
    return 0.0


# ─────────────────────────────────────────────
# Evaluate One System
# ─────────────────────────────────────────────

def evaluate(results, k_values=[1, 3, 5]):
    """
    Compute all metrics averaged across all queries.
    Returns dict like: {"P@1": 0.42, "R@5": 0.78, "MRR": 0.55, ...}
    """
    accumulators = {f"P@{k}": [] for k in k_values}
    accumulators.update({f"R@{k}": [] for k in k_values})
    accumulators["MRR"] = []

    for r in results:
        retrieved = r["retrieved_chunk_ids"]
        relevant = r["relevant_chunk_id"]

        for k in k_values:
            accumulators[f"P@{k}"].append(precision_at_k(retrieved, relevant, k))
            accumulators[f"R@{k}"].append(recall_at_k(retrieved, relevant, k))

        accumulators["MRR"].append(reciprocal_rank(retrieved, relevant))

    return {
        metric: round(sum(vals) / len(vals), 4)
        for metric, vals in accumulators.items()
    }


# ─────────────────────────────────────────────
# Compare Both Systems
# ─────────────────────────────────────────────

def compare_and_print(baseline_results, improved_results, k_values=[1, 3, 5]):
    b_metrics = evaluate(baseline_results, k_values)
    i_metrics = evaluate(improved_results, k_values)

    metric_order = (
        [f"P@{k}" for k in k_values] +
        [f"R@{k}" for k in k_values] +
        ["MRR"]
    )

    print("\n" + "=" * 58)
    print("  EVALUATION RESULTS")
    print("=" * 58)
    print(f"  {'Metric':<10} {'Baseline':>10} {'Improved':>10} {'Change':>10}")
    print("  " + "-" * 44)
    for metric in metric_order:
        b = b_metrics[metric]
        i = i_metrics[metric]
        delta = round(i - b, 4)
        sign = "↑" if delta > 0 else ("↓" if delta < 0 else "=")
        print(f"  {metric:<10} {b:>10.4f} {i:>10.4f}   {sign} {abs(delta):.4f}")
    print("=" * 58)

    return b_metrics, i_metrics


# ─────────────────────────────────────────────
# Per-Query Breakdown (case studies for report)
# ─────────────────────────────────────────────

def per_query_breakdown(baseline_results, improved_results, k=5):
    """
    Shows per-query recall so you can pick interesting examples for your report.
    Look for: queries where improved >> baseline (strong wins)
              queries where baseline > improved (failure cases)
    """
    print(f"\n--- Per-Query Recall@{k} ---")
    print(f"  {'ID':<8} {'Query':<40} {'Base':>6} {'Impr':>6} {'Status'}")
    print("  " + "-" * 72)

    wins, losses, ties = [], [], []

    for b, i in zip(baseline_results, improved_results):
        assert b["query_id"] == i["query_id"], "Query order mismatch!"
        b_r = recall_at_k(b["retrieved_chunk_ids"], b["relevant_chunk_id"], k)
        i_r = recall_at_k(i["retrieved_chunk_ids"], i["relevant_chunk_id"], k)
        delta = i_r - b_r
        status = "WIN ↑" if delta > 0 else ("LOSS ↓" if delta < 0 else "TIE")
        short_q = b["query"][:38]
        print(f"  {b['query_id']:<8} {short_q:<40} {b_r:>6.1f} {i_r:>6.1f}   {status}")

        if delta > 0:
            wins.append(b["query_id"])
        elif delta < 0:
            losses.append(b["query_id"])
        else:
            ties.append(b["query_id"])

    print(f"\n  Wins : {len(wins)}  |  Losses: {len(losses)}  |  Ties: {len(ties)}")
    if wins:
        print(f"  Strong wins (use for report): {wins[:5]}")
    if losses:
        print(f"  Failure cases (error analysis): {losses[:5]}")


# ─────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────

def save_metrics(b_metrics, i_metrics, path="evaluation/metrics.json"):
    os.makedirs("evaluation", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"baseline": b_metrics, "improved": i_metrics}, f, indent=2)
    print(f"\nMetrics saved to {path}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

if __name__ == "__main__":
    for path in ["retrieval/baseline_results.json", "retrieval/improved_results.json"]:
        if not os.path.exists(path):
            print(f"{path} not found. Run retrieval/retrieve.py first.")
            exit(1)

    with open("retrieval/baseline_results.json", "r", encoding="utf-8") as f:
        baseline_results = json.load(f)

    with open("retrieval/improved_results.json", "r", encoding="utf-8") as f:
        improved_results = json.load(f)

    print(f"Evaluating {len(baseline_results)} queries...")

    b_metrics, i_metrics = compare_and_print(
        baseline_results, improved_results, k_values=[1, 3, 5]
    )

    per_query_breakdown(baseline_results, improved_results, k=5)

    save_metrics(b_metrics, i_metrics)