"""
entropy_analysis.py
Computes Shannon entropy of the classifier's softmax outputs and checks whether
High-risk prompts are genuinely harder to classify (higher uncertainty) than
Low-risk ones, verified with a Kruskal-Wallis test.

Usage:
    python entropy_analysis.py \
        --logits_path results/logits.npy \
        --labels_path results/true_labels.npy \
        --output_dir  results/entropy
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import softmax
from scipy.stats import kruskal

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

LABEL_NAMES: dict[int, str] = {0: "Low", 1: "Medium", 2: "High"}
NUM_CLASSES: int = len(LABEL_NAMES)


def compute_probabilities(logits: np.ndarray) -> np.ndarray:
    """Softmax over the class axis → valid probability distribution per sample."""
    return softmax(logits, axis=1)


def shannon_entropy(probs: np.ndarray) -> np.ndarray:
    """
    H(p) = -∑ pᵢ · log₂(pᵢ)

    Clipping before log handles the 0 · log(0) = 0 edge case cleanly.
    Returns entropy in bits for each sample.
    """
    safe = np.clip(probs, 1e-12, None)
    return -np.sum(probs * np.log2(safe), axis=1)


def kruskal_wallis_test(entropy_by_class: dict[int, np.ndarray]) -> tuple[float, float]:
    """
    Non-parametric test for whether entropy distributions differ across risk classes.
    H0: all three distributions are identical.
    Returns (H statistic, p-value); p < 0.05 rejects H0.
    """
    groups = [entropy_by_class[k] for k in sorted(entropy_by_class)]
    h_stat, p_value = kruskal(*groups)
    return float(h_stat), float(p_value)


def plot_entropy_by_class(
    entropy_by_class: dict[int, np.ndarray],
    h_stat: float,
    p_value: float,
    output_path: Path,
) -> None:
    labels = [LABEL_NAMES[k] for k in sorted(entropy_by_class)]
    data = [entropy_by_class[k] for k in sorted(entropy_by_class)]
    colors = ["#4caf50", "#ff9800", "#f44336"]

    fig, ax = plt.subplots(figsize=(7, 5))

    bp = ax.boxplot(data, patch_artist=True, widths=0.45,
                    medianprops={"color": "white", "linewidth": 2})
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # jittered scatter so individual points are visible
    rng = np.random.default_rng(seed=42)
    for i, (values, color) in enumerate(zip(data, colors), start=1):
        jitter = rng.uniform(-0.15, 0.15, size=len(values))
        ax.scatter(i + jitter, values, color=color, alpha=0.35, s=12, zorder=3)

    h_max = np.log2(NUM_CLASSES)
    ax.axhline(h_max, linestyle="--", color="grey", linewidth=1.2,
               label=f"H_max = {h_max:.3f} bits (uniform dist.)")

    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("Shannon Entropy  H(p)  [bits]", fontsize=11)
    ax.set_title(
        "Predictive Entropy by Hallucination Risk Class\n"
        f"Kruskal-Wallis  H={h_stat:.2f},  p={p_value:.4f}",
        fontsize=11,
    )
    ax.legend(fontsize=9)
    ax.set_ylim(bottom=-0.05)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Plot saved -> %s", output_path)


def print_summary(
    entropy_by_class: dict[int, np.ndarray],
    h_stat: float,
    p_value: float,
) -> None:
    h_max = np.log2(NUM_CLASSES)
    print(f"\n{'='*55}")
    print(f"  Predictive Entropy Summary   (H_max = {h_max:.4f} bits)")
    print(f"{'='*55}")
    print(f"  {'Class':<10}  {'N':>6}  {'Mean':>8}  {'Std':>8}  {'Median':>8}")
    print(f"  {'-'*50}")
    for k in sorted(entropy_by_class):
        v = entropy_by_class[k]
        print(f"  {LABEL_NAMES[k]:<10}  {len(v):>6}  {v.mean():>8.4f}  {v.std():>8.4f}  {np.median(v):>8.4f}")
    sig = "SIGNIFICANT" if p_value < 0.05 else "not significant"
    print(f"\n  Kruskal-Wallis  H={h_stat:.4f},  p={p_value:.4f}  ->  {sig}")
    print(f"{'='*55}\n")


def run(logits_path: Path, labels_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    logits: np.ndarray = np.load(logits_path)
    labels: np.ndarray = np.load(labels_path)
    logger.info("Loaded %d samples, %d classes", logits.shape[0], logits.shape[1])

    probs = compute_probabilities(logits)
    entropy = shannon_entropy(probs)

    entropy_by_class = {k: entropy[labels == k] for k in range(NUM_CLASSES)}
    h_stat, p_value = kruskal_wallis_test(entropy_by_class)

    print_summary(entropy_by_class, h_stat, p_value)
    plot_entropy_by_class(entropy_by_class, h_stat, p_value,
                          output_dir / "entropy_by_class.png")

    # save per-sample results for downstream use
    predicted = np.argmax(probs, axis=1)
    confidence = probs[np.arange(len(probs)), predicted]
    rows = np.column_stack([labels, predicted, confidence, entropy])
    np.savetxt(output_dir / "per_sample_entropy.csv", rows,
               delimiter=",", header="true_label,predicted_class,confidence,entropy",
               comments="", fmt="%.6f")
    logger.info("Per-sample CSV saved -> %s", output_dir / "per_sample_entropy.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entropy analysis of classifier outputs.")
    parser.add_argument("--logits_path", type=Path, required=True)
    parser.add_argument("--labels_path", type=Path, required=True)
    parser.add_argument("--output_dir",  type=Path, default=Path("results/entropy"))
    args = parser.parse_args()
    run(args.logits_path, args.labels_path, args.output_dir)
