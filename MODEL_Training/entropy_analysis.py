"""
entropy_analysis.py
-------------------
Computes Shannon entropy of the classifier's output probability distribution
for each prompt, then tests whether entropy differs significantly across
hallucination-risk classes (Low / Medium / High).

Mathematical foundation
-----------------------
For a K-class classifier, the predictive entropy of a single sample is:

    H(p) = -∑ pᵢ · log₂(pᵢ)    for i = 1 … K

A perfectly confident prediction (p = [1, 0, 0]) has H = 0.
A maximally uncertain prediction (p = [⅓, ⅓, ⅓]) has H = log₂(3) ≈ 1.585.

High-risk prompts are expected to push the classifier towards higher entropy
(less certainty), which is a meaningful probabilistic claim we can verify.

Usage
-----
    python entropy_analysis.py \
        --logits_path  results/logits.npy \
        --labels_path  results/true_labels.npy \
        --output_dir   results/entropy
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import softmax
from scipy.stats import kruskal

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Label mapping
# ---------------------------------------------------------------------------
LABEL_NAMES: dict[int, str] = {0: "Low", 1: "Medium", 2: "High"}
NUM_CLASSES: int = len(LABEL_NAMES)


# ---------------------------------------------------------------------------
# Core maths
# ---------------------------------------------------------------------------

def compute_probabilities(logits: np.ndarray) -> np.ndarray:
    """Convert raw logits to a probability distribution via softmax.

    Parameters
    ----------
    logits : np.ndarray, shape (N, K)
        Raw model output before normalisation.

    Returns
    -------
    np.ndarray, shape (N, K)
        Each row is a valid probability distribution (sums to 1).
    """
    return softmax(logits, axis=1)


def shannon_entropy(probs: np.ndarray) -> np.ndarray:
    """Compute per-sample Shannon entropy in bits (base-2 logarithm).

    H(p) = -∑ pᵢ · log₂(pᵢ)

    Zero-probability classes contribute 0 to the sum (0 · log 0 := 0),
    handled by clipping before the log.

    Parameters
    ----------
    probs : np.ndarray, shape (N, K)
        Probability distributions; each row must sum to 1.

    Returns
    -------
    np.ndarray, shape (N,)
        Entropy value in bits for each sample.
    """
    # Clip to avoid log(0); epsilon is negligible relative to valid probs
    safe_probs = np.clip(probs, a_min=1e-12, a_max=None)
    return -np.sum(probs * np.log2(safe_probs), axis=1)


def max_possible_entropy(n_classes: int) -> float:
    """Return the theoretical maximum entropy for a uniform distribution.

    H_max = log₂(K)

    Parameters
    ----------
    n_classes : int
        Number of output classes.

    Returns
    -------
    float
        Maximum entropy in bits.
    """
    return np.log2(n_classes)


# ---------------------------------------------------------------------------
# Statistical test
# ---------------------------------------------------------------------------

def kruskal_wallis_test(
    entropy_by_class: dict[int, np.ndarray],
) -> tuple[float, float]:
    """Run a Kruskal–Wallis H-test across entropy groups.

    The Kruskal–Wallis test is a non-parametric alternative to one-way ANOVA.
    It tests whether entropy distributions are identical across risk classes
    without assuming Gaussian residuals.

    H₀: The entropy distributions of Low, Medium, and High risk classes
        are identical (any observed differences are due to chance).

    Parameters
    ----------
    entropy_by_class : dict[int, np.ndarray]
        Mapping from class index → entropy values for that class.

    Returns
    -------
    tuple[float, float]
        (H statistic, p-value).  p < 0.05 rejects H₀.
    """
    groups = [entropy_by_class[k] for k in sorted(entropy_by_class)]
    h_stat, p_value = kruskal(*groups)
    return float(h_stat), float(p_value)


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_entropy_by_class(
    entropy_by_class: dict[int, np.ndarray],
    h_stat: float,
    p_value: float,
    max_entropy: float,
    output_path: Path,
) -> None:
    """Save a box-and-strip plot of entropy distributions per risk class.

    Parameters
    ----------
    entropy_by_class : dict[int, np.ndarray]
        Entropy arrays keyed by class index.
    h_stat : float
        Kruskal–Wallis H statistic (shown in subtitle).
    p_value : float
        Corresponding p-value (shown in subtitle).
    max_entropy : float
        Theoretical maximum (log₂ K); drawn as a dashed reference line.
    output_path : Path
        Destination .png file.
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    labels = [LABEL_NAMES[k] for k in sorted(entropy_by_class)]
    data = [entropy_by_class[k] for k in sorted(entropy_by_class)]
    colors = ["#4caf50", "#ff9800", "#f44336"]  # green / amber / red

    bp = ax.boxplot(data, patch_artist=True, widths=0.45, medianprops={"color": "white", "linewidth": 2})
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Overlay individual points (jittered for readability)
    rng = np.random.default_rng(seed=42)
    for i, (values, color) in enumerate(zip(data, colors), start=1):
        jitter = rng.uniform(-0.15, 0.15, size=len(values))
        ax.scatter(i + jitter, values, color=color, alpha=0.35, s=12, zorder=3)

    # Reference line: maximum possible entropy
    ax.axhline(max_entropy, linestyle="--", color="grey", linewidth=1.2, label=f"H_max = {max_entropy:.3f} bits")

    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("Shannon Entropy  H(p)  [bits]", fontsize=11)
    ax.set_title(
        "Classifier Predictive Entropy by Hallucination Risk Class\n"
        f"Kruskal–Wallis  H = {h_stat:.2f},  p = {p_value:.4f}",
        fontsize=11,
    )
    ax.legend(fontsize=9)
    ax.set_ylim(bottom=-0.05)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Entropy plot saved → %s", output_path)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_summary(
    entropy_by_class: dict[int, np.ndarray],
    h_stat: float,
    p_value: float,
    max_entropy: float,
) -> None:
    """Print a concise summary table to stdout."""
    print("\n" + "=" * 55)
    print("  Predictive Entropy Summary")
    print("=" * 55)
    print(f"  Maximum possible entropy  : {max_entropy:.4f} bits  (log₂ {NUM_CLASSES})")
    print("-" * 55)
    print(f"  {'Class':<10}  {'N':>6}  {'Mean H':>9}  {'Std H':>9}  {'Median H':>9}")
    print("-" * 55)
    for k in sorted(entropy_by_class):
        vals = entropy_by_class[k]
        print(
            f"  {LABEL_NAMES[k]:<10}  {len(vals):>6}  "
            f"{vals.mean():>9.4f}  {vals.std():>9.4f}  {np.median(vals):>9.4f}"
        )
    print("-" * 55)
    significance = "SIGNIFICANT  ✓" if p_value < 0.05 else "not significant"
    print(f"  Kruskal–Wallis  H = {h_stat:.4f},  p = {p_value:.4f}  →  {significance}")
    print("=" * 55 + "\n")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(logits_path: Path, labels_path: Path, output_dir: Path) -> None:
    """Execute the full entropy analysis pipeline.

    Parameters
    ----------
    logits_path : Path
        Path to a .npy file of shape (N, K) containing raw model logits.
    labels_path : Path
        Path to a .npy file of shape (N,) containing integer true labels.
    output_dir : Path
        Directory where the plot and per-sample entropy CSV are written.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    logits: np.ndarray = np.load(logits_path)   # shape (N, K)
    labels: np.ndarray = np.load(labels_path)   # shape (N,)
    logger.info("Loaded %d samples with %d classes.", logits.shape[0], logits.shape[1])

    # 2. Compute probabilities and entropy
    probs = compute_probabilities(logits)                   # softmax → shape (N, K)
    entropy: np.ndarray = shannon_entropy(probs)            # shape (N,)
    h_max = max_possible_entropy(NUM_CLASSES)

    # 3. Group entropy by true class
    entropy_by_class: dict[int, np.ndarray] = {
        k: entropy[labels == k] for k in range(NUM_CLASSES)
    }

    # 4. Statistical test
    h_stat, p_value = kruskal_wallis_test(entropy_by_class)

    # 5. Report
    print_summary(entropy_by_class, h_stat, p_value, h_max)

    # 6. Save plot
    plot_path = output_dir / "entropy_by_class.png"
    plot_entropy_by_class(entropy_by_class, h_stat, p_value, h_max, plot_path)

    # 7. Save per-sample CSV for downstream use
    csv_path = output_dir / "per_sample_entropy.csv"
    header = "true_label,predicted_class,confidence,entropy"
    predicted = np.argmax(probs, axis=1)
    confidence = probs[np.arange(len(probs)), predicted]
    rows = np.column_stack([labels, predicted, confidence, entropy])
    np.savetxt(csv_path, rows, delimiter=",", header=header, comments="", fmt="%.6f")
    logger.info("Per-sample entropy saved → %s", csv_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Entropy analysis of hallucination-risk classifier outputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--logits_path", type=Path, required=True, help="Path to logits .npy file (N × K).")
    parser.add_argument("--labels_path", type=Path, required=True, help="Path to true labels .npy file (N,).")
    parser.add_argument("--output_dir", type=Path, default=Path("results/entropy"), help="Output directory.")
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()
    run(
        logits_path=args.logits_path,
        labels_path=args.labels_path,
        output_dir=args.output_dir,
    )
