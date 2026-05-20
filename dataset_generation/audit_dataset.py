#!/usr/bin/env python3
"""
Audit script: prints dataset stats and flags rows whose label
disagrees with the freeze taxonomy heuristics.

Usage:
    python audit_dataset.py <input_csv>
"""

import csv
import sys
from collections import Counter

# --- Heuristics (same as final taxonomy) ---
REALTIME_MARKERS = ["current", "today", "right now", "latest", "this year", "as of", "yesterday", "now"]
AMBIGUITY_MARKERS = ["missing constraints", "how many", "when did", "which one", "depends", "could", "might"]
FINANCE_MARKERS = ["invest", "interest rate", "return", "compound", "mortgage", "loan", "savings"]
MEDICAL_MARKERS = ["headache", "chest pain", "nausea", "fever", "medical", "doctor", "symptom", "should i"]
LEGAL_MARKERS = ["legal", "lawsuit", "jury", "law", "contract"]
FALSE_PREMISE_MARKERS = ["taller than mount everest", "nine lives", "if einstein's theories were false", "cure for common cold"]
DETERMINISTIC_MATH = ["solve", "calculate", "integral", "equation", "roots"]
CREATIVE_MARKERS = ["write a story", "imagine", "short story", "alien", "fiction", "fantasy", "roleplay"]

def detect_expected_label(prompt: str):
    p = prompt.lower()
    if any(k in p for k in CREATIVE_MARKERS):
        return None  # creative (should be excluded or tagged)
    if any(k in p for k in MEDICAL_MARKERS):
        return ("Yes", "High")
    if any(k in p for k in LEGAL_MARKERS):
        return ("Yes", "High")
    if any(k in p for k in FALSE_PREMISE_MARKERS):
        return ("Yes", "High")
    if any(k in p for k in REALTIME_MARKERS):
        return ("Yes", "Medium")
    if any(k in p for k in AMBIGUITY_MARKERS):
        return ("Yes", "Medium")
    if any(k in p for k in FINANCE_MARKERS):
        return ("Yes", "Medium")
    if any(k in p for k in DETERMINISTIC_MATH) and "%" not in p:
        return ("No", "Low")
    # default: factual / low-risk
    return ("No", "Low")

def normalize(s):
    return s.strip().lower()

def main(input_csv):
    rows = []
    with open(input_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, r in enumerate(reader, start=1):
            rows.append((i, r))

    counts = Counter()
    flagged = []

    for idx, r in rows:
        prompt = r.get("prompt", r.get("Prompt", "")).strip()
        risk = normalize(r.get("hallucination_risk", r.get("HallucinationRisk", "Yes")))
        level = normalize(r.get("risk_level", r.get("RiskLevel", "Medium")))
        conf = r.get("confidence", r.get("Confidence", "80")).strip()

        expected = detect_expected_label(prompt)
        counts[(risk, level)] += 1

        if expected is None:
            # creative: flag if dataset kept it
            flagged.append((idx, prompt, "creative_kept", risk, level, conf))
        else:
            exp_risk, exp_level = expected
            if normalize(exp_risk) != risk or normalize(exp_level) != level:
                flagged.append((idx, prompt, f"mismatch_expected_{exp_risk}_{exp_level}", risk, level, conf))

    # summary
    print("=== Dataset summary ===")
    for k, v in counts.items():
        print(f"{k}: {v}")
    print(f"Total rows: {len(rows)}")
    print()
    print("=== Flagged rows (sample) ===")
    for item in flagged[:200]:
        idx, prompt, issue, risk, level, conf = item
        print(f"[{idx}] {issue} | {risk}/{level}/{conf} | {prompt[:200]}")

    print()
    print(f"Total flagged: {len(flagged)}")
    if len(flagged) > 0:
        print("You should inspect flagged rows and either accept, correct, or drop them.")
    else:
        print("No obvious label mismatches found.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python audit_dataset.py <input_csv>")
        sys.exit(1)
    main(sys.argv[1])
