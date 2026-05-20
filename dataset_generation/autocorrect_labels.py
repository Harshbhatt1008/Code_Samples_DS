#!/usr/bin/env python3
"""
Autocorrect labels according to the final freeze taxonomy.
Writes <input>_corrected.csv
"""

import csv, sys, os

REALTIME_MARKERS = ["current", "today", "right now", "latest", "this year", "as of", "yesterday", "now"]
AMBIGUITY_MARKERS = ["missing constraints", "how many", "when did", "which one", "depends", "could", "might"]
FINANCE_MARKERS = ["invest", "interest rate", "return", "compound", "mortgage", "loan", "savings"]
MEDICAL_MARKERS = ["headache", "chest pain", "nausea", "fever", "medical", "doctor", "symptom", "should i"]
LEGAL_MARKERS = ["legal", "lawsuit", "jury", "law", "contract"]
FALSE_PREMISE_MARKERS = ["taller than mount everest", "nine lives", "if einstein's theories were false", "cure for common cold"]
DETERMINISTIC_MATH = ["solve", "calculate", "integral", "equation", "roots"]
CREATIVE_MARKERS = ["write a story", "imagine", "short story", "alien", "fiction", "fantasy", "roleplay"]

def is_creative(p):
    p = p.lower()
    return any(k in p for k in CREATIVE_MARKERS)

def postprocess(p):
    lp = p.lower()
    if is_creative(lp):
        return None  # drop creative prompts
    if any(k in lp for k in MEDICAL_MARKERS):
        return ("Yes", "High", "80")
    if any(k in lp for k in LEGAL_MARKERS):
        return ("Yes", "High", "80")
    if any(k in lp for k in FALSE_PREMISE_MARKERS):
        return ("Yes", "High", "85")
    if any(k in lp for k in REALTIME_MARKERS):
        return ("Yes", "Medium", "80")
    if any(k in lp for k in AMBIGUITY_MARKERS):
        return ("Yes", "Medium", "75")
    if any(k in lp for k in FINANCE_MARKERS):
        return ("Yes", "Medium", "80")
    if any(k in lp for k in DETERMINISTIC_MATH) and "%" not in lp:
        return ("No", "Low", "95")
    # default
    return ("No", "Low", "95")

def main(input_csv):
    out_csv = os.path.splitext(input_csv)[0] + "_corrected.csv"
    with open(input_csv, newline="", encoding="utf-8") as inf, open(out_csv, "w", newline="", encoding="utf-8") as outf:
        reader = csv.DictReader(inf)
        fieldnames = ["prompt", "hallucination_risk", "risk_level", "confidence"]
        writer = csv.DictWriter(outf, fieldnames=fieldnames)
        writer.writeheader()
        kept = 0
        dropped = 0
        for row in reader:
            p = row.get("prompt", "").strip()
            res = postprocess(p)
            if res is None:
                dropped += 1
                continue
            risk, level, conf = res
            writer.writerow({"prompt": p, "hallucination_risk": risk, "risk_level": level, "confidence": conf})
            kept += 1

    print(f"Wrote {out_csv}: kept={kept}, dropped={dropped}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python autocorrect_labels.py <input_csv>")
        sys.exit(1)
    main(sys.argv[1])
