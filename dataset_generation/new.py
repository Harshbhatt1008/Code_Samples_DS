import csv
import os
import requests
from typing import Tuple

# ============================
# Configuration
# ============================

BATCH_SIZE = 5
NUM_BATCHES = 100
OUTPUT_CSV = "hallucination_dataset_final.csv"

GENERATOR_MODEL = "mistral"
CLASSIFIER_MODEL = "hallucination-risk-classifier"

OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"

GEN_MAX_ATTEMPTS = 6


# ============================
# Ollama helper
# ============================

def ollama_generate(model: str, prompt: str) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }
    try:
        r = requests.post(OLLAMA_CHAT_URL, json=payload, timeout=30)
        r.raise_for_status()
        return r.json().get("message", {}).get("content", "").strip()
    except Exception:
        return ""


# ============================
# Marker lists (FINAL)
# ============================

REALTIME_MARKERS = [
    "current", "today", "right now", "latest", "this year",
    "as of", "yesterday", "today's", "now"
]

AMBIGUITY_MARKERS = [
    "missing constraints", "how many", "when did", "which one",
    "depends", "could", "might"
]

FINANCE_MARKERS = [
    "invest", "interest rate", "return", "compound",
    "mortgage", "loan", "savings"
]

MEDICAL_MARKERS = [
    "headache", "chest pain", "nausea", "fever",
    "medical", "doctor", "symptom", "should i"
]

LEGAL_MARKERS = [
    "legal", "lawsuit", "jury", "law", "contract"
]

FALSE_PREMISE_MARKERS = [
    "taller than mount everest",
    "nine lives",
    "if einstein's theories were false",
    "cure for common cold"
]

DETERMINISTIC_MATH = [
    "solve", "calculate", "integral", "equation", "roots"
]

CREATIVE_MARKERS = [
    "write a story", "imagine", "short story",
    "alien", "fiction", "fantasy", "roleplay"
]


# ============================
# Prompt generation
# ============================

def generate_prompt_batch(batch_size: int, seen: set):
    batch = []
    prompt = f"""
Generate EXACTLY {batch_size} user prompts.

Rules:
- One prompt per line
- No numbering or explanations
- Avoid pure fiction, roleplay, or storytelling
- Focus on factual, ambiguous, real-time, advice, or reasoning prompts
""".strip()

    for _ in range(GEN_MAX_ATTEMPTS):
        text = ollama_generate(GENERATOR_MODEL, prompt)
        if not text:
            continue

        for line in text.splitlines():
            p = line.strip().strip('"').strip("'")
            if not p or p in seen:
                continue
            if any(k in p.lower() for k in CREATIVE_MARKERS):
                continue
            batch.append(p)
            seen.add(p)
            if len(batch) >= batch_size:
                return batch

    return batch


# ============================
# Classification
# ============================

def classify_prompt(prompt: str) -> Tuple[str, str, str]:
    text = ollama_generate(CLASSIFIER_MODEL, prompt)
    risk, level, conf = "Yes", "Medium", "75"

    for line in text.splitlines():
        l = line.lower()
        if l.startswith("hallucinationrisk:"):
            risk = line.split(":", 1)[1].strip()
        elif l.startswith("risklevel:"):
            level = line.split(":", 1)[1].strip()
        elif l.startswith("confidence:"):
            conf = line.split(":", 1)[1].replace("%", "").strip()

    return risk, level, conf


# ============================
# FINAL POSTPROCESS (FREEZE)
# ============================

def postprocess(prompt: str, risk: str, level: str, confidence: str):
    p = prompt.lower()

    # 🚫 Creative → drop
    if any(k in p for k in CREATIVE_MARKERS):
        return None

    # 🧮 Deterministic math → Low
    if any(k in p for k in DETERMINISTIC_MATH) and "%" not in p:
        return "No", "Low", "95"

    # 🕒 Real-time → Medium
    if any(k in p for k in REALTIME_MARKERS):
        return "Yes", "Medium", "80"

    # ❓ Ambiguous → Medium
    if any(k in p for k in AMBIGUITY_MARKERS):
        return "Yes", "Medium", "75"

    # 💰 Finance → Medium
    if any(k in p for k in FINANCE_MARKERS):
        return "Yes", "Medium", "80"

    # 🩺 Medical → High
    if any(k in p for k in MEDICAL_MARKERS):
        return "Yes", "High", "80"

    # ⚖ Legal → High
    if any(k in p for k in LEGAL_MARKERS):
        return "Yes", "High", "80"

    # ❌ False premise → High
    if any(k in p for k in FALSE_PREMISE_MARKERS):
        return "Yes", "High", "85"

    # 📚 Default factual → Low
    return "No", "Low", min(confidence, "95")


# ============================
# Main
# ============================

def main():
    seen = set()

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["prompt", "hallucination_risk", "risk_level", "confidence"])

        for i in range(NUM_BATCHES):
            print(f"Batch {i+1}/{NUM_BATCHES}")
            prompts = generate_prompt_batch(BATCH_SIZE, seen)

            for p in prompts:
                risk, level, conf = classify_prompt(p)
                result = postprocess(p, risk, level, conf)
                if result:
                    writer.writerow([p, *result])

    print("✅ Dataset v1.0 generated and frozen.")


if __name__ == "__main__":
    main()
