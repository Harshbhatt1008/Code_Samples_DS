# pip install google-genai

import csv
import os
from google import genai
from google.genai import types

# ============================
# Configuration
# ============================

BATCH_SIZE = 5          # prompts per batch
NUM_BATCHES = 10        # set how many batches you want
OUTPUT_CSV = "hallucination_dataset.csv"
MODEL_NAME = "gemini-flash-latest"

# ============================
# System instructions
# ============================

CLASSIFIER_SYSTEM = """
You are a Hallucination-Risk Classifier for Large Language Model prompts.

Rules:
- Internally analyze hallucination risk.
- Do NOT reveal analysis or explanation.
- Output must be ONLY:

HallucinationRisk: Yes|No
RiskLevel: Low|Medium|High
Confidence: <number>
"""

GENERATOR_SYSTEM = """
Generate a list of user prompts that could be given to a large language model.
Prompts must span both:
- Grounded factual topics
- Difficult / risky hallucination scenarios

Constraints:
- Respond ONLY with the prompts
- One prompt per line
- No numbering, no bullets, no explanation
- Prompts should be diverse in domain: science, history, predictions, fictional info, trivia, medical, legal, etc.
"""

# ============================
# Prompt generation
# ============================

def generate_prompt_batch(client, model: str, batch_size: int, seen_prompts: set, max_attempts: int = 5):
    """
    Generate a batch of up to `batch_size` unique prompts.
    Ensures prompts are not repeated across all batches by checking `seen_prompts`.
    """
    batch = []

    for _ in range(max_attempts):
        if len(batch) >= batch_size:
            break

        response = client.models.generate_content(
            model=model,
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part(
                            text=(
                                f"{GENERATOR_SYSTEM}\n"
                                f"Generate exactly {batch_size} prompts."
                            )
                        )
                    ],
                )
            ],
        )

        text = (response.text or "").strip()
        if not text:
            continue

        lines = [line.strip() for line in text.splitlines() if line.strip()]

        for p in lines:
            if p not in seen_prompts and len(batch) < batch_size:
                batch.append(p)
                seen_prompts.add(p)

        # If still not enough, loop again (up to max_attempts)

    return batch

# ============================
# Classification
# ============================

def classify_prompt(client, model: str, user_prompt: str):
    """
    Classify a single prompt and return (risk, level, confidence).
    """
    response = client.models.generate_content(
        model=model,
        contents=[
            types.Content(
                role="user",
                parts=[
                    types.Part(
                        text=f"{CLASSIFIER_SYSTEM}\nUser Prompt:\n{user_prompt}"
                    )
                ],
            )
        ],
        config=types.GenerateContentConfig(
            temperature=0.0,  # keep it as deterministic as possible
        ),
    )

    text = (response.text or "").strip()
    risk, level, confidence = None, None, None

    for line in text.splitlines():
        line_stripped = line.strip()
        lower_line = line_stripped.lower()

        if lower_line.startswith("hallucinationrisk:"):
            risk = line_stripped.split(":", 1)[1].strip()
        elif lower_line.startswith("risklevel:"):
            level = line_stripped.split(":", 1)[1].strip()
        elif lower_line.startswith("confidence:"):
            conf_raw = line_stripped.split(":", 1)[1].strip()
            conf_raw = conf_raw.replace("%", "")
            confidence = conf_raw

    return risk, level, confidence

# ============================
# Main
# ============================

def main():
    api_key ="AIzaSyBSaenX-kQbR3CwWtTp_4OaNaAM4Wyq2VE"
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable is not set.")

    client = genai.Client(api_key=api_key)

    seen_prompts = set()

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["prompt", "hallucination_risk", "risk_level", "confidence"])

        for batch_idx in range(NUM_BATCHES):
            print(f"Generating batch {batch_idx + 1}/{NUM_BATCHES}...")
            batch_prompts = generate_prompt_batch(
                client, MODEL_NAME, BATCH_SIZE, seen_prompts
            )

            if not batch_prompts:
                print(f"Warning: no prompts generated in batch {batch_idx + 1}.")
                continue

            print(f"Classifying {len(batch_prompts)} prompts in batch {batch_idx + 1}...")

            for prompt in batch_prompts:
                risk, level, confidence = classify_prompt(client, MODEL_NAME, prompt)
                writer.writerow([prompt, risk, level, confidence])

    print(f"Done. Dataset saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
