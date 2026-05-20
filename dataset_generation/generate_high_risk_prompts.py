import requests
import csv
import time
import json

# ============================
# CONFIG
# ============================

GENERATOR_MODEL = "mistral"
OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"

NUM_BATCHES = 5     # ~2000 × 5 = 10000 prompts (adjust)
BATCH_SIZE = 5
OUTPUT_CSV = "high_risk_prompts_4.csv"

SLEEP_SECONDS = 0.3


# ============================
# Ollama helper
# ============================

def ollama_generate(prompt: str) -> str:
    payload = {
        "model": GENERATOR_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }
    r = requests.post(OLLAMA_CHAT_URL, json=payload, timeout=30)
    r.raise_for_status()
    return r.json()["message"]["content"].strip()


# ============================
# High-risk generator prompt
# ============================

GENERATOR_PROMPT = f"""
Generate EXACTLY {BATCH_SIZE} UNIQUE user prompts with risk assessments.

Each prompt MUST belong to ONE of these hallucination-prone categories:
- False-premise questions (contain an incorrect assumption)
- Underspecified quantitative questions (missing key variables)
- Casual medical or legal advice requests
- Adversarial or leading questions with biased framing
- Real-time or "latest" questions requiring up-to-date information

RISK LEVELS:
- LOW (0-33): Minor ambiguity, easily clarifiable, low chance of confident hallucination
- MID (34-66): Moderate issues, some missing context, medium hallucination risk
- HIGH (67-100): Severe issues, false premises, or requires real-time data, high hallucination risk

OUTPUT FORMAT (JSON array):
[
  {{"prompt": "prompt text here", "risk_level": "HIGH", "risk_score": 85}},
  {{"prompt": "another prompt", "risk_level": "MID", "risk_score": 50}}
]

STRICT RULES:
- Output ONLY valid JSON array, no explanation
- Do NOT generate fiction, stories, roleplay, or hypothetical worlds
- Do NOT generate textbook definitions or generic explanations
- Prompts must sound like real users, slightly casual
- Avoid repeating common examples
- Each prompt must realistically cause an LLM to hallucinate if answered confidently
- risk_level must be: LOW, MID, or HIGH
- risk_score must be 0-100 and match the risk_level range

Output ONLY the JSON array.
""".strip()


# ============================
# Parse JSON response
# ============================

def parse_prompts(text: str) -> list:
    """Extract JSON array from response, handling potential formatting issues"""
    try:
        # Try direct JSON parse
        data = json.loads(text)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass
    
    # Try to find JSON array in text
    start = text.find('[')
    end = text.rfind(']')
    if start != -1 and end != -1:
        try:
            data = json.loads(text[start:end+1])
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass
    
    return []


# ============================
# Validate prompt entry
# ============================

def validate_entry(entry: dict) -> bool:
    """Check if entry has required fields and valid values"""
    if not isinstance(entry, dict):
        return False
    
    required_keys = ["prompt", "risk_level", "risk_score"]
    if not all(key in entry for key in required_keys):
        return False
    
    if entry["risk_level"] not in ["LOW", "MID", "HIGH"]:
        return False
    
    try:
        score = int(entry["risk_score"])
        if not (0 <= score <= 100):
            return False
    except (ValueError, TypeError):
        return False
    
    return True


# ============================
# Main
# ============================

def main():
    seen = set()
    total_prompts = 0

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["prompt", "risk_level", "risk_score"])

        for i in range(NUM_BATCHES):
            print(f"Generating batch {i+1}/{NUM_BATCHES}")
            try:
                text = ollama_generate(GENERATOR_PROMPT)
                prompts = parse_prompts(text)
                
                if not prompts:
                    print(f"  ⚠️  Batch {i+1}: Could not parse JSON response")
                    continue
                
                valid_count = 0
                for entry in prompts:
                    if not validate_entry(entry):
                        continue
                    
                    prompt_text = entry["prompt"].strip()
                    if prompt_text not in seen:
                        seen.add(prompt_text)
                        writer.writerow([
                            prompt_text,
                            entry["risk_level"],
                            entry["risk_score"]
                        ])
                        valid_count += 1
                        total_prompts += 1
                
                print(f"  ✓ Added {valid_count} unique prompts (Total: {total_prompts})")
                
            except Exception as e:
                print(f"  ⚠️  Batch {i+1} error:", e)
                continue

            time.sleep(SLEEP_SECONDS)

    print(f"\n✅ High-risk prompts saved to {OUTPUT_CSV}")
    print(f"Total unique prompts: {total_prompts}")
    
    # Print statistics
    if total_prompts > 0:
        print("\nGenerating statistics...")
        with open(OUTPUT_CSV, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            risk_counts = {"LOW": 0, "MID": 0, "HIGH": 0}
            for row in reader:
                risk_counts[row["risk_level"]] += 1
        
        print(f"  LOW risk:  {risk_counts['LOW']} ({risk_counts['LOW']/total_prompts*100:.1f}%)")
        print(f"  MID risk:  {risk_counts['MID']} ({risk_counts['MID']/total_prompts*100:.1f}%)")
        print(f"  HIGH risk: {risk_counts['HIGH']} ({risk_counts['HIGH']/total_prompts*100:.1f}%)")


if __name__ == "__main__":
    main()