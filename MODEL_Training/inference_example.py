import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = "models/hallucination_binary"

label_map = {0: "No", 1: "Yes"}

def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()
    return tokenizer, model

def predict(prompt, tokenizer, model, max_length=128):
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1).squeeze(0)

    pred_label_id = int(torch.argmax(probs).item())
    pred_label = label_map[pred_label_id]
    prob_yes = float(probs[1].item())

    return {
        "prompt": prompt,
        "hallucination_risk": pred_label,
        "p_yes": prob_yes,
    }

if __name__ == "__main__":
    tokenizer, model = load_model()

    test_prompts = [
        "What are some potential impacts of artificial intelligence on the job market by 2050",
        "In J.R.R. Tolkien's Middle Earth, what is the history and culture of the Elven kingdom of Rivendell?",
        "If a cat suddenly developed the ability to speak human language fluently, how might that change its relationship with its owner?",
        "Describe the process of photosynthesis and its importance to life on Earth.",
        "Who was Queen Elizabeth I and what were her major accomplishments during her reign?",
        "Imagine a world where unicorns exist, describe their physical characteristics and habitats.",
        "What are the side effects and risks associated with long-term use of ibuprofen?",
        "Explain the Miranda Rights as they are stated in the United States legal system.",
        "Describe the structure of DNA and its role in heredity.",
            ]

    for p in test_prompts:
        result = predict(p, tokenizer, model)
        print(result)
