"""
Hallucination Classifier client.
"""

import requests
from dataclasses import dataclass
from typing import Dict

CLASSIFIER_URL = "http://localhost:8000"


@dataclass
class ClassificationResult:
    label: str  # LOW, MID, HIGH
    probabilities: Dict[str, float]
    risk_score: float
    
    def needs_web_search(self) -> bool:
        """Returns True if the prompt has HIGH or MID hallucination risk."""
        return self.label in ("HIGH", "MID")


def classify_prompt(prompt: str) -> ClassificationResult:
    """
    Classify a prompt for hallucination risk.
    
    Args:
        prompt: The user prompt to classify
        
    Returns:
        ClassificationResult with label, probabilities, and risk_score
    """
    payload = {"prompt": prompt}
    
    try:
        response = requests.post(
            f"{CLASSIFIER_URL}/predict",
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        result = response.json()
        
        return ClassificationResult(
            label=result["label"],
            probabilities=result["probabilities"],
            risk_score=result["risk_score"]
        )
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Classifier request failed: {e}")


if __name__ == "__main__":
    # Test the client
    result = classify_prompt("What is the capital of France?")
    print(f"Label: {result.label}")
    print(f"Probabilities: {result.probabilities}")
    print(f"Risk Score: {result.risk_score}")
    print(f"Needs Web Search: {result.needs_web_search()}")
