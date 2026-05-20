"""
Mistral client using Ollama API.
"""

import requests
from typing import Optional

OLLAMA_URL = "http://localhost:11434"


def generate_with_mistral(
    prompt: str,
    context: Optional[str] = None,
    model: str = "mistral"
) -> str:
    """
    Generate a response using Mistral via Ollama.
    
    Args:
        prompt: The user's prompt/question
        context: Optional context from web search to include
        model: Model name (default: mistral)
    
    Returns:
        Generated response text
    """
    # Build the full prompt with context if provided
    if context:
        full_prompt = f"""Use the following context to answer the question accurately.

Context:
{context}

Question: {prompt}

Answer:"""
    else:
        full_prompt = prompt
    
    payload = {
        "model": model,
        "prompt": full_prompt,
        "stream": False
    }
    
    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json=payload,
            timeout=120
        )
        response.raise_for_status()
        result = response.json()
        return result.get("response", "")
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Ollama request failed: {e}")


if __name__ == "__main__":
    # Test the client
    result = generate_with_mistral("What is the capital of France?")
    print(result)
