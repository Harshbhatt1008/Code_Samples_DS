"""
Gemini comparison and scoring module with Google Search grounding.
"""

import os
from google import genai
from google.genai import types
from typing import Dict, Tuple
from dotenv import load_dotenv

load_dotenv()


def get_gemini_client() -> genai.Client:
    """Get configured Gemini client."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable not set")
    return genai.Client(api_key=api_key)


def compare_and_score_outputs(
    original_prompt: str,
    output1: str,
    output2: str,
    output3: str
) -> Tuple[int, int, int]:
    """
    Use Gemini with Google Search grounding to compare and score three outputs.
    
    Args:
        original_prompt: The original user query
        output1: Output from Pipeline 1 (Classifier-controlled web search)
        output2: Output from Pipeline 2 (Always web search)
        output3: Output from Pipeline 3 (No enhancement)
        
    Returns:
        Tuple of (score1, score2, score3) where each score is 0-100
    """
    client = get_gemini_client()
    
    comparison_prompt = f"""You are an expert evaluator. You need to compare three AI-generated responses to the same question and score each one based on:

1. **Factual Accuracy**: Is the information correct and verifiable?
2. **Relevance**: Does it directly address the question?
3. **Completeness**: Does it provide a thorough answer?
4. **Coherence**: Is it well-structured and easy to understand?

Use Google Search to verify facts when needed.

**Original Question:**
{original_prompt}

**Response 1 (Classifier-controlled with conditional web search):**
{output1}

**Response 2 (Always uses web search):**
{output2}

**Response 3 (No web search - baseline):**
{output3}

**Your Task:**
Score each response from 0 to 100. Higher scores mean better quality.

Respond ONLY in this exact format:
SCORE1: [number]
SCORE2: [number]
SCORE3: [number]
REASONING: [brief explanation]
"""

    model = "gemini-2.5-flash"
    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=comparison_prompt)],
        ),
    ]
    
    # Enable Google Search grounding
    tools = [
        types.Tool(googleSearch=types.GoogleSearch()),
    ]
    
    generate_content_config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(
            thinking_budget=-1,  # Extended reasoning
        ),
        tools=tools,
        temperature=0.2,  # Lower temperature for more consistent scoring
    )
    
    # Collect full response
    full_response = ""
    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        if chunk.text:
            full_response += chunk.text
    
    # Parse scores from response
    scores = parse_scores(full_response)
    return scores


def parse_scores(response: str) -> Tuple[int, int, int]:
    """
    Parse scores from Gemini response.
    
    Args:
        response: Raw response text from Gemini
        
    Returns:
        Tuple of (score1, score2, score3)
    """
    import re
    
    score1 = score2 = score3 = 0
    
    # Try to find SCORE1, SCORE2, SCORE3 patterns
    score1_match = re.search(r'SCORE1:\s*(\d+)', response, re.IGNORECASE)
    score2_match = re.search(r'SCORE2:\s*(\d+)', response, re.IGNORECASE)
    score3_match = re.search(r'SCORE3:\s*(\d+)', response, re.IGNORECASE)
    
    if score1_match:
        score1 = min(100, max(0, int(score1_match.group(1))))
    if score2_match:
        score2 = min(100, max(0, int(score2_match.group(1))))
    if score3_match:
        score3 = min(100, max(0, int(score3_match.group(1))))
    
    return (score1, score2, score3)


if __name__ == "__main__":
    # Test the scorer
    scores = compare_and_score_outputs(
        original_prompt="What is the capital of France?",
        output1="The capital of France is Paris.",
        output2="Paris is the capital of France, located on the Seine River.",
        output3="France's capital is Paris."
    )
    print(f"Scores: {scores}")
