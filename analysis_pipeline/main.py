"""
Multi-Pipeline Hallucination Analysis System

This system runs three parallel pipelines with different hallucination mitigation strategies:
- Pipeline 1: Classifier-controlled web search (searches only if HIGH/MID risk)
- Pipeline 2: Always uses web search
- Pipeline 3: No enhancement (baseline)

All outputs are compared and scored by Gemini with Google Search grounding.
"""

import sys
import os
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Tuple, List

from mistral_client import generate_with_mistral
from classifier_client import classify_prompt
from search_client import search_and_format
from gemini_scorer import compare_and_score_outputs
from csv_logger import log_result

# Path to the prompts CSV file
PROMPTS_CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts.csv")


def load_prompts_from_csv(filepath: str = PROMPTS_CSV_PATH) -> List[str]:
    """
    Load prompts from a CSV file.
    
    Args:
        filepath: Path to the CSV file containing prompts (one per line)
        
    Returns:
        List of prompt strings
    """
    prompts = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if row and row[0].strip():  # Skip empty rows
                prompts.append(row[0].strip())
    return prompts


def pipeline1_classifier_controlled(prompt: str) -> str:
    """
    Pipeline 1: Classify prompt, use web search if HIGH/MID risk.
    
    1. Classify user prompt for hallucination risk
    2. If HIGH or MID risk → perform web search first
    3. Generate response with Mistral (with search context if risk was high)
    """
    print("[Pipeline 1] Classifying prompt for hallucination risk...")
    
    try:
        classification = classify_prompt(prompt)
        print(f"[Pipeline 1] Classification: {classification.label} (risk score: {classification.risk_score})")
        
        if classification.needs_web_search():
            print("[Pipeline 1] HIGH/MID risk detected - performing web search...")
            context = search_and_format(prompt)
            response = generate_with_mistral(prompt, context=context)
        else:
            print("[Pipeline 1] LOW risk - generating without web search...")
            response = generate_with_mistral(prompt)
            
    except Exception as e:
        print(f"[Pipeline 1] Classifier error: {e}, falling back to no-search mode")
        response = generate_with_mistral(prompt)
    
    return response


def pipeline2_always_websearch(prompt: str) -> str:
    """
    Pipeline 2: Always perform web search before generating.
    """
    print("[Pipeline 2] Performing web search...")
    
    try:
        context = search_and_format(prompt)
        print("[Pipeline 2] Web search complete, generating response...")
        response = generate_with_mistral(prompt, context=context)
    except Exception as e:
        print(f"[Pipeline 2] Web search error: {e}, generating without context")
        response = generate_with_mistral(prompt)
    
    return response


def pipeline3_no_enhancement(prompt: str) -> str:
    """
    Pipeline 3: Direct generation with no enhancements (baseline).
    """
    print("[Pipeline 3] Generating response without enhancements...")
    response = generate_with_mistral(prompt)
    return response


def run_all_pipelines(prompt: str) -> Tuple[str, str, str]:
    """
    Run all three pipelines in parallel.
    
    Returns:
        Tuple of (output1, output2, output3)
    """
    print("\n" + "="*60)
    print("Running all pipelines...")
    print("="*60 + "\n")
    
    results = {}
    
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(pipeline1_classifier_controlled, prompt): "pipeline1",
            executor.submit(pipeline2_always_websearch, prompt): "pipeline2",
            executor.submit(pipeline3_no_enhancement, prompt): "pipeline3",
        }
        
        for future in as_completed(futures):
            pipeline_name = futures[future]
            try:
                results[pipeline_name] = future.result()
            except Exception as e:
                print(f"[{pipeline_name}] Failed with error: {e}")
                results[pipeline_name] = f"Error: {e}"
    
    return (
        results.get("pipeline1", "Error"),
        results.get("pipeline2", "Error"),
        results.get("pipeline3", "Error")
    )


def main():
    """
    Main entry point: Load prompts from CSV and process each one.
    """
    print("\n" + "="*60)
    print("Multi-Pipeline Hallucination Analysis System")
    print("="*60)
    
    # Load prompts from CSV
    try:
        prompts = load_prompts_from_csv()
        print(f"\nLoaded {len(prompts)} prompts from {PROMPTS_CSV_PATH}")
    except FileNotFoundError:
        print(f"Error: prompts.csv not found at {PROMPTS_CSV_PATH}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading prompts: {e}")
        sys.exit(1)
    
    if not prompts:
        print("Error: No prompts found in prompts.csv")
        sys.exit(1)
    
    # Process each prompt
    for i, prompt in enumerate(prompts, start=1):
        print(f"\n{'#'*60}")
        print(f"Processing prompt {i}/{len(prompts)}")
        print(f"{'#'*60}")
        print(f"Prompt: {prompt[:100]}..." if len(prompt) > 100 else f"Prompt: {prompt}")
        
        try:
            process_prompt(prompt)
        except Exception as e:
            print(f"Error processing prompt {i}: {e}")
            continue
    
    print(f"\n{'='*60}")
    print(f"Completed processing all {len(prompts)} prompts.")
    print(f"{'='*60}")


def process_prompt(prompt: str):
    """Process a single prompt through all pipelines and score results."""
    
    # Run all pipelines
    output1, output2, output3 = run_all_pipelines(prompt)
    
    # Display outputs
    print("\n" + "="*60)
    print("PIPELINE OUTPUTS")
    print("="*60)
    
    print("\n[Output 1 - Classifier Controlled]")
    print("-" * 40)
    print(output1[:500] + "..." if len(output1) > 500 else output1)
    
    print("\n[Output 2 - Always Web Search]")
    print("-" * 40)
    print(output2[:500] + "..." if len(output2) > 500 else output2)
    
    print("\n[Output 3 - No Enhancement]")
    print("-" * 40)
    print(output3[:500] + "..." if len(output3) > 500 else output3)
    
    # Score with Gemini
    print("\n" + "="*60)
    print("Scoring outputs with Gemini...")
    print("="*60 + "\n")
    
    try:
        score1, score2, score3 = compare_and_score_outputs(
            original_prompt=prompt,
            output1=output1,
            output2=output2,
            output3=output3
        )
        
        print(f"[Scores]")
        print(f"  Output 1 (Classifier Controlled): {score1}/100")
        print(f"  Output 2 (Always Web Search):     {score2}/100")
        print(f"  Output 3 (No Enhancement):        {score3}/100")
        
        # Log to CSV
        csv_path = log_result(
            output1=output1,
            output2=output2,
            output3=output3,
            score1=score1,
            score2=score2,
            score3=score3
        )
        print(f"\n[Results saved to: {csv_path}]")
        
    except Exception as e:
        print(f"[Error scoring outputs: {e}]")
        # Still log without scores
        csv_path = log_result(
            output1=output1,
            output2=output2,
            output3=output3,
            score1=0,
            score2=0,
            score3=0
        )
        print(f"[Results saved without scores to: {csv_path}]")


if __name__ == "__main__":
    main()
