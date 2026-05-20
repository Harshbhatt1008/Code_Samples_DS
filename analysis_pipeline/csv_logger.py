"""
CSV logging module for storing pipeline results.
"""

import csv
import os
from typing import Optional
from pathlib import Path

CSV_FILENAME = "results.csv"
CSV_HEADERS = [
    "sr no.",
    "output 1",
    "output 2", 
    "output 3",
    "output 1 score",
    "output 2 score",
    "output 3 score"
]


def get_next_sr_no(csv_path: str) -> int:
    """
    Get the next serial number for the CSV.
    
    Args:
        csv_path: Path to the CSV file
        
    Returns:
        Next serial number
    """
    if not os.path.exists(csv_path):
        return 1
    
    try:
        with open(csv_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            if not rows:
                return 1
            last_sr = rows[-1].get("sr no.", "0")
            return int(last_sr) + 1
    except (ValueError, KeyError):
        return 1


def log_result(
    output1: str,
    output2: str,
    output3: str,
    score1: int,
    score2: int,
    score3: int,
    output_dir: Optional[str] = None
) -> str:
    """
    Log a result to the CSV file.
    
    Args:
        output1: Output from Pipeline 1
        output2: Output from Pipeline 2
        output3: Output from Pipeline 3
        score1: Score for output 1
        score2: Score for output 2
        score3: Score for output 3
        output_dir: Directory for the CSV file (default: current directory)
        
    Returns:
        Path to the CSV file
    """
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))
    
    csv_path = os.path.join(output_dir, CSV_FILENAME)
    file_exists = os.path.exists(csv_path)
    
    sr_no = get_next_sr_no(csv_path)
    
    # Truncate long outputs for CSV readability
    def truncate(text: str, max_len: int = 500) -> str:
        text = text.replace('\n', ' ').replace('\r', ' ')
        if len(text) > max_len:
            return text[:max_len] + "..."
        return text
    
    row = {
        "sr no.": sr_no,
        "output 1": truncate(output1),
        "output 2": truncate(output2),
        "output 3": truncate(output3),
        "output 1 score": score1,
        "output 2 score": score2,
        "output 3 score": score3
    }
    
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow(row)
    
    return csv_path


if __name__ == "__main__":
    # Test the logger
    path = log_result(
        output1="Test output 1",
        output2="Test output 2",
        output3="Test output 3",
        score1=85,
        score2=90,
        score3=70
    )
    print(f"Logged to: {path}")
