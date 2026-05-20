import re
import csv

input_file = "hallucination_dataset.csv"
output_file = "hallucination_dataset_cleaned.csv"

def clean_question(text: str) -> str:
    # Remove leading numbering like: 1.  12.  3. etc.
    return re.sub(r'^\s*\d+\.\s*', '', text)

with open(input_file, "r", encoding="utf-8", newline="") as infile, \
     open(output_file, "w", encoding="utf-8", newline="") as outfile:

    reader = csv.reader(infile)
    writer = csv.writer(outfile)

    for row in reader:
        if row and len(row) > 0:
            row[0] = clean_question(row[0])
        writer.writerow(row)

print("🎉 Done! Cleaned dataset saved as:", output_file)
