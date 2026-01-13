import csv
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent  # notebook/
PROJECT_DIR = BASE_DIR.parent              # WR_Project/

input_txt = PROJECT_DIR / "data" / "kh-polar.ver1.0.txt"
output_csv = PROJECT_DIR / "data" / "kh-polar.ver1.0.csv"


with open(input_txt, "r", encoding="utf-8") as txt_file, \
     open(output_csv, "w", newline="", encoding="utf-8") as csv_file:

    writer = csv.writer(csv_file)

    # Write CSV header
    writer.writerow(["text", "keyword", "label"])

    for line in txt_file:
        parts = line.strip().split("|||")
        if len(parts) == 3:
            text = parts[0].strip()
            keyword = parts[1].strip()
            label = parts[2].strip()

            writer.writerow([text, keyword, label])

print("TXT file converted to CSV successfully.")
