import pandas as pd
import os
import csv

def detect_problem_lines(csv_path):
    print("🔍 Scanning for problematic lines...")
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=",", quotechar='"')
        for i, row in enumerate(reader, start=1):
            try:
                # Expecting exactly 3 fields: question, answer, links
                if len(row) != 3:
                    print(f"❌ Line {i} has {len(row)} fields: {row}")
            except Exception as e:
                print(f"⚠️ Error parsing line {i}: {e}")

def clean_and_resave_csv(input_csv="knowledgbase.csv", output_csv="kb_semicolumn.csv"):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(base_dir, input_csv)
    output_path = os.path.join(base_dir, output_csv)

    try:
        # First check for broken lines
        detect_problem_lines(input_path)

        # Then try to load with pandas
        df = pd.read_csv(
            input_path,
            delimiter=",",
            quotechar='"',
            quoting=csv.QUOTE_MINIMAL,
            encoding="utf-8"
        )

        # Strip whitespace
        for col in df.select_dtypes(include=["object"]).columns:
            df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)

        df.to_csv(output_path, sep=";", index=False, quoting=csv.QUOTE_NONNUMERIC)
        print(f"✅ CSV cleaned and saved with semicolons at: {output_path}")

    except Exception as e:
        print(f"❌ Error processing CSV: {e}")

if __name__ == "__main__":
    clean_and_resave_csv()
