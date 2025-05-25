import os
import pandas as pd
import numpy as np
import openai
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    response = client.embeddings.create(
        model=model,
        input=[text]
    )
    return response.data[0].embedding

def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def embed_qa_pairs(df, model="text-embedding-3-small"):
    # Combine question + answer to represent the semantic meaning
    print("📌 Available columns:", df.columns.tolist())

    df["embedding"] = df.apply(
        lambda row: get_embedding(f"שאלה: {row['question']} תשובה: {row['answer']}", model=model),
        axis=1
    )
    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate and save embeddings for a knowledge base CSV.")
    parser.add_argument("--csv-path", type=str, default="kb_semicolumn.csv", help="Path to your input CSV file")
    parser.add_argument("--output-path", type=str, default="knowledge_base_with_embeddings.pkl", help="Output path to save embeddings")

    args = parser.parse_args()

    df = pd.read_csv(args.csv_path, delimiter=";")
    df = embed_qa_pairs(df)
    df.to_pickle(args.output_path)

    print(f"✅ Embeddings saved to {args.output_path}")
