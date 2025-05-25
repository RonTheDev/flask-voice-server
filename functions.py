import pandas as pd
from embeddings import get_embedding, cosine_similarity

# Load the CSV with embeddings (assumes this was created earlier)
df = pd.read_pickle("knowledge_base_with_embeddings.pkl")

def query_knowledgebase(query: str, top_k: int = 3):
    query_embedding = get_embedding(query, model="text-embedding-3-small")
    df["similarity"] = df["embedding"].apply(lambda x: cosine_similarity(x, query_embedding))
    top = df.sort_values("similarity", ascending=False).head(top_k)
    return {
        "results": [
            {
                "question": row["question"],
                "answer": row["answer"],
                "link": row["links"]
            }
            for _, row in top.iterrows()
        ]
    }



tool_definitions = [
    {
        "type": "function",
        "function": {
            "name": "query_knowledgebase",
            "description": "Use this tool to answer every user input",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The question asked by the user."
                    },
                    "top_k": {
                        "type": "integer",
                        "default": 3,
                        "description": "Number of top similar results to return."
                    }
                },
                "required": ["query"]
            }
        }
    }
]
