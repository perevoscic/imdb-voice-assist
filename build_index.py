import os

from dotenv import load_dotenv
from openai import OpenAI

from imdb_data import ImdbConfig, load_imdb_data
from indexing import build_embeddings
from vector_store import VectorStorePaths, save_vector_store, save_meta


def main():
    load_dotenv()
    client = OpenAI()

    config = ImdbConfig()
    df = load_imdb_data(config)

    texts = df["Overview"].fillna("").astype(str).tolist()
    model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    embeddings = build_embeddings(client, texts, model=model)

    row_ids = df.index.tolist()
    paths = VectorStorePaths()

    save_vector_store(embeddings, row_ids, paths)
    save_meta(
        {
            "rows": len(row_ids),
            "model": model,
            "source": config.csv_path,
        },
        paths,
    )

    print(f"Saved embeddings to {paths.embeddings_path}")


if __name__ == "__main__":
    main()
