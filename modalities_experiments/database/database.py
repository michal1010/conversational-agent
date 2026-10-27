import hashlib
import json
import chromadb


def get_text_id(text: str) -> str:
    """SHA-256 hash of the text used as the document ID."""
    return hashlib.sha256(text.encode()).hexdigest()


class PromptDatabase:
    def __init__(self, path: str = "./memory_storage_chroma", collection_name: str = "prompts"):
        self.client = chromadb.PersistentClient(path=path)
        self.collection = self.client.get_or_create_collection(collection_name)

    def add(self, text: str, emotions: dict[str, float]) -> str:
        """Store a (text, emotions) tuple. Text is embedded; emotions stored in metadata."""
        doc_id = get_text_id(text)
        self.collection.upsert(
            ids=[doc_id],
            documents=[text],
            metadatas=[{"emotions": json.dumps(emotions)}],
        )
        return doc_id

    def query(self, query_text: str, n_results: int = 5) -> list[dict]:
        """Semantic search — returns all stored prompts ranked by similarity to query_text."""
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results,
        )
        return [
            {
                "text": doc,
                "emotions": json.loads(meta["emotions"]) if meta else None,
                "distance": dist,
            }
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ]

    def get(self, text: str) -> dict | None:
        """Exact lookup by text — returns {text, emotions} or None."""
        doc_id = get_text_id(text)
        result = self.collection.get(ids=[doc_id])
        if not result["documents"]:
            return None
        meta = result["metadatas"][0]
        return {
            "text": result["documents"][0],
            "emotions": json.loads(meta["emotions"]) if meta else None,
        }

    def delete(self, text: str) -> None:
        doc_id = get_text_id(text)
        self.collection.delete(ids=[doc_id])


if __name__ == "__main__":
    db = PromptDatabase()

    entries = [
        ("Summarize the following text in three sentences.",     {"neutral": 0.6, "curious": 0.4}),
        ("Give me a brief overview of the article.",             {"neutral": 0.7, "curious": 0.3}),
        ("What are the key points of this document?",           {"curious": 0.8, "neutral": 0.2}),
        ("Translate the following text to French.",              {"happy": 0.7, "sad": 0.3}),
        ("Convert this paragraph into Spanish.",                 {"happy": 0.6, "neutral": 0.4}),
        ("Extract all named entities from the text below.",      {"neutral": 0.9, "anxious": 0.1}),
        ("List every person, place, and organization mentioned.", {"neutral": 0.8, "curious": 0.2}),
        ("Rewrite this in a formal academic tone.",              {"neutral": 0.5, "anxious": 0.5}),
        ("Make this text sound more professional.",              {"neutral": 0.6, "happy": 0.4}),
        ("Write a Python function that sorts a list.",           {"curious": 0.9, "happy": 0.1}),
        ("Generate a haiku about the ocean.",                    {"happy": 0.8, "sad": 0.2}),
        ("Explain quantum entanglement to a 10-year-old.",       {"curious": 1.0}),
    ]

    print("Storing prompts...\n")
    for text, emotions in entries:
        db.add(text, emotions)

    query = "Summarize this article briefly"
    print(f'Query: "{query}"\n')
    print(f"{'Distance':>10}  Emotions                          Prompt")
    print("-" * 100)

    results = db.query(query, n_results=len(entries))
    for r in results:
        emotions_str = ", ".join(f"{k}:{v}" for k, v in r["emotions"].items())
        print(f"  {r['distance']:.4f}    {emotions_str:<32}  {r['text']}")
