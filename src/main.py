"""
Command line runner for the RAG Music Recommender.

Requires:
  - GEMINI_API_KEY environment variable set
  - Dependencies installed: pip install -r requirements.txt
"""

import logging
import sys
from pathlib import Path

# Ensure the project root is importable so `from src.X import Y` works when
# this file is invoked as `python src/main.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

from src.rag_recommender import RAGMusicRecommender

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "songs.csv"


def main() -> None:
    print("Initializing RAG music recommender...")
    recommender = RAGMusicRecommender(str(DATA_PATH))

    print("\nDescribe what you want to listen to (natural language).")
    print('Example: "something chill for late-night studying"\n')

    query = input("Your request: ").strip()
    if not query:
        print("No query provided. Exiting.")
        sys.exit(0)

    print("\nGenerating recommendations...\n" + "=" * 50)
    result = recommender.recommend(query)
    print("\n" + result)
    print("=" * 50)

    s = recommender.last_stats
    print(
        f"\nRAG pipeline status: retrieval working — "
        f"{s['n_retrieved']} songs retrieved, "
        f"avg confidence {s['avg_confidence']}, "
        f"top match confidence {s['top_confidence']}."
    )


if __name__ == "__main__":
    main()
