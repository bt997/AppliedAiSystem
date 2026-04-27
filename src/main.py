"""
Command line runner for the RAG Music Recommender.

Requires:
  - G environment variable set
  - Dependencies installed: pip install -r requirements.txt
"""

import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

from rag_recommender import RAGMusicRecommender

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
    gen_status = "active" if s["generation_ok"] else "unavailable (API key issue)"
    print(
        f"\nRAG pipeline status: retrieval working — "
        f"{s['n_retrieved']} songs retrieved, "
        f"avg confidence {s['avg_confidence']}, "
        f"top match confidence {s['top_confidence']}, "
        f"generation {gen_status}."
    )


if __name__ == "__main__":
    main()
