"""
RAG-based music recommender.

Orchestrates three independent pieces:
  - SongIndexer                : embeds and stores songs in ChromaDB
  - SongRetriever              : runs semantic queries against the store
  - RecommendationGenerator    : turns retrieved docs into natural-language output via Gemini

The orchestrator owns the wiring; each piece owns one concern.
"""

import logging
import os

from google import genai
import chromadb

from src.models import load_songs
from src.recommendation_generator import RecommendationGenerator
from src.song_indexer import SongIndexer
from src.song_retriever import SongRetriever

logger = logging.getLogger(__name__)


class RAGMusicRecommender:
    """Retrieval-Augmented Generation music recommender (orchestrator)."""

    def __init__(self, songs_path: str):
        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        chroma = chromadb.Client()

        self._indexer = SongIndexer(chroma)
        self._indexer.index(load_songs(songs_path))

        self._retriever = SongRetriever(self._indexer.collection)
        self._generator = RecommendationGenerator(client)

        # Exposed for tests / introspection.
        self._collection = self._indexer.collection
        self._client = client

    def recommend(self, user_query: str, k: int = 5) -> str:
        if not user_query.strip():
            raise ValueError("user_query cannot be empty")

        result = self._retriever.retrieve(user_query, k)

        logger.info("Top %d songs retrieved from vector store:", k)
        for doc, conf in zip(result.documents, result.confidences):
            logger.info("  [confidence=%.3f] %s", conf, doc)

        text, generation_ok = self._generator.generate(user_query, result.documents)

        self.last_stats = {
            "n_retrieved": len(result.documents),
            "avg_confidence": result.avg_confidence,
            "top_confidence": result.top_confidence,
            "generation_ok": generation_ok,
        }
        return text
