"""
RAG-based music recommender.

Orchestrates three independent pieces:
  - SongIndexer                : embeds and stores songs in ChromaDB
  - SongRetriever              : runs semantic queries against the store
  - RecommendationGenerator    : turns retrieved docs into prose

The orchestrator does not construct its dependencies — wire them up with
`build_recommender(songs_path)` or pass them in directly.
"""

import logging
import os
from dataclasses import dataclass

from google import genai
import chromadb

from src.models import load_songs
from src.recommendation_generator import RecommendationGenerator
from src.song_indexer import SongIndexer
from src.song_retriever import SongRetriever

logger = logging.getLogger(__name__)


@dataclass
class RecommendationResult:
    text: str
    n_retrieved: int
    avg_confidence: float
    top_confidence: float
    generation_ok: bool


class RAGMusicRecommender:
    """Retrieval-Augmented Generation music recommender (orchestrator)."""

    def __init__(
        self,
        indexer: SongIndexer,
        retriever: SongRetriever,
        generator: RecommendationGenerator,
    ):
        self._indexer = indexer
        self._retriever = retriever
        self._generator = generator

    def recommend(self, user_query: str, k: int = 5) -> RecommendationResult:
        if not user_query.strip():
            raise ValueError("user_query cannot be empty")

        retrieval = self._retriever.retrieve(user_query, k)

        logger.info("Top %d songs retrieved from vector store:", k)
        for doc, conf in zip(retrieval.documents, retrieval.confidences):
            logger.info("  [confidence=%.3f] %s", conf, doc)

        text, generation_ok = self._generator.generate(
            user_query, retrieval.documents,
        )

        return RecommendationResult(
            text=text,
            n_retrieved=len(retrieval.documents),
            avg_confidence=retrieval.avg_confidence,
            top_confidence=retrieval.top_confidence,
            generation_ok=generation_ok,
        )


def build_recommender(songs_path: str) -> RAGMusicRecommender:
    """Wire up the production recommender: real Chroma + real Gemini client."""
    gemini = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    chroma = chromadb.Client()

    indexer = SongIndexer(chroma)
    indexer.index(load_songs(songs_path))

    return RAGMusicRecommender(
        indexer=indexer,
        retriever=SongRetriever(indexer.collection),
        generator=RecommendationGenerator(gemini),
    )
