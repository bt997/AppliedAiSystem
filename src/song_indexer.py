"""Indexes Song objects into a Chroma collection for semantic retrieval."""

import logging

from chromadb.api import ClientAPI
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

from src.models import Song

logger = logging.getLogger(__name__)


class SongIndexer:
    """Owns a Chroma collection and upserts songs into it."""

    def __init__(self, chroma_client: ClientAPI, collection_name: str = "music_catalog"):
        self.collection = chroma_client.get_or_create_collection(
            name=collection_name,
            embedding_function=DefaultEmbeddingFunction(),
            metadata={"hnsw:space": "cosine"},
        )

    def index(self, songs: list[Song]) -> None:
        """Embed every song's document representation and upsert into the store."""
        self.collection.upsert(
            ids=[s.id for s in songs],
            documents=[s.to_document() for s in songs],
            metadatas=[
                {
                    "title": s.title,
                    "artist": s.artist,
                    "genre": s.genre,
                    "mood": s.mood,
                }
                for s in songs
            ],
        )
        logger.info("Indexed %d songs into vector store.", len(songs))
