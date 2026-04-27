"""
RAG-based music recommender.

Pipeline
--------
1. Index   – each song is converted to a rich text document, embedded by a
             sentence-transformer model, and stored in a ChromaDB collection.
2. Retrieve – the user's natural-language query is embedded the same way and
              matched against the collection by cosine similarity.
3. Generate – the top-k retrieved songs are passed as context to Gemini, which
              writes a personalised recommendation in natural language.

This replaces the rule-based exact-match scorer in recommender.py.
"""

import csv
import logging
import os
from dataclasses import dataclass

from google import genai
import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Song:
    id: str
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: int
    valence: float
    danceability: float
    acousticness: float


def _load_songs(filepath: str) -> list[Song]:
    songs = []
    with open(filepath, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            songs.append(Song(
                id=row["id"],
                title=row["title"],
                artist=row["artist"],
                genre=row["genre"],
                mood=row["mood"],
                energy=float(row["energy"]),
                tempo_bpm=int(float(row["tempo_bpm"])),
                valence=float(row["valence"]),
                danceability=float(row["danceability"]),
                acousticness=float(row["acousticness"]),
            ))
    return songs


def _song_to_document(song: Song) -> str:
    """
    Convert a Song into a rich prose description used for embedding.

    Dense retrieval works on semantics, so the more descriptive the text the
    better the similarity signal — include every attribute the user might
    naturally mention in a query.
    """
    energy_label = (
        "high-energy" if song.energy > 0.7
        else "medium-energy" if song.energy > 0.4
        else "low-energy"
    )
    texture = "acoustic" if song.acousticness > 0.5 else "electronic"
    dance_label = "very danceable" if song.danceability > 0.75 else "moderately danceable"

    return (
        f'"{song.title}" by {song.artist}. '
        f"Genre: {song.genre}. Mood: {song.mood}. "
        f"{energy_label}, {texture}, {dance_label} at {song.tempo_bpm} BPM."
    )


# ---------------------------------------------------------------------------
# RAG recommender
# ---------------------------------------------------------------------------

class RAGMusicRecommender:
    """
    Retrieval-Augmented Generation music recommender.

    Retrieval  – ChromaDB with the default sentence-transformer embedding
                 function (all-MiniLM-L6-v2 via ONNX Runtime).
    Generation – Gemini 1.5 Flash via the Google Generative AI SDK (free tier).

    Usage
    -----
    Set the GEMINI_API_KEY environment variable, then:

        recommender = RAGMusicRecommender("data/songs.csv")
        print(recommender.recommend("something chill for late-night studying"))
    """

    _COLLECTION = "music_catalog"

    def __init__(self, songs_path: str):
        # Configure Gemini — reads GEMINI_API_KEY from the environment.
        self._client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

        # In-memory ChromaDB instance — no server needed.
        # cosine distance keeps confidence scores in [0, 1].
        self._chroma = chromadb.Client()
        self._collection = self._chroma.get_or_create_collection(
            name=self._COLLECTION,
            embedding_function=DefaultEmbeddingFunction(),
            metadata={"hnsw:space": "cosine"},
        )

        self._index(songs_path)

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def _index(self, songs_path: str) -> None:
        """Embed every song document and upsert into the vector store."""
        songs = _load_songs(songs_path)
        self._collection.upsert(
            ids=[s.id for s in songs],
            documents=[_song_to_document(s) for s in songs],
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

    # ------------------------------------------------------------------
    # Retrieval + Generation
    # ------------------------------------------------------------------

    def recommend(self, user_query: str, k: int = 5) -> str:
        """
        Full RAG pipeline.

        Parameters
        ----------
        user_query : str
            Free-text description of what the user wants to hear.
        k : int
            Number of songs to retrieve from the vector store before
            passing them to Gemini.

        Returns
        -------
        str
            Gemini's natural-language recommendation.
        """
        if not user_query.strip():
            raise ValueError("user_query cannot be empty")

        # --- Step 1: Retrieve ---
        results = self._collection.query(query_texts=[user_query], n_results=k)
        retrieved_docs: list[str] = results["documents"][0]
        distances: list[float] = results["distances"][0]
        confidences = [round(1 - d, 3) for d in distances]

        logger.info("Top %d songs retrieved from vector store:", k)
        for doc, conf in zip(retrieved_docs, confidences):
            logger.info("  [confidence=%.3f] %s", conf, doc)

        catalog_context = "\n".join(f"- {doc}" for doc in retrieved_docs)

        # --- Step 2: Generate ---
        prompt = (
            "You are a music recommendation assistant.\n\n"
            f"User request: {user_query}\n\n"
            "The following songs were retrieved from the catalog as the "
            "closest semantic matches to the user's request:\n"
            f"{catalog_context}\n\n"
            "Recommend the most fitting songs from this list. For each one, "
            "explain specifically why it suits the user's request."
        )

        generation_ok = False
        try:
            response = self._client.models.generate_content(
                model="gemini-2.0-flash-lite",
                contents=prompt,
            )
            generation_ok = True
            text = response.text
        except Exception as e:
            # Extract just the status message, not the full error dict.
            reason = getattr(e, "message", None) or str(e).split(".")[0]
            logger.warning("Gemini unavailable: %s", reason)
            text = (
                "Generation unavailable — here are the top retrieved matches:\n\n"
                + catalog_context
            )

        # Store stats so main.py can display a pipeline summary.
        self.last_stats = {
            "n_retrieved": len(retrieved_docs),
            "avg_confidence": round(sum(confidences) / len(confidences), 3),
            "top_confidence": confidences[0],
            "generation_ok": generation_ok,
        }

        return text
