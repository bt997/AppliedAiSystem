"""
Tests for the RAG music recommender.

Reliability measures covered
-----------------------------
- Automated tests      : indexing, retrieval relevance, end-to-end pipeline
- Confidence scoring   : distances are valid, ordered, semantically meaningful
- Logging              : INFO retrieval logs are emitted during recommend()
- Error handling       : empty query, missing API key
"""

import logging
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import chromadb
import pytest

from src.models import Song, load_songs
from src.rag_recommender import (
    RAGMusicRecommender,
    RecommendationResult,
    build_recommender,
)
from src.recommendation_generator import RecommendationGenerator
from src.song_indexer import SongIndexer
from src.song_retriever import SongRetriever

DATA_PATH = str(Path(__file__).resolve().parent.parent / "data" / "songs.csv")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def collection():
    """An indexed Chroma collection, shared by read-only retrieval tests."""
    chroma = chromadb.Client()
    indexer = SongIndexer(chroma)
    indexer.index(load_songs(DATA_PATH))
    return indexer.collection


@pytest.fixture
def gemini():
    """A fresh Gemini mock for each test that needs one."""
    return MagicMock()


@pytest.fixture
def rec(collection, gemini):
    """Recommender wired with the shared collection and a fresh Gemini mock."""
    chroma = chromadb.Client()
    indexer = SongIndexer(chroma)
    return RAGMusicRecommender(
        indexer=indexer,
        retriever=SongRetriever(collection),
        generator=RecommendationGenerator(gemini),
    )


# ---------------------------------------------------------------------------
# Unit tests — Song.to_document()
# ---------------------------------------------------------------------------

class TestSongToDocument:
    def test_contains_title_and_artist(self):
        song = Song(id="1", title="Night Rain", artist="LoRoom", genre="lofi",
                    mood="chill", energy=0.35, tempo_bpm=75, valence=0.6,
                    danceability=0.5, acousticness=0.85)
        doc = song.to_document()
        assert "Night Rain" in doc
        assert "LoRoom" in doc

    def test_low_energy_label(self):
        song = Song(id="1", title="T", artist="A", genre="ambient",
                    mood="calm", energy=0.2, tempo_bpm=60, valence=0.5,
                    danceability=0.3, acousticness=0.9)
        assert "low-energy" in song.to_document()

    def test_high_energy_label(self):
        song = Song(id="1", title="T", artist="A", genre="rock",
                    mood="intense", energy=0.9, tempo_bpm=160, valence=0.4,
                    danceability=0.7, acousticness=0.1)
        assert "high-energy" in song.to_document()

    def test_acoustic_texture(self):
        song = Song(id="1", title="T", artist="A", genre="folk", mood="calm",
                    energy=0.3, tempo_bpm=70, valence=0.7, danceability=0.4,
                    acousticness=0.8)
        assert "acoustic" in song.to_document()

    def test_electronic_texture(self):
        song = Song(id="1", title="T", artist="A", genre="synthwave",
                    mood="moody", energy=0.7, tempo_bpm=115, valence=0.5,
                    danceability=0.7, acousticness=0.1)
        assert "electronic" in song.to_document()


# ---------------------------------------------------------------------------
# Indexing tests
# ---------------------------------------------------------------------------

class TestIndexing:
    def test_all_songs_indexed(self, collection):
        assert collection.count() == 18

    def test_query_returns_k_results(self, collection):
        results = collection.query(query_texts=["music"], n_results=5)
        assert len(results["documents"][0]) == 5


# ---------------------------------------------------------------------------
# Retrieval relevance tests (automated — no API key needed)
# ---------------------------------------------------------------------------

class TestRetrievalRelevance:
    def test_chill_query_returns_calm_songs(self, collection):
        results = collection.query(
            query_texts=["slow chill lofi music for late-night studying"],
            n_results=3,
        )
        combined = " ".join(results["documents"][0]).lower()
        assert any(w in combined for w in ["lofi", "chill", "ambient", "jazz"])
        assert "metal" not in combined

    def test_energetic_query_returns_high_energy_songs(self, collection):
        results = collection.query(
            query_texts=["intense high-energy workout pump-up music"],
            n_results=3,
        )
        docs = [d.lower() for d in results["documents"][0]]
        assert "high-energy" in docs[0]
        high_count = sum("high-energy" in d for d in docs)
        assert high_count >= 2, f"Expected majority high-energy, got: {docs}"

    def test_sad_query_returns_melancholic_songs(self, collection):
        results = collection.query(
            query_texts=["sad melancholic rainy day music"], n_results=3,
        )
        combined = " ".join(results["documents"][0]).lower()
        assert any(
            w in combined
            for w in ["sad", "melancholic", "soul", "classical"]
        )

    def test_genre_specific_query(self, collection):
        results = collection.query(
            query_texts=["jazz coffee shop relaxed vibes"], n_results=1,
        )
        top_doc = results["documents"][0][0].lower()
        assert any(w in top_doc for w in ["jazz", "relaxed", "soul"])


# ---------------------------------------------------------------------------
# Confidence scoring tests
# ---------------------------------------------------------------------------

class TestConfidenceScoring:
    def test_distances_are_non_negative(self, collection):
        results = collection.query(
            query_texts=["happy pop music"], n_results=5,
        )
        assert all(d >= 0 for d in results["distances"][0])

    def test_results_ordered_closest_first(self, collection):
        results = collection.query(
            query_texts=["romantic jazz"], n_results=5,
        )
        distances = results["distances"][0]
        assert distances == sorted(distances), \
            "Closest match should be first"

    def test_strong_match_beats_weak_match(self, collection):
        """A lofi query should rank lofi songs closer than metal songs."""
        lofi_results = collection.query(
            query_texts=["lofi chill beats to study to"], n_results=1,
        )
        metal_results = collection.query(
            query_texts=["heavy metal angry aggressive"], n_results=1,
        )
        lofi_top = lofi_results["documents"][0][0].lower()
        metal_top = metal_results["documents"][0][0].lower()

        assert "lofi" in lofi_top or "chill" in lofi_top
        assert "metal" in metal_top or "angry" in metal_top

    def test_confidence_formula_in_range(self, collection):
        """Confidence = 1 - distance. For cosine distances on real queries this
        should stay in [0, 1] for typical semantic similarity values."""
        results = collection.query(
            query_texts=["energetic dance music"], n_results=5,
        )
        confidences = [1 - d for d in results["distances"][0]]
        assert all(-0.1 <= c <= 1.1 for c in confidences), \
            f"Unexpected confidence range: {confidences}"


# ---------------------------------------------------------------------------
# Error handling tests
# ---------------------------------------------------------------------------

class TestErrorHandling:
    def test_empty_query_raises_value_error(self, rec):
        with pytest.raises(ValueError, match="cannot be empty"):
            rec.recommend("")

    def test_whitespace_query_raises_value_error(self, rec):
        with pytest.raises(ValueError, match="cannot be empty"):
            rec.recommend("   ")

    def test_missing_api_key_raises_key_error(self):
        """build_recommender reads GEMINI_API_KEY; orchestrator does not."""
        env_without_key = {
            k: v for k, v in os.environ.items() if k != "GEMINI_API_KEY"
        }
        with patch.dict(os.environ, env_without_key, clear=True), \
             patch("google.genai.Client"):
            with pytest.raises(KeyError):
                build_recommender(DATA_PATH)


# ---------------------------------------------------------------------------
# Logging tests
# ---------------------------------------------------------------------------

class TestLogging:
    def test_retrieval_step_is_logged(self, rec, gemini, caplog):
        gemini.models.generate_content.return_value = MagicMock(
            text="Here are some recommendations."
        )
        with caplog.at_level(logging.INFO, logger="src.rag_recommender"):
            rec.recommend("chill music", k=3)

        messages = [r.message for r in caplog.records]
        assert any("retrieved" in m.lower() for m in messages), \
            f"Expected retrieval log, got: {messages}"

    def test_confidence_scores_appear_in_logs(self, rec, gemini, caplog):
        gemini.models.generate_content.return_value = MagicMock(
            text="Recommendations."
        )
        with caplog.at_level(logging.INFO, logger="src.rag_recommender"):
            rec.recommend("happy pop", k=2)

        messages = " ".join(r.message for r in caplog.records)
        assert "confidence" in messages.lower()


# ---------------------------------------------------------------------------
# End-to-end tests (Gemini mocked, dependencies injected)
# ---------------------------------------------------------------------------

class TestEndToEnd:
    def test_recommend_returns_recommendation_result(self, rec, gemini):
        gemini.models.generate_content.return_value = MagicMock(
            text="I recommend these songs for your study session."
        )
        result = rec.recommend("chill music for studying", k=3)
        assert isinstance(result, RecommendationResult)
        assert result.text.strip() != ""
        assert result.n_retrieved == 3
        assert result.generation_ok is True

    def test_gemini_called_exactly_once(self, rec, gemini):
        gemini.models.generate_content.return_value = MagicMock(
            text="Recommendations."
        )
        rec.recommend("happy upbeat music", k=3)
        gemini.models.generate_content.assert_called_once()

    def test_retrieved_songs_injected_into_prompt(self, rec, gemini):
        """Retrieved songs must appear in the prompt sent to Gemini."""
        captured = []

        def capture(**kwargs):
            captured.append(kwargs.get("contents", ""))
            return MagicMock(text="ok")

        gemini.models.generate_content.side_effect = capture
        rec.recommend("chill lofi music", k=3)

        assert len(captured) == 1
        assert (
            "lofi" in captured[0].lower() or "chill" in captured[0].lower()
        ), "Retrieved songs were not injected into the Gemini prompt"

    def test_generation_failure_falls_back_gracefully(self, rec, gemini):
        gemini.models.generate_content.side_effect = RuntimeError("quota")
        result = rec.recommend("chill music", k=3)
        assert result.generation_ok is False
        assert "Generation unavailable" in result.text
