"""
Tests for the RAG music recommender.

Reliability measures covered
-----------------------------
- Automated tests      : indexing, retrieval relevance, end-to-end pipeline
- Confidence scoring   : distances are valid, ordered, and reflect semantic similarity
- Logging              : INFO-level retrieval logs are emitted during recommend()
- Error handling       : empty query, missing API key
"""

import logging
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.rag_recommender import RAGMusicRecommender, Song, _song_to_document

DATA_PATH = str(Path(__file__).resolve().parent.parent / "data" / "songs.csv")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_recommender() -> RAGMusicRecommender:
    """Create a RAGMusicRecommender with Gemini mocked out (no API key needed)."""
    with patch.dict(os.environ, {"GEMINI_API_KEY": "fake-key"}), \
         patch("google.genai.Client"):
        return RAGMusicRecommender(DATA_PATH)


# ---------------------------------------------------------------------------
# Unit tests — _song_to_document()
# ---------------------------------------------------------------------------

class TestSongToDocument:
    def test_contains_title_and_artist(self):
        song = Song(id="1", title="Night Rain", artist="LoRoom", genre="lofi",
                    mood="chill", energy=0.35, tempo_bpm=75, valence=0.6,
                    danceability=0.5, acousticness=0.85)
        doc = _song_to_document(song)
        assert "Night Rain" in doc
        assert "LoRoom" in doc

    def test_low_energy_label(self):
        song = Song(id="1", title="T", artist="A", genre="ambient", mood="calm",
                    energy=0.2, tempo_bpm=60, valence=0.5, danceability=0.3,
                    acousticness=0.9)
        assert "low-energy" in _song_to_document(song)

    def test_high_energy_label(self):
        song = Song(id="1", title="T", artist="A", genre="rock", mood="intense",
                    energy=0.9, tempo_bpm=160, valence=0.4, danceability=0.7,
                    acousticness=0.1)
        assert "high-energy" in _song_to_document(song)

    def test_acoustic_texture(self):
        song = Song(id="1", title="T", artist="A", genre="folk", mood="calm",
                    energy=0.3, tempo_bpm=70, valence=0.7, danceability=0.4,
                    acousticness=0.8)
        assert "acoustic" in _song_to_document(song)

    def test_electronic_texture(self):
        song = Song(id="1", title="T", artist="A", genre="synthwave", mood="moody",
                    energy=0.7, tempo_bpm=115, valence=0.5, danceability=0.7,
                    acousticness=0.1)
        assert "electronic" in _song_to_document(song)


# ---------------------------------------------------------------------------
# Indexing tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def rec():
    """Single recommender instance shared across retrieval/scoring/logging tests."""
    return _make_recommender()


class TestIndexing:
    def test_all_songs_indexed(self, rec):
        assert rec._collection.count() == 18

    def test_query_returns_k_results(self, rec):
        results = rec._collection.query(query_texts=["music"], n_results=5)
        assert len(results["documents"][0]) == 5


# ---------------------------------------------------------------------------
# Retrieval relevance tests (automated — no API key needed)
#
# These verify that semantic search actually works: a chill query should
# surface lofi/ambient songs, not metal; an energetic query should do
# the opposite. If embeddings were random these would fail ~50% of the time.
# ---------------------------------------------------------------------------

class TestRetrievalRelevance:
    def test_chill_query_returns_calm_songs(self, rec):
        results = rec._collection.query(
            query_texts=["slow chill lofi music for late-night studying"], n_results=3
        )
        combined = " ".join(results["documents"][0]).lower()
        assert any(w in combined for w in ["lofi", "chill", "ambient", "jazz"])
        assert "metal" not in combined

    def test_energetic_query_returns_high_energy_songs(self, rec):
        results = rec._collection.query(
            query_texts=["intense high-energy workout pump-up music"], n_results=3
        )
        combined = " ".join(results["documents"][0]).lower()
        assert "high-energy" in combined
        assert "low-energy" not in combined

    def test_sad_query_returns_melancholic_songs(self, rec):
        results = rec._collection.query(
            query_texts=["sad melancholic rainy day music"], n_results=3
        )
        combined = " ".join(results["documents"][0]).lower()
        assert any(w in combined for w in ["sad", "melancholic", "soul", "classical"])

    def test_genre_specific_query(self, rec):
        results = rec._collection.query(
            query_texts=["jazz coffee shop relaxed vibes"], n_results=1
        )
        top_doc = results["documents"][0][0].lower()
        assert any(w in top_doc for w in ["jazz", "relaxed", "soul"])


# ---------------------------------------------------------------------------
# Confidence scoring tests
#
# ChromaDB returns a "distance" per result (lower = more similar).
# These tests verify that scores are valid and results are correctly ordered.
# ---------------------------------------------------------------------------

class TestConfidenceScoring:
    def test_distances_are_non_negative(self, rec):
        results = rec._collection.query(query_texts=["happy pop music"], n_results=5)
        assert all(d >= 0 for d in results["distances"][0])

    def test_results_ordered_closest_first(self, rec):
        results = rec._collection.query(query_texts=["romantic jazz"], n_results=5)
        distances = results["distances"][0]
        assert distances == sorted(distances), "Closest match should be first"

    def test_strong_match_beats_weak_match(self, rec):
        """A clearly lofi query should rank lofi songs closer than metal songs."""
        lofi_results = rec._collection.query(
            query_texts=["lofi chill beats to study to"], n_results=1
        )
        metal_results = rec._collection.query(
            query_texts=["heavy metal angry aggressive"], n_results=1
        )
        lofi_top = lofi_results["documents"][0][0].lower()
        metal_top = metal_results["documents"][0][0].lower()

        assert "lofi" in lofi_top or "chill" in lofi_top
        assert "metal" in metal_top or "angry" in metal_top

    def test_confidence_formula_in_range(self, rec):
        """Confidence = 1 - distance. For L2 distances on real queries, this
        should stay between 0 and 1 for typical semantic similarity values."""
        results = rec._collection.query(query_texts=["energetic dance music"], n_results=5)
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
        env_without_key = {
            k: v for k, v in os.environ.items()
            if k != "GEMINI_API_KEY"
        }
        with patch.dict(os.environ, env_without_key, clear=True), \
             patch("google.genai.Client"):
            with pytest.raises(KeyError):
                RAGMusicRecommender(DATA_PATH)


# ---------------------------------------------------------------------------
# Logging tests
# ---------------------------------------------------------------------------

class TestLogging:
    def test_retrieval_step_is_logged(self, rec, caplog):
        mock_response = MagicMock()
        mock_response.text = "Here are some recommendations."
        rec._client.models.generate_content.return_value = mock_response

        with caplog.at_level(logging.INFO, logger="src.rag_recommender"):
            rec.recommend("chill music", k=3)

        messages = [r.message for r in caplog.records]
        assert any("retrieved" in m.lower() for m in messages), \
            f"Expected retrieval log, got: {messages}"

    def test_confidence_scores_appear_in_logs(self, rec, caplog):
        mock_response = MagicMock()
        mock_response.text = "Recommendations."
        rec._client.models.generate_content.return_value = mock_response

        with caplog.at_level(logging.INFO, logger="src.rag_recommender"):
            rec.recommend("happy pop", k=2)

        messages = " ".join(r.message for r in caplog.records)
        assert "confidence" in messages.lower()


# ---------------------------------------------------------------------------
# End-to-end tests (Gemini mocked)
# ---------------------------------------------------------------------------

class TestEndToEnd:
    def test_recommend_returns_non_empty_string(self):
        mock_response = MagicMock()
        mock_response.text = "I recommend these songs for your study session."

        with patch.dict(os.environ, {"GEMINI_API_KEY": "fake-key"}), \
             patch("google.genai.Client") as MockClient:
            MockClient.return_value.models.generate_content.return_value = (
                mock_response
            )
            r = RAGMusicRecommender(DATA_PATH)
            result = r.recommend("chill music for studying", k=3)

        assert isinstance(result, str)
        assert result.strip() != ""

    def test_gemini_called_exactly_once(self):
        mock_response = MagicMock()
        mock_response.text = "Recommendations."

        with patch.dict(os.environ, {"GEMINI_API_KEY": "fake-key"}), \
             patch("google.genai.Client") as MockClient:
            mock_models = MockClient.return_value.models
            mock_models.generate_content.return_value = mock_response
            r = RAGMusicRecommender(DATA_PATH)
            r.recommend("happy upbeat music", k=3)

        mock_models.generate_content.assert_called_once()

    def test_retrieved_songs_injected_into_prompt(self):
        """RAG context (retrieved songs) must appear in the prompt to Gemini."""
        captured = []

        def capture(**kwargs):
            captured.append(kwargs.get("contents", ""))
            return MagicMock(text="ok")

        with patch.dict(os.environ, {"GEMINI_API_KEY": "fake-key"}), \
             patch("google.genai.Client") as MockClient:
            MockClient.return_value.models.generate_content.side_effect = (
                capture
            )
            r = RAGMusicRecommender(DATA_PATH)
            r.recommend("chill lofi music", k=3)

        assert len(captured) == 1
        assert "lofi" in captured[0].lower() or "chill" in captured[0].lower(), \
            "Retrieved songs were not injected into the Gemini prompt"
