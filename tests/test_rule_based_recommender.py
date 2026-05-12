"""Tests for the rule-based scoring fallback in src.rule_based_recommender."""

import logging

import pytest

from src.models import Song
from src.rule_based_recommender import recommend_songs, score_song


def _song(**overrides) -> Song:
    """Build a Song with sensible defaults; override only what each test needs."""
    defaults = dict(
        id="1",
        title="Test Track",
        artist="Test Artist",
        genre="pop",
        mood="happy",
        energy=0.8,
        tempo_bpm=120,
        valence=0.8,
        danceability=0.7,
        acousticness=0.2,
    )
    defaults.update(overrides)
    return Song(**defaults)


# ---------------------------------------------------------------------------
# score_song
# ---------------------------------------------------------------------------

class TestScoreSong:
    def test_no_match_scores_zero(self):
        song = _song(genre="rock", mood="intense", artist="Other")
        prefs = {"genre": "pop", "mood": "happy", "artist": "Me"}
        score, reasons = score_song(prefs, song)
        assert score == 0
        assert reasons == []

    def test_genre_match_adds_two(self):
        song = _song(genre="pop", mood="intense", artist="Other")
        score, reasons = score_song({"genre": "pop"}, song)
        assert score == 2
        assert "genre match (+2)" in reasons

    def test_mood_match_adds_three(self):
        song = _song(genre="rock", mood="happy", artist="Other")
        score, reasons = score_song({"mood": "happy"}, song)
        assert score == 3
        assert "mood match (+3)" in reasons

    def test_artist_match_adds_five(self):
        song = _song(genre="rock", mood="intense", artist="Neon Echo")
        score, reasons = score_song({"artist": "Neon Echo"}, song)
        assert score == 5
        assert "artist match (+5)" in reasons

    def test_all_three_match_sums_to_ten(self):
        song = _song(genre="pop", mood="happy", artist="Neon Echo")
        prefs = {"genre": "pop", "mood": "happy", "artist": "Neon Echo"}
        score, reasons = score_song(prefs, song)
        assert score == 10
        assert len(reasons) == 3

    def test_match_is_case_insensitive(self):
        song = _song(genre="POP", mood="Happy", artist="NEON ECHO")
        prefs = {"genre": "pop", "mood": "happy", "artist": "neon echo"}
        score, _ = score_song(prefs, song)
        assert score == 10


# ---------------------------------------------------------------------------
# recommend_songs
# ---------------------------------------------------------------------------

class TestRecommendSongs:
    def test_returns_top_k(self):
        songs = [_song(id=str(i), title=f"T{i}") for i in range(10)]
        results = recommend_songs({"genre": "pop"}, songs, k=3)
        assert len(results) == 3

    def test_higher_score_ranks_first(self):
        match = _song(id="1", title="Match", genre="pop", mood="happy")
        partial = _song(id="2", title="Partial", genre="pop", mood="sad")
        none = _song(id="3", title="None", genre="rock", mood="sad")
        results = recommend_songs(
            {"genre": "pop", "mood": "happy"}, [none, partial, match], k=3
        )
        assert results[0][0].title == "Match"
        assert results[1][0].title == "Partial"
        assert results[2][0].title == "None"

    def test_tie_broken_by_danceability(self):
        a = _song(id="1", title="LowDance", genre="pop", danceability=0.2)
        b = _song(id="2", title="HighDance", genre="pop", danceability=0.9)
        results = recommend_songs({"genre": "pop"}, [a, b], k=2)
        assert results[0][0].title == "HighDance"
        assert results[1][0].title == "LowDance"

    def test_reason_string_present_for_match(self):
        song = _song(genre="pop")
        results = recommend_songs({"genre": "pop"}, [song], k=1)
        assert "genre match" in results[0][2]

    def test_reason_is_no_match_when_score_is_zero(self):
        song = _song(genre="rock", mood="intense", artist="Other")
        results = recommend_songs(
            {"genre": "pop", "mood": "happy", "artist": "Me"}, [song], k=1
        )
        assert results[0][2] == "No match"

    def test_logs_warning_when_no_match(self, caplog):
        song = _song(genre="rock", mood="intense", artist="Other")
        with caplog.at_level(logging.WARNING, logger="src.rule_based_recommender"):
            recommend_songs({"genre": "pop"}, [song], k=1)
        assert any("no good match" in r.message.lower() for r in caplog.records)
