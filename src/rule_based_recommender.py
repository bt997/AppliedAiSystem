"""
Rule-based music recommender.

Deterministic, API-free baseline. Scores songs by exact-match against the
user's preferred genre (+2), mood (+3), and artist (+5), with danceability
as the tiebreaker. Used as a fallback when the RAG path is unavailable
(no API key, quota exhausted, network issues).
"""

import logging

from src.models import Song

logger = logging.getLogger(__name__)


def score_song(user_prefs: dict, song: Song) -> tuple[float, list[str]]:
    """Return (score, reasons) for one song against the user's preferences."""
    score = 0.0
    reasons: list[str] = []

    if user_prefs.get("genre", "").lower() == song.genre.lower():
        score += 2
        reasons.append("genre match (+2)")

    if user_prefs.get("mood", "").lower() == song.mood.lower():
        score += 3
        reasons.append("mood match (+3)")

    if user_prefs.get("artist", "").lower() == song.artist.lower():
        score += 5
        reasons.append("artist match (+5)")

    return score, reasons


def recommend_songs(
    user_prefs: dict,
    songs: list[Song],
    k: int = 5,
) -> list[tuple[Song, float, str]]:
    """Rank songs against user_prefs and return the top k as (song, score, reason)."""
    scored = [(song, *score_song(user_prefs, song)) for song in songs]
    sorted_songs = sorted(
        scored,
        key=lambda x: (x[1], x[0].danceability),
        reverse=True,
    )

    top_score = sorted_songs[0][1]
    if top_score == 0:
        logger.warning("No good match found for prefs: %s", user_prefs)

    ties = [s for s in sorted_songs if s[1] == top_score]
    if len(ties) > 1:
        logger.info(
            "Tie broken by danceability among %d songs at score %s",
            len(ties), top_score,
        )

    top = sorted_songs[:k]
    return [
        (song, score, ", ".join(reasons) if reasons else "No match")
        for song, score, reasons in top
    ]
