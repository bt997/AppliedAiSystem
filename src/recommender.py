from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

@dataclass
class Song:
    """
    Represents a song and its attributes.
    Required by tests/test_recommender.py
    """
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float

@dataclass
class UserProfile:
    """
    Represents a user's taste preferences.
    Required by tests/test_recommender.py
    """
    favorite_genre: str
    favorite_mood: str
    target_energy: float
    likes_acoustic: bool

class Recommender:
    """
    OOP implementation of the recommendation logic.
    Required by tests/test_recommender.py
    """
    def __init__(self, songs: List[Song]):
        self.songs = songs

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        # TODO: Implement recommendation logic
        return self.songs[:k]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        # TODO: Implement explanation logic
        return "Explanation placeholder"

def load_songs(csv_path: str) -> List[Dict]:
    """Reads a CSV file of songs and returns a list of dicts with typed fields."""
    import csv
    songs = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["id"] = int(row["id"])
            row["energy"] = float(row["energy"])
            row["tempo_bpm"] = float(row["tempo_bpm"])
            row["valence"] = float(row["valence"])
            row["danceability"] = float(row["danceability"])
            row["acousticness"] = float(row["acousticness"])
            songs.append(row)
    print(f"Loaded songs: {len(songs)}")
    return songs

def score_song(user_prefs: Dict, song: Dict) -> Tuple[float, List[str]]:
    """Returns a (score, reasons) tuple based on genre (+2), mood (+3), and artist (+5) matches."""
    score = 0.0
    reasons = []

    if user_prefs.get("genre", "").lower() == song["genre"].lower():
        score += 2
        reasons.append("genre match (+2)")

    if user_prefs.get("mood", "").lower() == song["mood"].lower():
        score += 3
        reasons.append("mood match (+3)")

    if user_prefs.get("artist", "").lower() == song["artist"].lower():
        score += 5
        reasons.append("artist match (+5)")

    return score, reasons


def recommend_songs(user_prefs: Dict, songs: List[Dict], k: int = 5) -> List[Tuple[Dict, float, str]]:
    """Scores and ranks all songs against user preferences, returning the top k results."""
    scored = [
        (song, *score_song(user_prefs, song))
        for song in songs
    ]

    sorted_songs = sorted(scored, key=lambda x: (x[1], x[0]["danceability"]), reverse=True)

    if sorted_songs[0][1] == 0:
        print("Warning: No good match found.")

    top_score = sorted_songs[0][1]
    ties = [s for s in sorted_songs if s[1] == top_score]
    if len(ties) > 1:
        print(f"Tie broken by danceability among {len(ties)} songs with score {top_score}.")

    top = sorted_songs[:k]
    return [(song, score, ", ".join(reasons) if reasons else "No match") for song, score, reasons in top]
