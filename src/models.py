"""Shared data model for the music recommender system."""

import csv
from dataclasses import dataclass


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

    def to_document(self) -> str:
        """Rich prose description used for embedding and LLM context."""
        energy_label = (
            "high-energy" if self.energy > 0.7
            else "medium-energy" if self.energy > 0.4
            else "low-energy"
        )
        texture = "acoustic" if self.acousticness > 0.5 else "electronic"
        dance_label = (
            "very danceable" if self.danceability > 0.75
            else "moderately danceable"
        )

        return (
            f'"{self.title}" by {self.artist}. '
            f"Genre: {self.genre}. Mood: {self.mood}. "
            f"{energy_label}, {texture}, {dance_label} "
            f"at {self.tempo_bpm} BPM."
        )


def load_songs(csv_path: str) -> list[Song]:
    """Read a CSV catalog and return Song objects."""
    songs: list[Song] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
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
