"""Retrieves songs from a Chroma collection by semantic similarity."""

from dataclasses import dataclass


@dataclass
class RetrievalResult:
    documents: list[str]
    confidences: list[float]

    @property
    def avg_confidence(self) -> float:
        return round(sum(self.confidences) / len(self.confidences), 3)

    @property
    def top_confidence(self) -> float:
        return self.confidences[0]


class SongRetriever:
    """Queries an indexed Chroma collection and returns documents + confidences."""

    def __init__(self, collection):
        self._collection = collection

    def retrieve(self, query: str, k: int = 5) -> RetrievalResult:
        results = self._collection.query(query_texts=[query], n_results=k)
        documents = results["documents"][0]
        distances = results["distances"][0]
        confidences = [round(1 - d, 3) for d in distances]
        return RetrievalResult(documents=documents, confidences=confidences)
