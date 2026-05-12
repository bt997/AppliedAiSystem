"""Generates natural-language recommendations from retrieved songs via Gemini."""

import logging

logger = logging.getLogger(__name__)


_PROMPT_TEMPLATE = (
    "You are a music recommendation assistant.\n\n"
    "User request: {query}\n\n"
    "The following songs were retrieved from the catalog as the "
    "closest semantic matches to the user's request:\n"
    "{catalog_context}\n\n"
    "Recommend the most fitting songs from this list. For each one, "
    "explain specifically why it suits the user's request."
)


class RecommendationGenerator:
    """Wraps a Gemini client and produces a recommendation string."""

    def __init__(self, client, model: str = "gemini-2.0-flash-lite"):
        self._client = client
        self._model = model

    def generate(self, query: str, retrieved_docs: list[str]) -> tuple[str, bool]:
        """
        Returns (text, generation_ok).

        On Gemini failure, returns a fallback string built from the retrieved
        documents so the caller still has something useful to show.
        """
        catalog_context = "\n".join(f"- {doc}" for doc in retrieved_docs)
        prompt = _PROMPT_TEMPLATE.format(
            query=query, catalog_context=catalog_context
        )

        try:
            response = self._client.models.generate_content(
                model=self._model, contents=prompt,
            )
            return response.text, True
        except Exception as e:
            reason = getattr(e, "message", None) or str(e).split(".")[0]
            logger.warning("Gemini unavailable: %s", reason)
            fallback = (
                "Generation unavailable — here are the top retrieved matches:\n\n"
                + catalog_context
            )
            return fallback, False
