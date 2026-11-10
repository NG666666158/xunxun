"""Home-page "猜你想问": a local question pool, sampled into four cards."""

from .service import SuggestionService
from .store import SuggestionStore

__all__ = ["SuggestionService", "SuggestionStore"]
