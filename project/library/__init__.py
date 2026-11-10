"""Local file model: watched folders, file registry, background indexing and change tracking."""

from .registry import FILE_STATUSES, SEARCHABLE_STATUSES, FileRegistry
from .service import LibraryError, LibraryService

__all__ = ["FILE_STATUSES", "SEARCHABLE_STATUSES", "FileRegistry", "LibraryError", "LibraryService"]
