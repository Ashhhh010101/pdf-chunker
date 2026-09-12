"""Structure-preserving PDF retrieval. All page numbers are one-based."""

from .parser import PDFParser
from .chunker import StructuralChunker
from .index import SearchIndex

__all__ = ["PDFParser", "StructuralChunker", "SearchIndex"]
