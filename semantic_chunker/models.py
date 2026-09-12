from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import re


def stable_id(*parts: object) -> str:
    return hashlib.sha256("\x00".join(map(str, parts)).encode()).hexdigest()[:24]


def token_count(text: str) -> int:
    """Conservative sizing estimate; actual embedding token limits are checked separately."""
    return max(len(re.findall(r"\w+|[^\w\s]", text)), (len(text.encode("utf-8")) + 2) // 3)


@dataclass
class Element:
    id: str
    page: int
    bbox: list[float]
    kind: str
    text: str
    font_size: float = 0.0
    bold: bool = False
    level: int = 0
    section_id: str = ""
    section_path: list[str] = field(default_factory=list)
    rows: list[list[str]] = field(default_factory=list)
    row_boxes: list[list[float]] = field(default_factory=list)
    header: list[str] = field(default_factory=list)
    table_id: str = ""
    extraction: str = "native"
    confidence: float = 1.0
    excluded: bool = False


@dataclass
class Section:
    id: str
    title: str
    level: int
    parent_id: str | None
    page: int
    path: list[str]


@dataclass
class Document:
    id: str
    source: str
    sha256: str
    title: str
    page_count: int
    elements: list[Element]
    sections: list[Section]
    quality: list[dict]
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Document":
        data = dict(data)
        data["elements"] = [Element(**e) for e in data["elements"]]
        data["sections"] = [Section(**s) for s in data["sections"]]
        return cls(**data)


@dataclass
class Chunk:
    id: str
    document_id: str
    section_id: str
    section_path: list[str]
    kind: str
    text: str
    context: str
    citations: list[dict]
    element_ids: list[str]
    token_count: int
    previous_id: str | None = None
    next_id: str | None = None
    table_id: str = ""
    row_range: list[int] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)
