from __future__ import annotations

from dataclasses import replace
import re

from .models import Chunk, Document, Element, stable_id, token_count


def split_text(text: str, budget: int) -> list[str]:
    """Lossless sentence/whitespace splitting with a hard fallback for giant tokens."""
    if token_count(text) <= budget:
        return [text]
    parts, start = [], 0
    while start < len(text):
        lo, hi = start + 1, len(text)
        while lo <= hi:
            mid = (lo + hi) // 2
            if token_count(text[start:mid]) <= budget:
                lo = mid + 1
            else:
                hi = mid - 1
        end = max(start + 1, hi)
        if end < len(text):
            segment = text[start:end]
            boundaries = list(re.finditer(r"[.!?]\s+|\n\s*\n", segment))
            if boundaries and boundaries[-1].end() > len(segment) // 3:
                end = start + boundaries[-1].end()
            else:
                spaces = list(re.finditer(r"\s+", segment))
                if spaces and spaces[-1].end() > len(segment) // 3:
                    end = start + spaces[-1].end()
        parts.append(text[start:end])
        start = end
    return parts


class StructuralChunker:
    def __init__(self, max_tokens: int = 384, min_tokens: int = 80,
                 embedder=None, semantic_threshold: float = .35):
        if not 32 <= max_tokens or not 0 <= min_tokens <= max_tokens:
            raise ValueError("Require max_tokens >= 32 and 0 <= min_tokens <= max_tokens")
        if not -1 <= semantic_threshold <= 1:
            raise ValueError("semantic_threshold must be in [-1, 1]")
        self.max_tokens, self.min_tokens = max_tokens, min_tokens
        self.embedder, self.semantic_threshold = embedder, semantic_threshold

    def chunk(self, doc: Document) -> list[Chunk]:
        chunks, pending = [], []
        active = [e for e in doc.elements if not e.excluded and e.text.strip()]
        vectors = {}
        if self.embedder:
            prose = [e for e in active if e.kind in {"paragraph", "list_item"}]
            if prose:
                vectors = dict(zip((e.id for e in prose), self.embedder.encode_documents([e.text for e in prose])))

        def emit(items: list[Element], text: str, kind: str, table_id="", row_range=None, boxes=None):
            first = items[0]
            citations, seen = [], set()
            for i, e in enumerate(items):
                bbox = boxes[i] if boxes else e.bbox
                key = (e.page, tuple(bbox))
                if key not in seen:
                    citations.append({"source": doc.source, "page": e.page, "bbox": bbox,
                                      "element_id": e.id, "extraction": e.extraction,
                                      "confidence": e.confidence})
                    seen.add(key)
            context = " > ".join([doc.title] + first.section_path)
            chunks.append(Chunk(stable_id(doc.id, first.section_id, len(chunks), text), doc.id,
                                first.section_id, first.section_path, kind, text, context, citations,
                                list(dict.fromkeys(e.id for e in items)), token_count(text),
                                table_id=table_id, row_range=row_range or [],
                                metadata={"title": doc.title, "category": doc.metadata.get("category", ""),
                                          "source_url": doc.metadata.get("download_url", ""),
                                          "chunking": "structure+embedding" if self.embedder else "structure",
                                          "needs_review": any(doc.quality[e.page-1]["warnings"] and
                                                              any(w.startswith(("needs_ocr", "ocr_", "low_", "replacement_", "visual_", "table_detection_failed"))
                                                                  for w in doc.quality[e.page-1]["warnings"])
                                                              for e in items)}))

        def flush():
            if pending:
                emit(pending, "\n\n".join(e.text for e in pending), "text")
                pending.clear()

        for e in active:
            if pending and pending[0].section_id != e.section_id:
                flush()
            if e.kind == "table":
                flush()
                header = " | ".join(e.header)
                # Keep header overhead bounded, preserving the full header in parsed artifacts.
                large_header = token_count(header) > max(8, self.max_tokens // 4)
                header_part = " | ".join(f"Col {i+1}" for i in range(len(e.header))) if large_header else header
                header_part = split_text(header_part, max(8, self.max_tokens // 4))[0]
                prefix = header_part + "\n" if header else ""
                start = 1 if e.rows and e.rows[0] == e.header and not large_header else 0
                if len(e.rows) == 1 and e.rows[0] == e.header:
                    # Forms can contain a header-only table; retain it as searchable content.
                    start, prefix = 0, ""
                row_texts, row_indices = [], []
                def emit_rows():
                    if row_texts:
                        emit([e] * len(row_indices), prefix + "\n".join(row_texts), "table", e.table_id,
                             [row_indices[0] + 1, row_indices[-1] + 1],
                             [e.row_boxes[r] if r < len(e.row_boxes) else e.bbox for r in row_indices])
                        row_texts.clear()
                        row_indices.clear()
                for r in range(start, len(e.rows)):
                    row = " | ".join(e.rows[r])
                    if row_texts and token_count(prefix + "\n".join(row_texts + [row])) > self.max_tokens:
                        emit_rows()
                    if token_count(prefix + row) > self.max_tokens:
                        emit_rows()
                        for part in split_text(row, self.max_tokens - token_count(prefix) - 2):
                            emit([e], prefix + part, "table", e.table_id, [r+1, r+1],
                                 [e.row_boxes[r] if r < len(e.row_boxes) else e.bbox])
                            chunks[-1].metadata["row_fragment"] = True
                    else:
                        row_texts.append(row)
                        row_indices.append(r)
                emit_rows()
                continue
            for part in split_text(e.text, self.max_tokens):
                fragment = replace(e, text=part)
                if pending:
                    combined = "\n\n".join(x.text for x in pending + [fragment])
                    topic_break = (pending[-1].id != e.id and pending[-1].id in vectors and e.id in vectors
                                   and token_count("\n\n".join(x.text for x in pending)) >= self.min_tokens
                                   and float(vectors[pending[-1].id] @ vectors[e.id]) < self.semantic_threshold)
                    if token_count(combined) > self.max_tokens or topic_break:
                        flush()
                pending.append(fragment)
        flush()
        for i, chunk in enumerate(chunks):
            chunk.previous_id = chunks[i-1].id if i else None
            chunk.next_id = chunks[i+1].id if i+1 < len(chunks) else None
        return chunks
