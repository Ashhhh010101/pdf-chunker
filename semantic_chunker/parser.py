from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
from pathlib import Path
import re
import unicodedata

import fitz

from .models import Document, Element, Section, stable_id


def clean(text: str) -> str:
    # Preserve numbers, punctuation and line boundaries; never silently dehyphenate IDs.
    return "\n".join(re.sub(r"[\t \u00a0]+", " ", line).strip()
                     for line in unicodedata.normalize("NFKC", text).splitlines()).strip()


def normalized(text: str) -> str:
    return re.sub(r"\W+", "", text).casefold()


def margin_key(text: str) -> str:
    return re.sub(r"\d+", "#", clean(text).casefold())


def reading_order(elements: list[Element], page_width: float) -> list[Element]:
    """Recursive XY cut: full-width blocks divide bands, vertical gaps divide columns."""
    if len(elements) < 2:
        return elements
    # A meaningful empty vertical strip separates independent text columns.
    spans = sorted((e.bbox[0], e.bbox[2]) for e in elements)
    end = spans[0][1]
    gaps = []
    for left, right in spans[1:]:
        if left - end > max(18, page_width * .035):
            gaps.append((left - end, (end + left) / 2))
        end = max(end, right)
    if gaps:
        _, cut = max(gaps)
        left = [e for e in elements if e.bbox[2] <= cut]
        right = [e for e in elements if e.bbox[0] >= cut]
        if left and right and len(left) + len(right) == len(elements):
            return reading_order(left, page_width) + reading_order(right, page_width)
    # Split at horizontal whitespace, enabling column cuts below spanning headings.
    spans = sorted((e.bbox[1], e.bbox[3]) for e in elements)
    end = spans[0][1]
    for top, bottom in spans[1:]:
        if top - end > 5:
            cut = (end + top) / 2
            above = [e for e in elements if e.bbox[3] <= cut]
            below = [e for e in elements if e.bbox[1] >= cut]
            if above and below and len(above) + len(below) == len(elements):
                return reading_order(above, page_width) + reading_order(below, page_width)
        end = max(end, bottom)
    return sorted(elements, key=lambda e: (round(e.bbox[1] / 3), e.bbox[0]))


class PDFParser:
    def __init__(self, ocr: str = "auto", ocr_language: str = "eng"):
        if ocr not in {"auto", "off", "tesseract", "rapidocr"}:
            raise ValueError("Unknown OCR mode")
        self.ocr = ocr
        self.ocr_language = ocr_language
        self._rapid = None

    def _ocr(self, page) -> tuple[list[dict], str, list[float]]:
        if self.ocr in {"auto", "tesseract"}:
            try:
                tp = page.get_textpage_ocr(language=self.ocr_language, dpi=200, full=True)
                return page.get_text("dict", textpage=tp)["blocks"], "tesseract", []
            except Exception:
                if self.ocr == "tesseract":
                    raise
        if self.ocr_language != "eng":
            raise RuntimeError("RapidOCR bundled model is English/Chinese; install Tesseract language data for other languages")
        from rapidocr_onnxruntime import RapidOCR
        import numpy as np
        if self._rapid is None:
            self._rapid = RapidOCR(intra_op_num_threads=2, inter_op_num_threads=2)
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
        result, _ = self._rapid(np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3))
        blocks, scores = [], []
        for box, text, score in result or []:
            xs, ys = [p[0] / 2 for p in box], [p[1] / 2 for p in box]
            bbox = [min(xs), min(ys), max(xs), max(ys)]
            blocks.append({"type": 0, "bbox": bbox, "lines": [{"bbox": bbox, "spans": [
                {"text": text, "bbox": bbox, "size": (bbox[3] - bbox[1]) * .75, "flags": 0}
            ]}], "confidence": float(score)})
            scores.append(float(score))
        return blocks, "rapidocr", scores

    def parse(self, path: str | Path, metadata: dict | None = None) -> Document:
        path = Path(path).resolve()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        doc_id = stable_id(digest)
        elements, quality = [], []
        page_sizes = {}
        bookmarks = defaultdict(dict)
        with fitz.open(path) as pdf:
            if pdf.needs_pass:
                raise ValueError(f"Encrypted PDF requires a password: {path}")
            title = (metadata or {}).get("title") or clean((pdf.metadata or {}).get("title", "")) or path.stem
            for level, heading, page in pdf.get_toc():
                if len(normalized(heading)) > 3:
                    bookmarks[page][normalized(heading)] = level
            for page in pdf:
                pn = page.number + 1
                page_sizes[pn] = (page.rect.width, page.rect.height)
                native = page.get_text("dict", flags=fitz.TEXTFLAGS_DICT & ~fitz.TEXT_PRESERVE_IMAGES)
                native_text = page.get_text()
                page_blocks, method, scores = native["blocks"], "native", []
                warnings = []
                sparse = len(native_text.strip()) < 80
                corrupt = native_text.count("\ufffd") / max(1, len(native_text)) > .03
                # Sparse scanned pages need OCR; sparse blank/digital pages are retained and flagged.
                has_image = bool(page.get_images())
                image_regions = [list(i["bbox"]) for i in page.get_image_info()
                                 if fitz.Rect(i["bbox"]).get_area() > page.rect.get_area() * .2]
                image_dominated = any(fitz.Rect(b).get_area() > page.rect.get_area() * .65 for b in image_regions)
                needs_ocr = has_image and (sparse or image_dominated and len(native_text.strip()) < 300)
                if (needs_ocr or corrupt) and self.ocr != "off":
                    try:
                        page_blocks, method, scores = self._ocr(page)
                    except Exception as exc:
                        warnings.append(f"ocr_failed: {type(exc).__name__}: {exc}")
                if needs_ocr and method == "native":
                    warnings.append("needs_ocr")
                elif image_regions and method == "native":
                    warnings.append("visual_content_requires_review")
                if corrupt:
                    warnings.append("replacement_characters_in_native_text")
                if scores and min(scores) < .8:
                    warnings.append("low_ocr_confidence")
                if method != "native":
                    warnings.append("ocr_layout_requires_review")
                tables = []
                if method == "native":
                    try:
                        for table in page.find_tables().tables:
                            rows = [[clean(c or "") for c in row] for row in table.extract()]
                            # A page border or a one-column form is not a relational table.
                            if not rows or table.col_count < 2:
                                continue
                            if sum(sum(bool(c) for c in row) >= 2 for row in rows) < min(2, len(rows)):
                                continue
                            bbox = list(table.bbox)
                            tid = stable_id(doc_id, pn, "table", bbox)
                            header = [clean(c or "") for c in table.header.names]
                            # Internal header is the first physical row; external header is separate.
                            if not table.header.external:
                                header = rows[0]
                            # PyMuPDF labels the first row a header even when it is data.
                            plausible_header = (all(len(c) < 120 for c in header) and
                                                any(re.search(r"\b(item|description|particulars?|qty|quantity|specifications?|requirements?|criteria|parameter|unit|amount|rate|details|s\.?\s?no)\b", c, re.I)
                                                    for c in header))
                            if not plausible_header:
                                header = [f"Column {i+1}" for i in range(table.col_count)]
                            tables.append(Element(tid, pn, bbox, "table", "\n".join(" | ".join(r) for r in rows),
                                                  rows=rows, row_boxes=[list(r.bbox) for r in table.rows],
                                                  header=header, table_id=tid))
                    except Exception as exc:
                        warnings.append(f"table_detection_failed: {type(exc).__name__}")
                else:
                    try:
                        from .ocr_tables import scan_tables
                        tables = scan_tables(page, page_blocks, doc_id, method)
                    except Exception as exc:
                        warnings.append(f"ocr_table_detection_failed: {type(exc).__name__}")
                for block in page_blocks:
                    if block.get("type") != 0:
                        continue
                    pending = []
                    def flush():
                        if not pending:
                            return
                        box = fitz.Rect(pending[0][1])
                        for _, b, _, _ in pending[1:]:
                            box |= fitz.Rect(b)
                        text = clean("\n".join(x[0] for x in pending))
                        if text:
                            size = sum(x[2] * len(x[0]) for x in pending) / max(1, sum(len(x[0]) for x in pending))
                            bold = sum(len(x[0]) for x in pending if x[3]) > len(text) * .6
                            elements.append(Element(stable_id(doc_id, pn, len(elements), text), pn, list(box),
                                                    "paragraph", text, size, bold, extraction=method,
                                                    confidence=float(block.get("confidence", 1.0))))
                        pending.clear()
                    for line in block.get("lines", []):
                        spans = line["spans"]
                        text = clean("".join(s["text"] for s in spans))
                        if not text:
                            continue
                        box = line["bbox"]
                        center = fitz.Point((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
                        if any(center in fitz.Rect(t.bbox) for t in tables):
                            continue
                        size = max(s["size"] for s in spans)
                        bold = sum(len(s["text"]) for s in spans if s["flags"] & 16) >= len(text) * .6
                        boundary = bool(re.match(r"^(?:\d+(?:\.\d+)*[.)]?\s|[•\uf0b7\ufffd]|\([a-zivx]+\))", text, re.I))
                        if pending and (boundary or bold != pending[-1][3] or abs(size - pending[-1][2]) > 1):
                            flush()
                        pending.append((text, box, size, bold))
                    flush()
                elements.extend(tables)
                extracted = sum(len(e.text) for e in elements if e.page == pn)
                if extracted < 80:
                    warnings.append("sparse_page")
                quality.append({"page": pn, "method": method, "characters": extracted,
                                "tables": len(tables), "warnings": warnings,
                                "image_regions": image_regions,
                                "ocr_mean_confidence": sum(scores) / len(scores) if scores else None})
            page_count = len(pdf)
        ordered, sections = self._structure(doc_id, title, page_count, elements, page_sizes, bookmarks, quality)
        return Document(doc_id, str(path), digest, title, page_count, ordered, sections, quality, metadata or {})

    def refresh_structure(self, doc: Document) -> Document:
        """Reapply heading rules to stored geometry without repeating extraction or OCR."""
        bookmarks = defaultdict(dict)
        with fitz.open(doc.source) as pdf:
            if hashlib.sha256(Path(doc.source).read_bytes()).hexdigest() != doc.sha256:
                raise ValueError("Source PDF changed; run ingest again instead of refreshing cached geometry")
            page_sizes = {p.number+1: (p.rect.width, p.rect.height) for p in pdf}
            for level, heading, pn in pdf.get_toc():
                if len(normalized(heading)) > 3:
                    bookmarks[pn][normalized(heading)] = level
        for e in doc.elements:
            if e.kind != "table":
                e.kind = "paragraph"
            e.excluded, e.level, e.section_id, e.section_path = False, 0, "", []
        for q in doc.quality:
            q["warnings"] = [w for w in q["warnings"] if w != "table_of_contents_excluded"]
        doc.elements, doc.sections = self._structure(doc.id, doc.title, doc.page_count, doc.elements,
                                                     page_sizes, bookmarks, doc.quality)
        return doc

    def _structure(self, doc_id, title, page_count, elements, page_sizes, bookmarks, quality):
        # Infer boilerplate only from repeated text in the same margin, across distinct pages.
        margins = defaultdict(set)
        for e in elements:
            height = page_sizes[e.page][1]
            if e.bbox[3] < height * .09 or e.bbox[1] > height * .91:
                side = "top" if e.bbox[3] < height * .09 else "bottom"
                margins[(side, margin_key(e.text))].add(e.page)
        sizes = Counter()
        page_fonts = defaultdict(Counter)
        for e in elements:
            if e.kind == "paragraph":
                sizes[round(e.font_size, 1)] += len(e.text)
                page_fonts[e.page][round(e.font_size, 1)] += len(e.text)
        body_size = sizes.most_common(1)[0][0] if sizes else 11
        big_sizes = sorted([s for s, n in sizes.items() if s > body_size * 1.08 and n > 30], reverse=True)
        for e in elements:
            height = page_sizes[e.page][1]
            side = "top" if e.bbox[3] < height * .09 else "bottom" if e.bbox[1] > height * .91 else None
            if side and (len(margins[(side, margin_key(e.text))]) >= max(3, page_count * .25)
                         or re.fullmatch(r"(?:page\s*)?\d+(?:\s*(?:of|/)\s*\d+)?", e.text, re.I)):
                e.kind, e.excluded = "header" if side == "top" else "footer", True
                continue
            if e.kind == "table":
                continue
            short = len(e.text) < 180 and len(e.text.splitlines()) <= 3
            number = re.match(r"^(\d+(?:\.\d+)*)(?:\.|\))?\s+([A-Za-z].*)", e.text, re.S)
            explicit = re.match(r"^(section|chapter|part|annexure|annex|appendix|schedule)\s*[-–:]?\s*([\dIVXLC]+|[A-Z])(?:\b|:)", e.text, re.I)
            bookmark = bookmarks[e.page].get(normalized(e.text))
            local_size = page_fonts[e.page].most_common(1)[0][0] if page_fonts[e.page] else body_size
            # OCR box height is an unreliable proxy for typography; require a much larger difference.
            size_ratio = 1.08 if e.extraction == "native" else 1.35
            styled = e.bold or e.font_size > max(body_size, local_size) * size_ratio
            bullet = bool(re.match(r"^(?:[•\uf0b7\ufffd]|\([a-zivx]+\))", e.text, re.I))
            # Numbered prose stays a clause/list, unless there is title-like evidence.
            title_like = (short and 2 <= len(e.text.split()) <= 20 and not bullet
                          and not re.search(r"[.;,]$", e.text) and (styled or e.text.isupper())
                          and not re.match(r"^(?:yours |\(signature|\(to be given)", e.text, re.I))
            if short and (bookmark or explicit or title_like):
                e.kind = "heading"
                if number:
                    e.level = min(6, number[1].count(".") + 1)
                elif explicit:
                    e.level = 1
                elif bookmark:
                    e.level = min(6, bookmark)
                elif e.text.isupper() and len(e.text.split()) >= 3:
                    e.level = 1
                else:
                    e.level = min(6, 1 + big_sizes.index(round(e.font_size, 1))) if round(e.font_size, 1) in big_sizes else 2
            elif number or re.match(r"^(?:[•\uf0b7\ufffd]|\([a-zivx]+\))", e.text, re.I):
                e.kind = "list_item"
        ordered = []
        for pn, (width, _) in page_sizes.items():
            page_elements = [e for e in elements if e.page == pn]
            # Printed TOCs are navigational content, retained in parse but not indexed.
            dotted = sum(bool(re.search(r"\.{3,}\s*\d+", e.text)) for e in page_elements)
            toc_title = any(re.fullmatch(r"(?:table of )?contents", e.text, re.I) for e in page_elements)
            if dotted >= 3 or toc_title:
                for e in page_elements:
                    e.excluded = True
                quality[pn - 1]["warnings"].append("table_of_contents_excluded")
            ordered.extend(reading_order(page_elements, width))
        root = Section(stable_id(doc_id, "root"), title, 0, None, 1, [])
        sections, stack = [root], [root]
        for e in ordered:
            if e.excluded:
                continue
            if e.kind == "heading":
                while len(stack) > 1 and stack[-1].level >= e.level:
                    stack.pop()
                section = Section(stable_id(doc_id, e.id, "section"), e.text, e.level, stack[-1].id,
                                  e.page, stack[-1].path + [e.text])
                sections.append(section)
                stack.append(section)
            e.section_id, e.section_path = stack[-1].id, list(stack[-1].path)
        return ordered, sections
