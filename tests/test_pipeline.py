from dataclasses import replace
import json

import fitz
import numpy as np
import pytest

from semantic_chunker import PDFParser, SearchIndex, StructuralChunker
from semantic_chunker.chunker import split_text
from semantic_chunker.models import Document, Element, Section, token_count
from semantic_chunker.parser import reading_order


def make_pdf(path):
    with fitz.open() as pdf:
        for pn in range(3):
            p = pdf.new_page()
            p.insert_text((50, 30), "Recurring tender header", fontsize=10)
            p.insert_text((50, 100), f"{pn+1}. REQUIREMENTS", fontname="hebo", fontsize=16)
            p.insert_text((50, 140), f"{pn+1}.1 Warranty", fontname="hebo", fontsize=12)
            p.insert_text((50, 175), "The supplier must provide five years of onsite warranty.", fontsize=11)
            p.insert_text((50, 800), f"Page {pn+1} of 3", fontsize=10)
        pdf.save(path)


def simple_doc(elements):
    return Document("doc", "source.pdf", "sha", "Tender", 1, elements,
                    [Section("s", "Warranty", 1, None, 1, ["Warranty"])],
                    [{"page": 1, "warnings": []}], {"category": "hardware"})


def test_pdf_hierarchy_citations_and_boilerplate(tmp_path):
    path = tmp_path / "test.pdf"
    make_pdf(path)
    doc = PDFParser(ocr="off").parse(path)
    assert any(s.path == ["1. REQUIREMENTS", "1.1 Warranty"] for s in doc.sections)
    chunks = StructuralChunker().chunk(doc)
    assert all("Recurring tender header" not in c.text and "Page 1 of" not in c.text for c in chunks)
    assert all(c.citations and c.section_id and c.token_count <= 384 for c in chunks)
    assert any("five years" in c.text and c.citations[0]["page"] == 2 for c in chunks)
    assert [c.id for c in chunks] == [c.id for c in StructuralChunker().chunk(PDFParser(ocr="off").parse(path))]
    refreshed = PDFParser(ocr="off").refresh_structure(doc)
    assert [c.id for c in chunks] == [c.id for c in StructuralChunker().chunk(refreshed)]


def test_lossless_splitting_and_boundaries():
    text = "Eligibility requires 12.50% security. " * 100 + "超" * 1200
    parts = split_text(text, 100)
    assert "".join(parts) == text
    assert all(token_count(p) <= 100 for p in parts)
    a = Element("a", 1, [0, 0, 100, 20], "paragraph", text, section_id="s", section_path=["Warranty"])
    b = replace(a, id="b", text="Separate section", section_id="other", section_path=["Other"])
    chunks = StructuralChunker(100).chunk(simple_doc([a, b]))
    assert all(c.token_count <= 100 for c in chunks)
    assert chunks[-1].element_ids == ["b"]
    assert all(c.section_id == "s" for c in chunks[:-1])


def test_table_headers_row_ranges_and_oversize():
    rows = [["Item", "Requirement"], ["GPU", "80GB memory"], ["Warranty", "onsite service " * 100]]
    e = Element("table", 1, [10, 20, 200, 250], "table", "table", section_id="s",
                rows=rows, header=rows[0], table_id="t", row_boxes=[[10, 20, 200, 40], [10, 40, 200, 60], [10, 60, 200, 250]])
    chunks = StructuralChunker(100).chunk(simple_doc([e]))
    assert chunks and all(c.kind == "table" and c.text.startswith("Item | Requirement") for c in chunks)
    assert all(c.token_count <= 100 for c in chunks)
    assert chunks[0].row_range == [2, 2]
    assert chunks[0].citations[0]["bbox"] == [10, 40, 200, 60]
    assert any(c.metadata.get("row_fragment") for c in chunks)


def test_native_table_extraction(tmp_path):
    path = tmp_path / "table.pdf"
    with fitz.open() as pdf:
        p = pdf.new_page()
        for x in [50, 200, 400]:
            p.draw_line((x, 100), (x, 190))
        for y in [100, 130, 160, 190]:
            p.draw_line((50, y), (400, y))
        for y, cells in [(120, ["Item", "Requirement"]), (150, ["GPU", "80GB"]), (180, ["Warranty", "5 years"])]:
            for x, text in zip([60, 210], cells):
                p.insert_text((x, y), text)
        pdf.save(path)
    doc = PDFParser(ocr="off").parse(path)
    tables = [e for e in doc.elements if e.kind == "table"]
    assert len(tables) == 1 and tables[0].rows[1] == ["GPU", "80GB"]
    assert not any(e.kind != "table" and "80GB" in e.text for e in doc.elements)


def test_reading_order_columns_and_heading():
    def e(id, box):
        return Element(id, 1, box, "paragraph", id)
    elements = [e("right2", [320, 160, 550, 185]), e("heading", [50, 30, 550, 50]),
                e("left2", [50, 150, 270, 175]), e("right1", [320, 80, 550, 150]),
                e("left1", [50, 80, 270, 140])]
    assert [x.id for x in reading_order(elements, 600)] == ["heading", "left1", "left2", "right1", "right2"]


class FakeEmbedder:
    name = "test-model"
    def encode_documents(self, texts):
        return np.array([[1, 0] if "warranty" in t.lower() else [0, 1] for t in texts], dtype=np.float32)
    def encode_query(self, query):
        return np.array([1, 0], dtype=np.float32)


def test_index_upsert_hybrid_filters_expansion_and_delete(tmp_path):
    e = Element("a", 1, [0, 0, 20, 20], "paragraph", "Five year warranty", section_id="s")
    doc = simple_doc([e])
    chunks = StructuralChunker().chunk(doc)
    with SearchIndex(tmp_path / "index.sqlite", FakeEmbedder()) as index:
        index.upsert(doc, chunks)
        index.upsert(doc, chunks)
        assert index.stats()["chunks"] == 1
        hit = index.search("onsite support", mode="hybrid")[0]
        assert hit["id"] == chunks[0].id and "dense" in hit["retrieval_ranks"]
        assert index.search("warranty", category="civil", mode="lexical") == []
        assert index.search('"warranty" OR * ()', mode="lexical")
        assert index.expand(hit["id"])["chunks"][0]["text"] == e.text
        index.delete_document(doc.id)
        assert index.search("warranty", mode="lexical") == []


def test_index_rejects_model_mixing_and_update_is_atomic(tmp_path):
    doc = simple_doc([Element("a", 1, [0, 0, 20, 20], "paragraph", "warranty", section_id="s")])
    chunks = StructuralChunker().chunk(doc)
    with SearchIndex(tmp_path / "index.sqlite") as index:
        index.upsert(doc, chunks)
        index.embedder = FakeEmbedder()
        with pytest.raises(ValueError, match="mix"):
            index.upsert(doc, chunks)
        index.embedder = None
        with pytest.raises(ValueError, match="requires"):
            index.search("warranty")
        with pytest.raises(Exception):
            index.upsert(doc, chunks + chunks)
        assert index.stats()["chunks"] == 1
        assert len(index.search("warranty", mode="lexical")) == 1


def test_scanned_page_is_flagged_not_silently_dropped(tmp_path):
    path = tmp_path / "scan.pdf"
    with fitz.open() as source:
        page = source.new_page()
        page.insert_text((50, 100), "Scanned warranty terms")
        pix = page.get_pixmap()
        with fitz.open() as pdf:
            p = pdf.new_page()
            p.insert_image(p.rect, pixmap=pix)
            pdf.save(path)
    doc = PDFParser(ocr="off").parse(path)
    assert "needs_ocr" in doc.quality[0]["warnings"]
    assert not StructuralChunker().chunk(doc)


def test_oversized_table_header_is_not_lost():
    header = ["Requirement " * 60, "Compliance"]
    e = Element("t", 1, [0, 0, 100, 100], "table", "", section_id="s",
                rows=[header, ["GPU", "Yes"]], header=header, table_id="t")
    e.text = " | ".join(header) + "\nGPU | Yes"
    chunks = StructuralChunker(100).chunk(simple_doc([e]))
    assert all(c.token_count <= 100 for c in chunks)
    assert sum(c.text.count("Requirement") for c in chunks) == 60


def test_header_only_table_is_retained():
    row = ["Item", "Description", "Amount"]
    e = Element("t", 1, [0, 0, 100, 20], "table", " | ".join(row), section_id="s",
                rows=[row], header=row, table_id="t")
    chunks = StructuralChunker().chunk(simple_doc([e]))
    assert len(chunks) == 1 and chunks[0].text == e.text
    assert chunks[0].row_range == [1, 1]


def test_semantic_boundary_is_within_section_only():
    a = Element("a", 1, [0, 0, 100, 20], "paragraph", "Warranty " * 35, section_id="s")
    b = replace(a, id="b", text="Network installation and cabling " * 10)
    doc = simple_doc([a, b])
    assert len(StructuralChunker(384).chunk(doc)) == 1
    chunks = StructuralChunker(384, embedder=FakeEmbedder()).chunk(doc)
    assert len(chunks) == 2
    assert all(c.metadata["chunking"] == "structure+embedding" for c in chunks)


def test_scan_grid_recovery_with_small_skew(tmp_path):
    pytest.importorskip("cv2")
    from semantic_chunker.ocr_tables import scan_tables
    with fitz.open() as pdf:
        p = pdf.new_page()
        for x in [50, 220, 500]:
            p.draw_line((x, 100 + x*.02), (x, 250 + x*.02))
        for y in [100, 150, 200, 250]:
            p.draw_line((50, y+1), (500, y+10))
        blocks = []
        for y, cells in [(130, ["Item", "Value"]), (180, ["EMD", "172048"]), (230, ["Duration", "244 days"])]:
            for x, text in zip([70, 240], cells):
                p.insert_text((x, y), text)
                blocks.append({"lines": [{"bbox": [x, y-10, x+80, y+2], "spans": [{"text": text}]}]})
        tables = scan_tables(p, blocks, "doc", "rapidocr")
        assert len(tables) == 1
        assert tables[0].rows[1] == ["EMD", "172048"]
        assert tables[0].rows[2] == ["Duration", "244 days"]


def test_reindex_does_not_resurrect_obsolete_exports(tmp_path):
    from semantic_chunker.cli import main, write_json
    source, target = tmp_path / "source", tmp_path / "target"
    doc = simple_doc([Element("e", 1, [0, 0, 20, 20], "paragraph", "Active warranty", section_id="s")])
    with SearchIndex(source / "index.sqlite") as index:
        index.upsert(doc, StructuralChunker().chunk(doc))
    write_json(source / "documents" / "doc.json", doc.to_dict())
    obsolete = doc.to_dict()
    obsolete["id"] = "obsolete"
    write_json(source / "documents" / "obsolete.json", obsolete)
    assert main(["reindex", str(source), "--output", str(target)]) == 0
    with SearchIndex(target / "index.sqlite") as index:
        assert index.stats()["documents"] == 1
        assert index.search("warranty", mode="lexical")[0]["document_id"] == "doc"


def test_reranker_scores_answer_beyond_first_window():
    from semantic_chunker.embeddings import CrossEncoderReranker
    class Tokenizer:
        def encode(self, text, **kwargs):
            return text.split()
        def decode(self, tokens):
            return " ".join(tokens)
    class Model:
        tokenizer = Tokenizer()
        max_length = 64
        def predict(self, pairs):
            assert all(len(q.split()) + len(t.split()) + 3 <= 64 for q, t in pairs)
            return np.array([10.0 if "answer" in t else -1.0 for _, t in pairs])
    reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
    reranker.model = Model()
    assert reranker.score("warranty", ["filler " * 200 + "answer", "irrelevant"] ) == [10.0, -1.0]


def test_answer_evidence_score_handles_alternatives_and_missing_facts():
    from semantic_chunker.cli import _answer_evidence_score
    case = {"answer_evidence": ["three years", ["OEM engineer", "manufacturer engineer"], "Delhi"]}
    results = [{"context": "Warranty", "text": "Minimum three years; service by an OEM engineer."}]
    score, matched = _answer_evidence_score(case, results)
    assert score == pytest.approx(2 / 3)
    assert matched == [True, True, False]
    assert _answer_evidence_score({}, results) == (None, [])


def test_answer_evidence_score_normalizes_case_punctuation_and_spacing():
    from semantic_chunker.cli import _answer_evidence_score
    case = {"answer_evidence": ["Rs. 4,05,000", "4 months"]}
    results = [{"context": "", "text": "RS 4 05 000\n4   MONTHS"}]
    assert _answer_evidence_score(case, results) == (1.0, [True, True])
