# Structural PDF parser, chunker and indexer

A local Python pipeline for arbitrary PDF files and directories. It emits structured document JSON, citation-bearing chunk JSONL, a persistent SQLite search index, and a per-page quality report. Source PDFs are never modified. No document content is sent to an LLM service.

## Install and run

Python 3.10 or newer:

```powershell
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[semantic,ocr,test]"

# Process a directory of PDFs (recursive) and search it.
.venv/Scripts/python -m semantic_chunker ingest C:/path/to/pdf-folder --output output/corpus --strict
.venv/Scripts/python -m semantic_chunker search "your question" --index output/corpus/index.sqlite

# Process one PDF.
.venv/Scripts/python -m semantic_chunker ingest C:/path/to/document.pdf --output output/single-pdf
.venv/Scripts/python -m semantic_chunker search "your question" --index output/single-pdf/index.sqlite --expand

# Structure + embedding topic boundaries, with hybrid retrieval.
.venv/Scripts/python -m semantic_chunker ingest C:/path/to/pdf-folder --output output/hybrid --embedding-model sentence-transformers/all-MiniLM-L6-v2 --semantic-boundaries --strict
.venv/Scripts/python -m semantic_chunker search "your question" --index output/hybrid/index.sqlite --mode hybrid --expand

# Optional second-stage reranking (downloads another local model on first use).
.venv/Scripts/python -m semantic_chunker search "your question" --index output/hybrid/index.sqlite --mode hybrid --reranker cross-encoder/ms-marco-MiniLM-L-6-v2
.venv/Scripts/python -m pytest -q

# Reuse parsed pages/OCR when experimenting with models or chunk thresholds.
.venv/Scripts/python -m semantic_chunker reindex output/corpus --output output/hybrid --embedding-model sentence-transformers/all-MiniLM-L6-v2 --semantic-boundaries --refresh-structure
.venv/Scripts/python scripts/audit_corpus.py output/hybrid --render
```

Base installation `pip install -e .` needs only PyMuPDF and NumPy. OCR and semantic models are optional extras. Models download on first use; pass `--model-cache` to choose the cache, or use an existing local model directory. RapidOCR includes its OCR model weights. Tesseract requires a separately installed executable/language data; `--ocr-language` selects its languages. The bundled RapidOCR fallback supports English/Chinese and refuses other requested languages rather than silently substituting them.

## How extraction and chunking work

1. Extract text geometry, font sizes, bold spans, native ruled tables, and PDF bookmarks. OCR scanned or severely damaged text pages when enabled. Recover ruled scan tables from raster grid lines and OCR coordinates.
2. Infer reading order using whitespace cuts. Suppress repeated margin boilerplate and printed contents pages from retrieval while retaining them in document JSON.
3. Reconstruct sections using numbering, bookmarks, typography, and explicit section/annexure labels. Clause numbering repairs flattened bookmark levels. Retain the parent tree and full heading path on every content element and chunk.
4. Pack paragraphs within the same section, preserving sentences where possible. Optional normalized embedding similarity adds topic boundaries inside a section after the minimum size. Section/table boundaries always take precedence.
5. Split tables by rows, repeating identified column headers. Large rows split only as a last resort, with `row_fragment=true`, table identity, physical row range, and row coordinates. First data rows are retained when there is no credible header. Large headers are kept as data rather than truncated away.
6. Index text and heading context in SQLite FTS5. Optional dense vectors use local Sentence Transformers; token windows are pooled so overlength input is not silently truncated. Hybrid search combines lexical and dense ranks with reciprocal rank fusion. Optional cross-encoder scores reorder the candidates.

The default 384-token chunk limit is a conservative byte/word estimate, **not a guarantee for a particular LLM tokenizer**. `token_count` covers chunk body text; context prefixes and expanded results consume additional tokens. The embedding adapter handles its own actual tokenizer limits. No overlap is copied into each chunk; linked neighbors allow bounded context expansion within the same section. A split sentence/row cites the original containing element/row box rather than pretending to have word-level geometry.

## Outputs and integration

The selected output directory contains:

| File | Purpose |
| --- | --- |
| `documents/<document-id>.json` | Elements, bounding boxes, raw table cells, section tree, manifest metadata and per-page quality |
| `chunks/<document-id>.jsonl` | Chunk text, heading path, source coordinates, neighbor IDs, table/row metadata |
| `index.sqlite` | Transactional documents, sections, chunks, full-text index and optional dense vectors |
| `ingestion_report.json` | Per-document counts, page warnings, unread pages and failures |

During ingestion/reindexing, the CLI prints timing and LLM-oriented sizing telemetry after every PDF and a final total: elapsed seconds, chunks per second, estimated body-token total and average/minimum/maximum chunk size, plus estimated context tokens including the hierarchy prefix. The same metrics are stored under each document's `processing` field in the report. Token values are conservative estimates; use the target model tokenizer for final prompt budgeting.

```python
from semantic_chunker import PDFParser, StructuralChunker, SearchIndex

doc = PDFParser(ocr="auto").parse("C:/path/to/document.pdf")
chunks = StructuralChunker(max_tokens=384).chunk(doc)
with SearchIndex("output/example.sqlite") as index:
    index.upsert(doc, chunks)
    hits = index.search("your question", mode="lexical", k=5)
    for hit in hits:
        print(hit["context"], hit["text"], hit["citations"])
        context = index.expand(hit["id"], max_tokens=1600)
```

For a hybrid API index, create `SentenceEmbedder` from `semantic_chunker.embeddings`, and pass that same model to `SearchIndex` on both ingestion and search. Pass it to `StructuralChunker(embedder=...)` to enable topic boundaries. Use a separate index when changing the embedding model. Mixed vector models or partial lexical/dense indexing are rejected.

`reindex --refresh-structure` reapplies the current heading rules to saved geometry. It reads page sizes/bookmarks from the original PDFs and verifies their SHA-256; it does not repeat text extraction or OCR. Without this option, reindexing only needs the saved document JSON.

Document identity is based on PDF content SHA-256. Chunk IDs are deterministic for the same document/configuration. Re-ingestion atomically replaces the existing content at the same path; identical content at another path is deduplicated and its citation path updated. Indexing a directory does not delete documents whose files were removed from the directory. Use `delete_document(id)` explicitly. JSON exports for obsolete IDs are historical artifacts and are not queried by the index. File export and the SQLite transaction are separate; rerun ingestion if export fails after indexing.

Search supports `--category`, `--document-id`, `-k`, and `--expand`. Results contain one-based PDF page numbers and bounding boxes in PDF points, together with local source paths and the original download URL when provided by the manifest. Scores are rankings, not confidence probabilities. Dense search can return irrelevant nearest neighbors; the consuming LLM must verify the cited text and abstain when it does not answer the question.

## Accuracy and known limits

“Highest accuracy” needs an independently labelled retrieval benchmark. This implementation provides preservation checks and page-labelled evaluation; it does not claim an unmeasured accuracy percentage.

```powershell
.venv/Scripts/python -m semantic_chunker evaluate tests/retrieval_cases.jsonl --index output/corpus/index.sqlite --output output/evaluation.json -k 5
```

Each evaluation line has `query` and `relevant: [{"source": "filename.pdf", "pages": [1]}]`; an optional `category` restricts retrieval. Cases may also contain a human-written `model_answer` and `answer_evidence`. Each `answer_evidence` item is a required phrase, or a list of acceptable alternative phrases for one fact. The evaluator reports page-retrieval metrics, per-query retrieval latency (plus mean/p50/p95), and how much required answer evidence occurs anywhere in the retrieved top-k chunks. Latency excludes answer scoring and report serialization. Answer evidence measures whether retrieval supplied the facts needed to produce the model answer; it does not judge generated prose or entailment.

```json
{"query":"What is the warranty?","model_answer":"The minimum warranty is three years.","answer_evidence":[["three years","3 years"]],"relevant":[{"source":"example.pdf","pages":[5]}]}
```

Create evaluation cases from your own PDFs and real user questions before choosing a model or tuning thresholds. Keep answers independently reviewed, concise, and limited to facts on the labelled pages.

`tests/tender_compliance_cases.jsonl` contains 100 tender-compliance questions grouped into ten verified intents. Each fact is expressed ten ways to test robustness to procurement-analyst phrasing such as “extract,” “prepare the compliance-matrix row,” “mandatory,” and “does the bidder comply.” Regenerate it with `python tests/build_tender_compliance_cases.py` after editing the reviewed intent definitions.

For a compliance matrix generated from one known tender, add `--scope-to-relevant-document` during evaluation. This applies the same document filter that production should pass to `SearchIndex.search(document_id=...)`; omit it only when benchmarking discovery across an entire mixed corpus.

Hybrid retrieval can run BM25 and query embedding concurrently with `--parallel`; `--query-variants` additionally searches conservative instruction-stripped/acronym-expanded variants. Both are opt-in because model quality and CPU contention can make lexical retrieval faster and more accurate on a specialized corpus. Use `--candidates N` to bound fusion/reranking work and benchmark on deployment hardware. Evaluation details include retrieval, merge, rerank, and formatting stage timings.

OCR, headings, table boundaries and reading order are heuristic. Scan OCR does not recover original bold typography. Complex borderless/rotated/nested tables, merged scan cells, diagrams, handwriting, and mixed scripts need review or a stronger document-layout/OCR backend. Native borderless tables remain positioned text; the parser does not invent cell structure. Multi-page tables remain separate page-local table elements: section ancestry provides context, but uncertain joins are not asserted. Native PDF tables may have ambiguous first-row headers; inspect the exported cells for critical facts. Images occupying a substantial page region, low OCR confidence, damaged text and sparse pages are explicitly flagged. The parser retains OCR text and coordinates but does not validate extracted amounts or dates against a second OCR engine.

`--strict` exits nonzero for document failures or pages left needing OCR. Other review warnings remain in the report and are not a certification that every figure or cell was understood. Inspect them before production use. Exact dense scanning is simple and reproducible for small and medium corpora; use a vector database/ANN index for much larger corpora. The optional reranker has its own sequence limit and may truncate long context; keep chunks modest and evaluate it against your labels.

Implementation references: [PyMuPDF page extraction, OCR and tables](https://pymupdf.readthedocs.io/en/latest/page.html), [Sentence Transformers semantic search](https://www.sbert.net/examples/sentence_transformer/applications/semantic-search/README.html), and [retrieve and rerank](https://www.sbert.net/examples/sentence_transformer/applications/retrieve_rerank/README.html).
