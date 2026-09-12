from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import sqlite3
from pathlib import Path
import sys
import time

from .chunker import StructuralChunker
from .index import SearchIndex
from .parser import PDFParser
from .models import Document


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def ingest(args):
    source = Path(args.source).resolve()
    files = [source] if source.is_file() else sorted(source.rglob("*.pdf"))
    if not files:
        raise ValueError(f"No PDFs found in {source}")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    manifest_root = source if source.is_dir() else source.parent
    manifest = next((root / "manifest.json" for root in [manifest_root, *list(manifest_root.parents)[:2]]
                     if (root / "manifest.json").exists()), manifest_root / "manifest.json")
    metadata = {}
    if manifest.exists():
        metadata = {str((manifest.parent / m["filename"]).resolve()): m for m in json.loads(manifest.read_text(encoding="utf-8"))}
    embedder = None
    if args.embedding_model:
        from .embeddings import SentenceEmbedder
        embedder = SentenceEmbedder(args.embedding_model, args.model_cache)
    if args.semantic_boundaries and not embedder:
        raise ValueError("--semantic-boundaries requires --embedding-model")
    parser = PDFParser(args.ocr, args.ocr_language)
    chunker = StructuralChunker(args.max_tokens, min(80, args.max_tokens),
                                embedder if args.semantic_boundaries else None, args.semantic_threshold)
    report = {"documents": [], "failures": [], "configuration": vars(args).copy()}
    report["configuration"].pop("func", None)
    started = time.time()
    with SearchIndex(output / "index.sqlite", embedder) as index:
        for i, path in enumerate(files, 1):
            print(f"[{i}/{len(files)}] {path.name}", file=sys.stderr, flush=True)
            try:
                doc = parser.parse(path, metadata.get(str(path), {"category": path.parent.name}))
                chunks = chunker.chunk(doc)
                index.upsert(doc, chunks)
                write_json(output / "documents" / f"{doc.id}.json", doc.to_dict())
                chunk_path = output / "chunks" / f"{doc.id}.jsonl"
                chunk_path.parent.mkdir(parents=True, exist_ok=True)
                temp = chunk_path.with_suffix(".jsonl.tmp")
                temp.write_text("".join(json.dumps(asdict(c), ensure_ascii=False) + "\n" for c in chunks), encoding="utf-8")
                temp.replace(chunk_path)
                flagged = [q for q in doc.quality if q["warnings"]]
                entry = {"source": str(path), "id": doc.id, "pages": doc.page_count,
                         "elements": len(doc.elements), "sections": len(doc.sections), "chunks": len(chunks),
                         "tables": sum(e.kind == "table" for e in doc.elements),
                         "ocr_pages": sum(q["method"] != "native" for q in doc.quality), "flagged_pages": flagged,
                         "unread_pages": [q["page"] for q in doc.quality if "needs_ocr" in q["warnings"]]}
                report["documents"].append(entry)
                print(f"  {len(chunks)} chunks, {entry['tables']} tables, {len(flagged)} flagged pages", file=sys.stderr, flush=True)
            except Exception as exc:
                report["failures"].append({"source": str(path), "error": f"{type(exc).__name__}: {exc}"})
                print(f"  FAILED: {exc}", file=sys.stderr, flush=True)
            report["elapsed_seconds"] = round(time.time() - started, 2)
            report["index"] = index.stats()
            write_json(output / "ingestion_report.json", report)
    print(json.dumps({"report": str(output / "ingestion_report.json"), **report["index"]}, indent=2))
    incomplete = any(d["unread_pages"] for d in report["documents"])
    return 2 if report["failures"] or incomplete and args.strict else 0


def open_search(args):
    if not Path(args.index).is_file():
        raise ValueError(f"Index does not exist: {args.index}")
    index = SearchIndex(args.index)
    if args.mode != "lexical":
        if not index.embedding_model:
            index.db.close()
            raise ValueError("Index has no embeddings. Use --mode lexical or build with --embedding-model")
        from .embeddings import SentenceEmbedder
        index.embedder = SentenceEmbedder(index.embedding_model, args.model_cache)
    return index


def reindex(args):
    """Re-chunk parsed artifacts without repeating PDF extraction or OCR."""
    files = sorted((Path(args.source) / "documents").glob("*.json"))
    source_index = Path(args.source) / "index.sqlite"
    if source_index.exists():
        # Ingestion retains historical JSON exports; only reindex the active revision set.
        with sqlite3.connect(source_index.resolve().as_uri() + "?mode=ro", uri=True) as db:
            active_ids = {row[0] for row in db.execute("SELECT id FROM documents")}
        files = [p for p in files if p.stem in active_ids]
    if not files:
        raise ValueError("No parsed documents found under source/documents")
    embedder = None
    if args.embedding_model:
        from .embeddings import SentenceEmbedder
        embedder = SentenceEmbedder(args.embedding_model, args.model_cache)
    if args.semantic_boundaries and not embedder:
        raise ValueError("--semantic-boundaries requires --embedding-model")
    chunker = StructuralChunker(args.max_tokens, min(80, args.max_tokens),
                                embedder if args.semantic_boundaries else None, args.semantic_threshold)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    report = {"documents": [], "source": args.source, "configuration": {k: v for k, v in vars(args).items() if k != "func"}}
    with SearchIndex(output / "index.sqlite", embedder) as index:
        for i, path in enumerate(files, 1):
            doc = Document.from_dict(json.loads(path.read_text(encoding="utf-8")))
            print(f"[{i}/{len(files)}] {Path(doc.source).name}", file=sys.stderr, flush=True)
            if args.refresh_structure:
                # Refresh curated title/source metadata along with the hierarchy when available.
                source_path = Path(doc.source)
                manifest = next((root / "manifest.json" for root in list(source_path.parents)[:3]
                                 if (root / "manifest.json").exists()), None)
                if manifest:
                    for item in json.loads(manifest.read_text(encoding="utf-8")):
                        if (manifest.parent / item["filename"]).resolve() == source_path.resolve():
                            doc.metadata.update(item)
                            doc.title = item.get("title") or doc.title
                            break
                doc = PDFParser(ocr="off").refresh_structure(doc)
            chunks = chunker.chunk(doc)
            index.upsert(doc, chunks)
            write_json(output / "documents" / path.name, doc.to_dict())
            target = output / "chunks" / f"{doc.id}.jsonl"
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(".jsonl.tmp")
            temporary.write_text("".join(json.dumps(c.to_dict(), ensure_ascii=False) + "\n" for c in chunks), encoding="utf-8")
            temporary.replace(target)
            report["documents"].append({"id": doc.id, "source": doc.source, "chunks": len(chunks)})
            report["index"] = index.stats()
            write_json(output / "reindex_report.json", report)
    print(json.dumps(report["index"], indent=2))
    return 0


def search(args):
    reranker = None
    if args.reranker:
        from .embeddings import CrossEncoderReranker
        reranker = CrossEncoderReranker(args.reranker, args.model_cache)
    with open_search(args) as index:
        results = index.search(args.query, args.k, args.mode, args.document_id, args.category, reranker)
        if args.expand:
            for result in results:
                result["parent_context"] = index.expand(result["id"], args.context_tokens)
        print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


def evaluate(args):
    """Page-labelled retrieval evaluation; a smoke benchmark is not an accuracy guarantee."""
    cases = [json.loads(line) for line in Path(args.cases).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not cases:
        raise ValueError("Evaluation cases are empty")
    details, reciprocal, recalls = [], [], []
    reranker = None
    if args.reranker:
        from .embeddings import CrossEncoderReranker
        reranker = CrossEncoderReranker(args.reranker, args.model_cache)
    with open_search(args) as index:
        for case in cases:
            relevant = {(r["source"], int(page)) for r in case["relevant"] for page in r["pages"]}
            if not relevant:
                raise ValueError("Every case must have relevant source/page labels")
            results = index.search(case["query"], args.k, args.mode, category=case.get("category"), reranker=reranker)
            found, first = set(), 0
            for rank, result in enumerate(results, 1):
                pages = {(Path(c["source"]).name, c["page"]) for c in result["citations"]}
                matches = pages & relevant
                found |= matches
                if matches and not first:
                    first = rank
            reciprocal.append(1 / first if first else 0)
            recalls.append(len(found) / len(relevant))
            details.append({"query": case["query"], "first_relevant_rank": first or None,
                            "page_recall": recalls[-1], "retrieved_ids": [r["id"] for r in results]})
    report = {"cases": len(cases), "k": args.k, "mode": args.mode,
              "reranker": args.reranker,
              "hit_rate": sum(r > 0 for r in reciprocal) / len(cases),
              "mrr": sum(reciprocal) / len(cases), "mean_page_recall": sum(recalls) / len(cases), "details": details}
    write_json(args.output, report)
    print(json.dumps(report, indent=2))
    return 0


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Parse, chunk, index and search PDFs with source coordinates")
    subs = parser.add_subparsers(dest="command", required=True)
    p = subs.add_parser("ingest")
    p.add_argument("source")
    p.add_argument("--output", default="output/corpus")
    p.add_argument("--ocr", choices=["auto", "off", "tesseract", "rapidocr"], default="auto")
    p.add_argument("--ocr-language", default="eng")
    p.add_argument("--max-tokens", type=int, default=384)
    p.add_argument("--embedding-model", help="Local model path or Sentence Transformers model ID")
    p.add_argument("--model-cache", default=".model-cache")
    p.add_argument("--semantic-boundaries", action="store_true")
    p.add_argument("--semantic-threshold", type=float, default=.35)
    p.add_argument("--strict", action="store_true", help="Exit nonzero if pages needing OCR remain unread")
    p.set_defaults(func=ingest)
    p = subs.add_parser("reindex", help="Build a new index from parsed JSON, without rerunning OCR")
    p.add_argument("source")
    p.add_argument("--output", default="output/hybrid")
    p.add_argument("--embedding-model")
    p.add_argument("--semantic-boundaries", action="store_true")
    p.add_argument("--semantic-threshold", type=float, default=.35)
    p.add_argument("--max-tokens", type=int, default=384)
    p.add_argument("--model-cache", default=".model-cache")
    p.add_argument("--refresh-structure", action="store_true", help="Reapply current hierarchy rules to saved geometry; source PDFs must be unchanged")
    p.set_defaults(func=reindex)
    for command in ("search", "evaluate"):
        p = subs.add_parser(command)
        p.add_argument("--index", default="output/corpus/index.sqlite")
        p.add_argument("--mode", choices=["lexical", "hybrid", "dense"], default="lexical")
        p.add_argument("--model-cache", default=".model-cache")
        p.add_argument("-k", type=int, default=5)
        p.add_argument("--reranker")
        if command == "search":
            p.add_argument("query")
            p.add_argument("--category")
            p.add_argument("--document-id")
            p.add_argument("--expand", action="store_true")
            p.add_argument("--context-tokens", type=int, default=1600)
            p.set_defaults(func=search)
        else:
            p.add_argument("cases")
            p.add_argument("--output", default="output/evaluation.json")
            p.set_defaults(func=evaluate)
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, ImportError, OSError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
