from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import sqlite3
import time

import numpy as np

from .models import Chunk, Document


class SearchIndex:
    """Transactional SQLite FTS5 + exact dense retrieval, suitable for local corpora."""
    QUERY_STOPWORDS = {
        "a", "an", "and", "are", "as", "be", "by", "can", "does", "for", "from", "how", "in", "is",
        "it", "of", "on", "or", "the", "this", "to", "what", "when", "which", "who", "with",
        "extract", "identify", "list", "prepare", "state", "summarize", "entry", "matrix", "row",
    }
    def __init__(self, path: str | Path, embedder=None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path.resolve()
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.embedder = embedder
        self._dense_cache = None
        self.last_search_timings = {}
        self.db.executescript("""
            PRAGMA foreign_keys=ON;
            CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS documents(id TEXT PRIMARY KEY, source TEXT UNIQUE, payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS sections(id TEXT PRIMARY KEY, document_id TEXT REFERENCES documents(id)
                ON DELETE CASCADE, payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS chunks(id TEXT PRIMARY KEY, document_id TEXT REFERENCES documents(id)
                ON DELETE CASCADE, section_id TEXT, category TEXT, payload TEXT NOT NULL, vector BLOB);
            CREATE INDEX IF NOT EXISTS chunks_document ON chunks(document_id);
            CREATE INDEX IF NOT EXISTS chunks_section ON chunks(section_id);
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(id UNINDEXED, context, text,
                tokenize='unicode61 remove_diacritics 2');
        """)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.db.close()

    @property
    def embedding_model(self):
        row = self.db.execute("SELECT value FROM settings WHERE key='embedding_model'").fetchone()
        return row[0] if row else None

    def _check_model(self):
        if self.embedder and self.embedding_model and self.embedder.name != self.embedding_model:
            raise ValueError(f"Index uses {self.embedding_model}, supplied model is {self.embedder.name}; rebuild in a new index")

    def upsert(self, document: Document, chunks: list[Chunk]):
        self._check_model()
        if self.embedding_model and not self.embedder:
            raise ValueError("A dense index must be updated with its embedding model")
        if self.embedder and not self.embedding_model and self.db.execute("SELECT count(*) FROM chunks").fetchone()[0]:
            raise ValueError("Cannot mix lexical-only and dense documents; build a new index")
        if any(c.document_id != document.id for c in chunks):
            raise ValueError("Chunk document ID mismatch")
        vectors = self.embedder.encode_documents([c.context + "\n" + c.text for c in chunks]) if self.embedder else None
        if vectors is not None and (len(vectors) != len(chunks) or not np.isfinite(vectors).all()):
            raise ValueError("Invalid embeddings")
        with self.db:
            # Remove both old content at this path and duplicate content from another path.
            old = self.db.execute("SELECT id FROM documents WHERE source=? OR id=?", (document.source, document.id)).fetchall()
            for row in old:
                self._delete(row[0])
            self.db.execute("INSERT INTO documents VALUES(?,?,?)", (document.id, document.source, json.dumps({
                k: v for k, v in document.to_dict().items() if k not in {"elements", "sections"}}, ensure_ascii=False)))
            for section in document.sections:
                from dataclasses import asdict
                self.db.execute("INSERT INTO sections VALUES(?,?,?)", (section.id, document.id, json.dumps(asdict(section), ensure_ascii=False)))
            for i, c in enumerate(chunks):
                vector = np.asarray(vectors[i], dtype="<f4").tobytes() if vectors is not None else None
                self.db.execute("INSERT INTO chunks VALUES(?,?,?,?,?,?)", (c.id, c.document_id, c.section_id,
                                c.metadata.get("category", ""), json.dumps(c.to_dict(), ensure_ascii=False), vector))
                self.db.execute("INSERT INTO chunks_fts VALUES(?,?,?)", (c.id, c.context, c.text))
            if self.embedder:
                self.db.execute("INSERT OR REPLACE INTO settings VALUES('embedding_model',?)", (self.embedder.name,))
        self._dense_cache = None

    def _delete(self, doc_id):
        self.db.execute("DELETE FROM chunks_fts WHERE id IN (SELECT id FROM chunks WHERE document_id=?)", (doc_id,))
        self.db.execute("DELETE FROM documents WHERE id=?", (doc_id,))

    def delete_document(self, doc_id: str):
        with self.db:
            self._delete(doc_id)
        self._dense_cache = None

    def stats(self):
        return {**{t: self.db.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
                   for t in ("documents", "sections", "chunks")}, "embedding_model": self.embedding_model}

    @staticmethod
    def query_variants(query: str) -> list[str]:
        """Conservative variants for tender-compliance instruction wording."""
        variants = []
        stripped = re.sub(
            r"^(?:please\s+)?(?:extract|identify|summarize|state|list|prepare)\s+", "", query,
            flags=re.IGNORECASE,
        )
        stripped = re.sub(r"\b(?:for|in)\s+the\s+(?:technical |commercial )?compliance[- ]matrix\b", "", stripped,
                          flags=re.IGNORECASE)
        stripped = " ".join(stripped.split()).strip(" .?")
        if stripped and stripped.casefold() != query.strip(" .?").casefold():
            variants.append(stripped)
        acronym = re.sub(r"\bEMD\b", "earnest money deposit", query, flags=re.IGNORECASE)
        acronym = re.sub(r"\bOEM\b", "original equipment manufacturer", acronym, flags=re.IGNORECASE)
        if acronym != query:
            variants.append(acronym)
        return list(dict.fromkeys(variants))[:2]

    def _lexical_rows(self, query, condition, params, pool, separate_connection=False):
        tokens = re.findall(r"\w+", query, flags=re.UNICODE)
        informative = [token for token in tokens if token.casefold() not in self.QUERY_STOPWORDS]
        tokens = informative or tokens
        match = " OR ".join('"' + t + '"' for t in dict.fromkeys(tokens[:80]))
        if not match:
            return []
        sql = f"""
            SELECT c.id, c.payload, bm25(chunks_fts, 0, 0.7, 1) AS score
            FROM chunks_fts JOIN chunks c ON c.id=chunks_fts.id
            WHERE chunks_fts MATCH ? AND {condition} ORDER BY score, c.id LIMIT ?
        """
        if not separate_connection:
            return self.db.execute(sql, [match] + params + [pool]).fetchall()
        uri = self.path.as_uri() + "?mode=ro"
        with sqlite3.connect(uri, uri=True) as db:
            db.row_factory = sqlite3.Row
            return db.execute(sql, [match] + params + [pool]).fetchall()

    def _ensure_dense_cache(self):
        if self._dense_cache is None:
            rows = self.db.execute("SELECT id,payload,document_id,category,vector FROM chunks WHERE vector IS NOT NULL").fetchall()
            matrix = (np.stack([np.frombuffer(row["vector"], dtype="<f4") for row in rows])
                      if rows else np.empty((0, 0), dtype=np.float32))
            self._dense_cache = (rows, matrix)
        return self._dense_cache

    def search(self, query: str, k: int = 5, mode: str = "hybrid", document_id: str | None = None,
               category: str | None = None, reranker=None, candidates: int = 50,
               parallel: bool = False, query_variants: bool = False) -> list[dict]:
        if mode not in {"lexical", "dense", "hybrid"}:
            raise ValueError("mode must be lexical, dense or hybrid")
        if k < 1 or candidates < 1:
            raise ValueError("k and candidates must be positive")
        if not query.strip():
            return []
        self._check_model()
        if mode != "lexical" and (not self.embedder or not self.embedding_model):
            raise ValueError("Dense/hybrid search requires an embedded index and its model; use mode='lexical'")
        where, params = ["1=1"], []
        if document_id:
            where.append("c.document_id=?")
            params.append(document_id)
        if category:
            where.append("c.category=?")
            params.append(category)
        condition = " AND ".join(where)
        # Extra candidates are only useful for fusion, variants, or reranking.
        pool = k if mode == "lexical" and not reranker and not query_variants else max(k, candidates)
        started = time.perf_counter()
        rankings, payloads, dense_query = {}, {}, None
        variants = self.query_variants(query) if query_variants and mode in {"lexical", "hybrid"} else []
        lexical_queries = [query] + variants if mode in {"lexical", "hybrid"} else []
        dense_rows, dense_matrix = self._ensure_dense_cache() if mode in {"dense", "hybrid"} else ([], None)

        if parallel and mode == "hybrid":
            with ThreadPoolExecutor(max_workers=2 + len(variants)) as executor:
                lexical_futures = [executor.submit(self._lexical_rows, text, condition, params, pool, True)
                                   for text in lexical_queries]
                dense_future = executor.submit(self.embedder.encode_query, query)
                lexical_results = [future.result() for future in lexical_futures]
                dense_query = dense_future.result()
        elif parallel and len(lexical_queries) > 1:
            with ThreadPoolExecutor(max_workers=len(lexical_queries)) as executor:
                futures = [executor.submit(self._lexical_rows, text, condition, params, pool, True)
                           for text in lexical_queries]
                lexical_results = [future.result() for future in futures]
            if mode == "dense":
                dense_query = self.embedder.encode_query(query)
        else:
            lexical_results = [self._lexical_rows(text, condition, params, pool) for text in lexical_queries]
            if mode in {"dense", "hybrid"}:
                dense_query = self.embedder.encode_query(query)

        for number, rows in enumerate(lexical_results):
            method = "lexical" if number == 0 else f"lexical_variant_{number}"
            rankings[method] = [row["id"] for row in rows]
            payloads.update({row["id"]: json.loads(row["payload"]) for row in rows})
        if mode in {"dense", "hybrid"}:
            q = np.asarray(dense_query, dtype=np.float32)
            rows, matrix = dense_rows, dense_matrix
            eligible = [i for i, row in enumerate(rows)
                        if (not document_id or row["document_id"] == document_id)
                        and (not category or row["category"] == category)]
            if eligible:
                filtered_matrix = matrix[eligible]
                if matrix.shape[1] != q.shape[0] or not np.isfinite(q).all():
                    raise ValueError("Query vector does not match index dimensions or is non-finite")
                scores = filtered_matrix @ q
                selected = np.argsort(-scores, kind="stable")[:pool]
                chosen = [eligible[i] for i in selected]
                rankings["dense"] = [rows[i]["id"] for i in chosen]
                payloads.update({rows[i]["id"]: json.loads(rows[i]["payload"]) for i in chosen})
        retrieval_finished = time.perf_counter()
        fused = defaultdict(float)
        ranks = defaultdict(dict)
        for method, ids in rankings.items():
            weight = 1.5 if method == "lexical" else .7 if method.startswith("lexical_variant") else 1.0
            for rank, cid in enumerate(ids, 1):
                fused[cid] += weight / (60 + rank)
                ranks[cid][method] = rank
        selected = sorted(fused, key=lambda cid: (-fused[cid], cid))[:pool]
        merged_finished = time.perf_counter()
        rerank_scores = {}
        if reranker and selected:
            values = reranker.score(query, [payloads[cid]["context"] + "\n" + payloads[cid]["text"] for cid in selected])
            rerank_scores = dict(zip(selected, values))
            selected.sort(key=lambda cid: (-rerank_scores[cid], -fused[cid]))
        rerank_finished = time.perf_counter()
        results = []
        for cid in selected[:k]:
            result = payloads[cid]
            result["score"] = float(rerank_scores.get(cid, fused[cid]))
            result["retrieval_ranks"] = ranks[cid]
            result["retrieval_mode"] = mode
            result["score_type"] = "cross_encoder" if reranker else "reciprocal_rank_fusion"
            results.append(result)
        self.last_search_timings = {
            "retrieval_ms": round((retrieval_finished - started) * 1000, 3),
            "merge_ms": round((merged_finished - retrieval_finished) * 1000, 3),
            "rerank_ms": round((rerank_finished - merged_finished) * 1000, 3),
            "format_ms": round((time.perf_counter() - rerank_finished) * 1000, 3),
        }
        return results

    def expand(self, chunk_id: str, max_tokens: int = 1600) -> dict:
        """Fetch bounded parent-section context; never cross a section or document boundary."""
        row = self.db.execute("SELECT payload FROM chunks WHERE id=?", (chunk_id,)).fetchone()
        if not row:
            raise KeyError(chunk_id)
        center = json.loads(row[0])
        if max_tokens < center["token_count"]:
            raise ValueError("Context budget is smaller than the matched chunk")
        chosen, used = [center], center["token_count"]
        for direction in ("previous_id", "next_id"):
            current = center
            while current[direction]:
                row = self.db.execute("SELECT payload FROM chunks WHERE id=?", (current[direction],)).fetchone()
                if not row:
                    break
                current = json.loads(row[0])
                if current["section_id"] != center["section_id"] or current["document_id"] != center["document_id"]:
                    break
                if used + current["token_count"] > max_tokens:
                    break
                chosen.insert(0, current) if direction == "previous_id" else chosen.append(current)
                used += current["token_count"]
        return {"section_id": center["section_id"], "section_path": center["section_path"],
                "chunks": chosen, "token_count": used}
