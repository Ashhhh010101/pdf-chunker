from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import re
import sqlite3

import numpy as np

from .models import Chunk, Document


class SearchIndex:
    """Transactional SQLite FTS5 + exact dense retrieval, suitable for local corpora."""
    def __init__(self, path: str | Path, embedder=None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.embedder = embedder
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

    def _delete(self, doc_id):
        self.db.execute("DELETE FROM chunks_fts WHERE id IN (SELECT id FROM chunks WHERE document_id=?)", (doc_id,))
        self.db.execute("DELETE FROM documents WHERE id=?", (doc_id,))

    def delete_document(self, doc_id: str):
        with self.db:
            self._delete(doc_id)

    def stats(self):
        return {**{t: self.db.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
                   for t in ("documents", "sections", "chunks")}, "embedding_model": self.embedding_model}

    def search(self, query: str, k: int = 5, mode: str = "hybrid", document_id: str | None = None,
               category: str | None = None, reranker=None, candidates: int = 50) -> list[dict]:
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
        pool = max(k, candidates)
        rankings, payloads = {}, {}
        if mode in {"lexical", "hybrid"}:
            tokens = re.findall(r"\w+", query, flags=re.UNICODE)
            # Quote tokens so user queries cannot inject FTS operators/syntax.
            match = " OR ".join('"' + t + '"' for t in dict.fromkeys(tokens[:80]))
            rows = [] if not match else self.db.execute(f"""
                SELECT c.id, c.payload, bm25(chunks_fts, 0, 0.7, 1) AS score
                FROM chunks_fts JOIN chunks c ON c.id=chunks_fts.id
                WHERE chunks_fts MATCH ? AND {condition} ORDER BY score, c.id LIMIT ?
            """, [match] + params + [pool]).fetchall()
            rankings["lexical"] = [r["id"] for r in rows]
            payloads.update({r["id"]: json.loads(r["payload"]) for r in rows})
        if mode in {"dense", "hybrid"}:
            q = np.asarray(self.embedder.encode_query(query), dtype=np.float32)
            rows = self.db.execute(f"SELECT c.id,c.payload,c.vector FROM chunks c WHERE {condition} AND c.vector IS NOT NULL", params).fetchall()
            if rows:
                matrix = np.stack([np.frombuffer(r["vector"], dtype="<f4") for r in rows])
                if matrix.shape[1] != q.shape[0] or not np.isfinite(q).all():
                    raise ValueError("Query vector does not match index dimensions or is non-finite")
                scores = matrix @ q
                selected = np.argsort(-scores, kind="stable")[:pool]
                rankings["dense"] = [rows[i]["id"] for i in selected]
                payloads.update({rows[i]["id"]: json.loads(rows[i]["payload"]) for i in selected})
        fused = defaultdict(float)
        ranks = defaultdict(dict)
        for method, ids in rankings.items():
            for rank, cid in enumerate(ids, 1):
                fused[cid] += 1 / (60 + rank)
                ranks[cid][method] = rank
        selected = sorted(fused, key=lambda cid: (-fused[cid], cid))[:pool]
        rerank_scores = {}
        if reranker and selected:
            values = reranker.score(query, [payloads[cid]["context"] + "\n" + payloads[cid]["text"] for cid in selected])
            rerank_scores = dict(zip(selected, values))
            selected.sort(key=lambda cid: (-rerank_scores[cid], -fused[cid]))
        results = []
        for cid in selected[:k]:
            result = payloads[cid]
            result["score"] = float(rerank_scores.get(cid, fused[cid]))
            result["retrieval_ranks"] = ranks[cid]
            result["retrieval_mode"] = mode
            result["score_type"] = "cross_encoder" if reranker else "reciprocal_rank_fusion"
            results.append(result)
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
