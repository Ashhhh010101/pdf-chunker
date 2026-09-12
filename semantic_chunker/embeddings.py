from __future__ import annotations

import numpy as np


class SentenceEmbedder:
    """Local embeddings with token-window pooling, never silent model truncation."""
    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2", cache_folder=None):
        from sentence_transformers import SentenceTransformer
        self.name = model
        try:
            self.model = SentenceTransformer(model, cache_folder=cache_folder, trust_remote_code=False,
                                             device="cpu", local_files_only=True)
        except OSError:
            self.model = SentenceTransformer(model, cache_folder=cache_folder, trust_remote_code=False, device="cpu")
        self.dimension = (self.model.get_embedding_dimension() if hasattr(self.model, "get_embedding_dimension")
                          else self.model.get_sentence_embedding_dimension())

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        tokenizer = self.model.tokenizer
        limit = max(16, self.model.max_seq_length - 16)
        windows, owners, weights = [], [], []
        for i, text in enumerate(texts):
            ids = tokenizer.encode(text, add_special_tokens=False, verbose=False)
            for start in range(0, max(1, len(ids)), limit):
                tokens = ids[start:start+limit]
                windows.append(tokenizer.decode(tokens) if len(ids) > limit else text)
                owners.append(i)
                weights.append(max(1, len(tokens)))
        encoded = self.model.encode_document(windows, batch_size=32, normalize_embeddings=True,
                                             convert_to_numpy=True, show_progress_bar=False)
        result = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for owner, weight, vector in zip(owners, weights, encoded):
            result[owner] += vector * weight
        result /= np.maximum(np.linalg.norm(result, axis=1, keepdims=True), 1e-12)
        return result

    def encode_query(self, text: str) -> np.ndarray:
        if len(self.model.tokenizer.encode(text)) > self.model.max_seq_length - 16:
            raise ValueError("Query exceeds embedding model token limit; shorten the query")
        return self.model.encode_query(text, normalize_embeddings=True, convert_to_numpy=True)


class CrossEncoderReranker:
    def __init__(self, model="cross-encoder/ms-marco-MiniLM-L-6-v2", cache_folder=None):
        from sentence_transformers import CrossEncoder
        try:
            self.model = CrossEncoder(model, cache_folder=cache_folder, trust_remote_code=False,
                                      device="cpu", local_files_only=True)
        except OSError:
            self.model = CrossEncoder(model, cache_folder=cache_folder, trust_remote_code=False, device="cpu")

    def score(self, query, texts):
        if not texts:
            return []
        tokenizer = self.model.tokenizer
        limit = self.model.max_length or 512
        query_length = len(tokenizer.encode(query, add_special_tokens=False, verbose=False))
        budget = limit - query_length - 8
        if budget < 32:
            raise ValueError("Query exceeds reranker token budget; shorten the query")
        pairs, owners = [], []
        for owner, text in enumerate(texts):
            tokens = tokenizer.encode(text, add_special_tokens=False, verbose=False)
            for start in range(0, max(1, len(tokens)), max(1, budget - 32)):
                window = tokenizer.decode(tokens[start:start+budget]) if len(tokens) > budget else text
                pairs.append((query, window))
                owners.append(owner)
                if start + budget >= len(tokens):
                    break
        values = np.asarray(self.model.predict(pairs), dtype=float).reshape(-1)
        scores = np.full(len(texts), -np.inf)
        for owner, score in zip(owners, values):
            scores[owner] = max(scores[owner], score)
        return scores.tolist()
