from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from pathlib import Path
from typing import Protocol

from .models import KnowledgeHit

TOKENS = re.compile(r"[a-z0-9][a-z0-9_-]+")


class Retriever(Protocol):
    source: str
    def search(self, query: str, limit: int = 4) -> list[KnowledgeHit]: ...


def sections_from_markdown(path: str | Path) -> list[tuple[str, str]]:
    sections, title, body = [], "Overview", []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            if body:
                sections.append((title, "\n".join(body).strip()))
            title, body = line[3:].strip(), []
        elif not line.startswith("# "):
            body.append(line)
    if body:
        sections.append((title, "\n".join(body).strip()))
    return [(title, text) for title, text in sections if text]


def sections_from_file(path: str | Path, display_name: str | None = None) -> list[tuple[str, str]]:
    """Load a Markdown, plain-text, or PDF knowledge source into named sections."""
    path = Path(path)
    name = display_name or path.name
    if path.suffix.lower() in {".md", ".markdown"}:
        return [(f"{name} · {title}", text) for title, text in sections_from_markdown(path)]
    if path.suffix.lower() == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise RuntimeError("Install pypdf to load PDF knowledge") from exc
        return [(f"{name} · page {number}", text) for number, page in enumerate(PdfReader(path).pages, 1) if (text := (page.extract_text() or "").strip())]
    if path.suffix.lower() == ".txt":
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        return [(name, text)] if text else []
    raise ValueError(f"Unsupported knowledge file: {path.suffix}")


class KnowledgeBase:
    def __init__(self, sections: list[tuple[str, str]], source: str = "knowledge") -> None:
        self.source = source
        self.sections = sections
        counts = [Counter(TOKENS.findall(f"{title} {text}".lower())) for title, text in sections]
        frequencies = Counter(token for count in counts for token in count)
        self.idf = {token: math.log((len(sections) + 1) / (frequency + 1)) + 1 for token, frequency in frequencies.items()}
        self.vectors = [self._vector(count) for count in counts]

    def _vector(self, count: Counter[str]) -> dict[str, float]:
        vector = {token: amount * self.idf.get(token, 0) for token, amount in count.items()}
        norm = math.sqrt(sum(value * value for value in vector.values())) or 1
        return {token: value / norm for token, value in vector.items()}

    def search(self, query: str, limit: int = 4) -> list[KnowledgeHit]:
        if limit < 1:
            raise ValueError("limit must be positive")
        vector = self._vector(Counter(TOKENS.findall(query.lower())))
        hits = [KnowledgeHit(title, text, sum(weight * candidate.get(token, 0) for token, weight in vector.items()), self.source) for (title, text), candidate in zip(self.sections, self.vectors)]
        return [hit for hit in sorted(hits, key=lambda item: item.score, reverse=True) if hit.score > 0][:limit]


class ChromaKnowledgeBase:
    """Lazy Chroma semantic retriever with injectable client/embedding function."""

    def __init__(
        self,
        sections: list[tuple[str, str]],
        *,
        source: str = "knowledge",
        persist_directory: str | Path = "data/chroma",
        collection_name: str | None = None,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        client=None,
        embedding_function=None,
    ) -> None:
        self.source = source
        if client is None:
            try:
                import chromadb
            except ImportError as exc:
                raise RuntimeError("Install chromadb to use semantic Chroma retrieval") from exc
            client = chromadb.PersistentClient(path=str(persist_directory))
        if embedding_function is None:
            try:
                from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
            except ImportError as exc:
                raise RuntimeError("Install chromadb and sentence-transformers for Chroma embeddings") from exc
            embedding_function = SentenceTransformerEmbeddingFunction(model_name=embedding_model)
        slug = re.sub(r"[^a-z0-9_-]+", "-", source.lower()).strip("-") or "knowledge"
        digest = hashlib.sha256("\n".join(f"{title}\0{text}" for title, text in sections).encode()).hexdigest()[:10]
        name = (collection_name or f"orbit-{slug}-{digest}")[:63]
        self.collection = client.get_or_create_collection(name=name, embedding_function=embedding_function, metadata={"hnsw:space": "cosine"})
        if sections:
            ids = [hashlib.sha256(f"{source}\0{title}\0{text}".encode()).hexdigest() for title, text in sections]
            self.collection.upsert(
                ids=ids,
                documents=[text for _, text in sections],
                metadatas=[{"title": title, "source": source} for title, _ in sections],
            )

    def search(self, query: str, limit: int = 4) -> list[KnowledgeHit]:
        if limit < 1:
            raise ValueError("limit must be positive")
        result = self.collection.query(query_texts=[query], n_results=limit, include=["documents", "metadatas", "distances"])
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        hits = []
        for text, metadata, distance in zip(documents, metadatas, distances):
            metadata = metadata or {}
            hits.append(
                KnowledgeHit(
                    str(metadata.get("title", "Untitled source")),
                    str(text),
                    max(0.0, min(1.0, 1.0 - float(distance))),
                    str(metadata.get("source", self.source)),
                )
            )
        return hits


class MultiSourceRetriever:
    """Fan out to named sources and merge the strongest distinct evidence."""

    source = "multi-source"

    def __init__(self, retrievers: dict[str, Retriever]) -> None:
        if not retrievers:
            raise ValueError("At least one retrieval source is required")
        self.retrievers = retrievers

    def search(self, query: str, limit: int = 4) -> list[KnowledgeHit]:
        if limit < 1:
            raise ValueError("limit must be positive")
        merged: list[KnowledgeHit] = []
        for name, retriever in self.retrievers.items():
            for hit in retriever.search(query, limit):
                if not hit.source or hit.source == "knowledge":
                    hit.source = name
                merged.append(hit)
        distinct: list[KnowledgeHit] = []
        seen: set[tuple[str, str]] = set()
        for hit in sorted(merged, key=lambda row: row.score, reverse=True):
            identity = (hit.source, hit.title)
            if identity not in seen:
                distinct.append(hit)
                seen.add(identity)
            if len(distinct) == limit:
                break
        return distinct


def build_retriever(
    sources: dict[str, list[tuple[str, str]]],
    *,
    backend: str = "lexical",
    persist_directory: str | Path = "data/chroma",
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> Retriever:
    if backend not in {"lexical", "chroma"}:
        raise ValueError("retrieval backend must be lexical or chroma")
    retrievers: dict[str, Retriever] = {}
    for name, sections in sources.items():
        retrievers[name] = (
            ChromaKnowledgeBase(sections, source=name, persist_directory=persist_directory, embedding_model=embedding_model)
            if backend == "chroma"
            else KnowledgeBase(sections, source=name)
        )
    return MultiSourceRetriever(retrievers) if len(retrievers) > 1 else next(iter(retrievers.values()))
