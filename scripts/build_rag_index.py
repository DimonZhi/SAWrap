#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


REPO_ROOT = _repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from UI.helpers_project_rag import (  # noqa: E402
    DEFAULT_EMBEDDING_MODEL,
    load_project_knowledge,
    load_thesis_knowledge,
    write_project_vector_index,
)


DEFAULT_THESIS_DIR = Path("/Users/dimonzhi/Downloads/ДипломML_SA-3")


def _resolve_optional_path(raw: str | None) -> Path | None:
    if not raw:
        return None
    path = Path(raw).expanduser()
    return path.resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build local vector RAG index for SAWrap knowledge, code, tables and thesis.",
    )
    parser.add_argument(
        "--repo-dir",
        default=str(REPO_ROOT),
        help="Project repository directory. Default: current SAWrap repository.",
    )
    parser.add_argument(
        "--index-dir",
        default=None,
        help="Where to write the index. Default: UI/rag_index so Docker can copy it with UI.",
    )
    parser.add_argument(
        "--thesis-dir",
        default=os.getenv("SAWRAP_THESIS_DIR", str(DEFAULT_THESIS_DIR)),
        help="LaTeX thesis directory. Set empty string to skip external thesis files.",
    )
    parser.add_argument(
        "--retriever",
        choices=["embeddings", "tfidf"],
        default=os.getenv("SAWRAP_RAG_RETRIEVER", "embeddings"),
        help="Index type. embeddings uses sentence-transformers; tfidf uses scikit-learn only.",
    )
    parser.add_argument(
        "--embedding-model",
        default=os.getenv("SAWRAP_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
        help="SentenceTransformers model for --retriever embeddings.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_dir = Path(args.repo_dir).expanduser().resolve()
    index_dir = _resolve_optional_path(args.index_dir)
    thesis_dir = _resolve_optional_path(args.thesis_dir)

    chunks = list(load_project_knowledge(str(repo_dir)))
    thesis_chunks = list(load_thesis_knowledge(repo_dir, thesis_dir))
    chunks.extend(thesis_chunks)

    manifest = write_project_vector_index(
        repo_dir,
        chunks,
        index_dir=index_dir,
        retriever=args.retriever,
        embedding_model=args.embedding_model,
    )
    print(f"RAG index written to: {manifest['index_dir']}")
    print(f"Retriever: {manifest['retriever']}")
    if manifest.get("embedding_model"):
        print(f"Embedding model: {manifest['embedding_model']}")
    print(f"Chunks: {manifest['chunk_count']}")
    if manifest.get("feature_count"):
        print(f"Features: {manifest['feature_count']}")
    if manifest.get("embedding_dim"):
        print(f"Embedding dim: {manifest['embedding_dim']}")
    if thesis_chunks:
        print(f"Thesis chunks: {len(thesis_chunks)} from {thesis_dir}")
    else:
        print("Thesis chunks: 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
