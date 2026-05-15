from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
import json
import os
from pathlib import Path
import pickle
import re
from typing import Any
from urllib.request import urlopen

import numpy as np
import pandas as pd

from .helpers_ai_advice import (
    DEFAULT_OPENROUTER_MODEL,
    _call_openrouter,
    _normalize_chat_history,
)


TOKEN_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9_]+")

STOPWORDS = {
    "а", "без", "более", "бы", "был", "была", "были", "было", "в", "во",
    "для", "до", "его", "ее", "если", "есть", "и", "из", "или", "к", "как",
    "ко", "на", "над", "не", "но", "о", "об", "от", "по", "при", "с", "со",
    "так", "то", "у", "что", "это", "этот", "эта", "эти", "the", "and", "or",
    "to", "of", "in", "for", "is", "are", "with", "as", "on", "by",
}

KNOWLEDGE_GLOBS = [
    "knowledge/*.md",
]

TEXT_SOURCE_PATHS = [
    "README.txt",
    "rank.py",
    "wrapSA.py",
    "run_many_server.py",
    "UI/app.py",
    "UI/helpers_ai_advice.py",
    "UI/helpers_tables.py",
    "UI/helpers_leaderboard.py",
]

DATASET_TABLES = {
    "actg",
    "gbsg",
    "pbc",
    "rott2",
    "smarto",
    "framingham",
    "support2",
}

THESIS_SOURCE_GLOBS = [
    "contents/*.tex",
    "main.tex",
    "title.tex",
    "links.txt",
]

RAG_INDEX_ENV = "SAWRAP_RAG_INDEX_DIR"
DEFAULT_RAG_INDEX_RELATIVE = Path("UI") / "rag_index"
LEGACY_RAG_INDEX_RELATIVE = Path("rag_index")
DEFAULT_EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
SUPPORTED_RETRIEVERS = {"tfidf", "embeddings", "semantic"}
DEFAULT_RAG_TOP_K = 14
OVERVIEW_SOURCES = [
    "knowledge/project_context.md",
    "knowledge/thesis_summary.md",
    "knowledge/experiments_summary.md",
    "knowledge/engineering_evidence.md",
    "knowledge/data_science_evidence.md",
    "knowledge/ai_usage_evidence.md",
    "knowledge/product_evidence.md",
    "knowledge/project_faq.md",
    "knowledge/product_context.md",
]
FOUNDATION_REQUIRED_SECTIONS = [
    ("knowledge/project_faq.md", "В чем главная научная идея"),
    ("knowledge/project_context.md", "Центральная идея"),
    ("knowledge/thesis_summary.md", "Научная новизна"),
    ("knowledge/experiments_summary.md", "Модели"),
    ("knowledge/experiments_summary.md", "Метрики"),
    ("knowledge/experiments_summary.md", "Формулы ключевых метрик"),
    ("knowledge/experiments_summary.md", "Протокол эксперимента"),
    ("knowledge/experiments_summary.md", "Итоговое ранжирование"),
    ("knowledge/experiments_summary.md", "Интерпретация"),
    ("knowledge/engineering_evidence.md", "Разработка и инженерия"),
    ("knowledge/data_science_evidence.md", "Data Science"),
    ("knowledge/ai_usage_evidence.md", "Применение AI"),
    ("knowledge/product_evidence.md", "Продуктовое мышление"),
]
FOUNDATION_TITLE_KEYWORDS = {
    "центральная идея",
    "научная идея",
    "научная новизна",
    "метрики",
    "методы оценки качества",
    "качество моделей",
    "модели",
    "протокол эксперимента",
    "итоговое ранжирование",
    "интерпретация",
    "результаты",
    "ключевые выводы",
    "metric:",
    "формул",
}


@dataclass(frozen=True)
class KnowledgeChunk:
    source: str
    title: str
    text: str
    score: float = 0.0

    def to_public_dict(self) -> dict[str, Any]:
        preview = " ".join(self.text.split())
        if len(preview) > 260:
            preview = preview[:257].rstrip() + "..."
        return {
            "source": self.source,
            "title": self.title,
            "preview": preview,
            "score": round(self.score, 3),
        }


def _repo_relative(repo_dir: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(repo_dir.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="ignore")


def _tokenize(text: str) -> list[str]:
    tokens = []
    for raw in TOKEN_RE.findall(text.lower()):
        if len(raw) < 2 or raw in STOPWORDS:
            continue
        tokens.append(raw)
    return tokens


def _compact_whitespace(text: str) -> str:
    lines = [line.rstrip() for line in text.replace("\r\n", "\n").splitlines()]
    return "\n".join(lines).strip()


def _split_long_text(text: str, max_chars: int = 1500) -> list[str]:
    paragraphs = re.split(r"\n\s*\n", text)
    chunks = []
    current = ""

    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(paragraph) > max_chars:
            if current:
                chunks.append(current.strip())
                current = ""
            for start in range(0, len(paragraph), max_chars):
                chunks.append(paragraph[start:start + max_chars].strip())
            continue
        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if len(candidate) > max_chars and current:
            chunks.append(current.strip())
            current = paragraph
        else:
            current = candidate

    if current:
        chunks.append(current.strip())
    return chunks


def _chunks_from_text(
    repo_dir: Path,
    path: Path,
    text: str,
    source_override: str | None = None,
) -> list[KnowledgeChunk]:
    source = source_override or _repo_relative(repo_dir, path)
    clean_text = _compact_whitespace(text)
    if not clean_text:
        return []

    sections: list[tuple[str, str]] = []
    title = path.name
    buffer: list[str] = []

    for line in clean_text.splitlines():
        heading_match = re.match(r"^\s{0,3}#{1,6}\s+(.+?)\s*$", line)
        if heading_match and buffer:
            sections.append((title, "\n".join(buffer).strip()))
            title = heading_match.group(1).strip()
            buffer = [line]
            continue
        if heading_match:
            title = heading_match.group(1).strip()
        buffer.append(line)

    if buffer:
        sections.append((title, "\n".join(buffer).strip()))

    chunks = []
    for section_title, section_text in sections:
        for part in _split_long_text(section_text):
            chunks.append(KnowledgeChunk(source=source, title=section_title, text=part))
    return chunks


def _clean_latex_text(text: str) -> str:
    text = re.sub(r"(?m)%.*$", "", text)
    text = re.sub(r"\\(?:section|subsection|subsubsection)\*?\{([^{}]+)\}", r"# \1", text)
    text = re.sub(r"\\(?:textbf|textit|texttt|emph|url|href)\{([^{}]+)\}", r"\1", text)
    text = re.sub(r"\\(?:begin|end)\{[^{}]+\}", "\n", text)
    text = re.sub(r"\\[A-Za-zА-Яа-яЁё]+\*?(?:\[[^\]]*\])?", " ", text)
    text = text.replace("{", " ").replace("}", " ")
    text = re.sub(r"\s+", " ", text)
    text = text.replace(". ", ".\n")
    return text


def _safe_table_preview(df: pd.DataFrame, max_rows: int = 12) -> str:
    compact = df.copy()
    compact.columns = [str(column).strip() for column in compact.columns]
    if "method" in {column.lower() for column in compact.columns}:
        method_col = next(column for column in compact.columns if column.lower() == "method")
        metric_cols = [column for column in compact.columns if column.lower().endswith("_mean")]
        compact = compact[[method_col, *metric_cols[:12]]]
    compact = compact.head(max_rows)
    return compact.to_csv(index=False)


def _table_chunks(repo_dir: Path) -> list[KnowledgeChunk]:
    table_dir = repo_dir / "UI" / "tables"
    if not table_dir.exists():
        return []

    chunks: list[KnowledgeChunk] = []
    for table_path in sorted(table_dir.glob("*.xlsx")):
        stem = table_path.stem.lower()
        if stem not in DATASET_TABLES and stem != "leaderboards_by_task":
            continue

        source = _repo_relative(repo_dir, table_path)
        try:
            if stem == "leaderboards_by_task":
                workbook = pd.ExcelFile(table_path)
                for sheet in workbook.sheet_names[:6]:
                    df = pd.read_excel(workbook, sheet_name=sheet)
                    text = (
                        f"Сводная таблица рангов {table_path.name}, лист {sheet}.\n"
                        f"Строк: {len(df)}. Колонки: {', '.join(map(str, df.columns[:16]))}.\n"
                        f"{_safe_table_preview(df, max_rows=16)}"
                    )
                    chunks.append(KnowledgeChunk(source=source, title=f"Таблица {sheet}", text=text))
                continue

            df = pd.read_excel(table_path)
        except Exception:
            continue

        metric_cols = [str(column) for column in df.columns if str(column).lower().endswith("_mean")]
        text = (
            f"Таблица результатов для датасета {table_path.stem}.\n"
            f"Строк: {len(df)}. Метрики: {', '.join(metric_cols[:18])}.\n"
            f"{_safe_table_preview(df)}"
        )
        chunks.append(KnowledgeChunk(source=source, title=f"Результаты {table_path.stem}", text=text))
    return chunks


def load_thesis_knowledge(repo_dir: Path, thesis_dir: Path | None) -> tuple[KnowledgeChunk, ...]:
    if thesis_dir is None or not thesis_dir.exists():
        return tuple()

    chunks: list[KnowledgeChunk] = []
    for pattern in THESIS_SOURCE_GLOBS:
        for path in sorted(thesis_dir.glob(pattern)):
            if not path.is_file():
                continue
            try:
                relative = path.relative_to(thesis_dir).as_posix()
            except ValueError:
                relative = path.name
            text = _read_text(path)
            if path.suffix == ".tex":
                text = _clean_latex_text(text)
            chunks.extend(
                _chunks_from_text(
                    repo_dir,
                    path,
                    text,
                    source_override=f"thesis/{relative}",
                )
            )
    return tuple(chunks)


@lru_cache(maxsize=4)
def load_project_knowledge(repo_dir_raw: str) -> tuple[KnowledgeChunk, ...]:
    repo_dir = Path(repo_dir_raw).resolve()
    chunks: list[KnowledgeChunk] = []

    for pattern in KNOWLEDGE_GLOBS:
        for path in sorted(repo_dir.glob(pattern)):
            if path.is_file():
                chunks.extend(_chunks_from_text(repo_dir, path, _read_text(path)))

    for relative_path in TEXT_SOURCE_PATHS:
        path = repo_dir / relative_path
        if path.exists() and path.is_file():
            chunks.extend(_chunks_from_text(repo_dir, path, _read_text(path)))

    chunks.extend(_table_chunks(repo_dir))
    return tuple(chunks)


def _rag_index_dir(repo_dir: Path, index_dir: Path | None = None) -> Path:
    raw = os.getenv(RAG_INDEX_ENV)
    if index_dir is not None:
        path = index_dir
    elif raw:
        path = Path(raw)
    else:
        ui_index = repo_dir / DEFAULT_RAG_INDEX_RELATIVE
        legacy_index = repo_dir / LEGACY_RAG_INDEX_RELATIVE
        path = ui_index if ui_index.exists() or not legacy_index.exists() else legacy_index
    if not path.is_absolute():
        path = repo_dir / path
    return path.resolve()


def _chunk_to_index_dict(chunk: KnowledgeChunk) -> dict[str, str]:
    return {
        "source": chunk.source,
        "title": chunk.title,
        "text": chunk.text,
    }


def _normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return array / norms


def _env_true(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _encode_with_model(embedder: Any, texts: list[str], *, batch_size: int = 32) -> np.ndarray:
    try:
        vectors = embedder.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
    except TypeError:
        try:
            vectors = embedder.encode(texts, normalize_embeddings=True)
        except TypeError:
            vectors = embedder.encode(texts)
    return _normalize_vectors(np.asarray(vectors, dtype=np.float32))


@lru_cache(maxsize=4)
def _load_sentence_transformer(model_name: str, local_files_only: bool = True):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return None
    try:
        return SentenceTransformer(model_name, local_files_only=local_files_only)
    except TypeError:
        try:
            return SentenceTransformer(model_name)
        except Exception:
            return None
    except Exception:
        return None


def _write_chunks_file(target_dir: Path, selected_chunks: list[KnowledgeChunk]) -> None:
    (target_dir / "chunks.json").write_text(
        json.dumps([_chunk_to_index_dict(chunk) for chunk in selected_chunks], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _write_tfidf_index(
    target_dir: Path,
    texts: list[str],
) -> dict[str, Any]:
    from scipy import sparse
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=1,
        max_df=1.0,
        sublinear_tf=True,
        token_pattern=r"(?u)\b\w\w+\b",
    )
    matrix = vectorizer.fit_transform(texts)
    with (target_dir / "vectorizer.pkl").open("wb") as file:
        pickle.dump(vectorizer, file)
    sparse.save_npz(target_dir / "matrix.npz", matrix)
    return {
        "retriever": "tfidf",
        "feature_count": int(matrix.shape[1]),
        "embedding_dim": None,
        "embedding_model": None,
        "files": ["chunks.json", "vectorizer.pkl", "matrix.npz", "manifest.json"],
    }


def _write_embedding_index(
    target_dir: Path,
    texts: list[str],
    *,
    embedding_model: str,
    embedder: Any | None = None,
) -> dict[str, Any]:
    if embedder is None:
        embedder = _load_sentence_transformer(embedding_model, local_files_only=True)
        if embedder is None:
            embedder = _load_sentence_transformer(embedding_model, local_files_only=False)
    if embedder is None:
        raise RuntimeError(
            "sentence-transformers не установлен или embedding-модель недоступна. "
            "Установи зависимость: python3 -m pip install sentence-transformers"
        )

    passages = [f"passage: {text}" for text in texts]
    embeddings = _encode_with_model(embedder, passages)
    np.save(target_dir / "embeddings.npy", embeddings)
    return {
        "retriever": "embeddings",
        "feature_count": None,
        "embedding_dim": int(embeddings.shape[1]),
        "embedding_model": embedding_model,
        "files": ["chunks.json", "embeddings.npy", "manifest.json"],
    }


def write_project_vector_index(
    repo_dir: Path,
    chunks: list[KnowledgeChunk] | tuple[KnowledgeChunk, ...],
    index_dir: Path | None = None,
    *,
    retriever: str = "tfidf",
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    embedder: Any | None = None,
) -> dict[str, Any]:
    if retriever not in SUPPORTED_RETRIEVERS:
        raise ValueError(f"Неизвестный retriever: {retriever}. Доступно: {', '.join(sorted(SUPPORTED_RETRIEVERS))}.")
    canonical_retriever = "embeddings" if retriever == "semantic" else retriever
    selected_chunks = [chunk for chunk in chunks if chunk.text.strip()]
    if not selected_chunks:
        raise ValueError("Нет фрагментов для построения RAG-индекса.")

    texts = [f"{chunk.title}\n{chunk.source}\n{chunk.text}" for chunk in selected_chunks]

    target_dir = _rag_index_dir(repo_dir.resolve(), index_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    for stale_file in ("vectorizer.pkl", "matrix.npz", "embeddings.npy", "manifest.json"):
        (target_dir / stale_file).unlink(missing_ok=True)

    _write_chunks_file(target_dir, selected_chunks)
    if canonical_retriever == "embeddings":
        index_info = _write_embedding_index(
            target_dir,
            texts,
            embedding_model=embedding_model,
            embedder=embedder,
        )
    else:
        index_info = _write_tfidf_index(target_dir, texts)

    manifest = {
        **index_info,
        "chunk_count": len(selected_chunks),
        "index_dir": str(target_dir),
    }
    (target_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _load_project_vector_index_cached.cache_clear()
    return manifest


def _vector_index_cache_key(index_dir: Path) -> int:
    manifest_path = index_dir / "manifest.json"
    if not manifest_path.exists():
        return -1
    return manifest_path.stat().st_mtime_ns


@lru_cache(maxsize=4)
def _load_project_vector_index_cached(index_dir_raw: str, cache_key: int):
    del cache_key

    index_dir = Path(index_dir_raw)
    chunks_path = index_dir / "chunks.json"
    manifest_path = index_dir / "manifest.json"
    if not chunks_path.exists() or not manifest_path.exists():
        return None

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    chunks_payload = json.loads(chunks_path.read_text(encoding="utf-8"))
    chunks = tuple(
        KnowledgeChunk(
            source=str(item.get("source", "")),
            title=str(item.get("title", "")),
            text=str(item.get("text", "")),
        )
        for item in chunks_payload
        if str(item.get("text", "")).strip()
    )
    retriever = manifest.get("retriever")

    if retriever in {"embeddings", "semantic"}:
        embeddings_path = index_dir / "embeddings.npy"
        if not embeddings_path.exists():
            return None
        embeddings = _normalize_vectors(np.load(embeddings_path))
        return {"chunks": chunks, "embeddings": embeddings, "manifest": manifest}

    if retriever == "tfidf":
        from scipy import sparse

        vectorizer_path = index_dir / "vectorizer.pkl"
        matrix_path = index_dir / "matrix.npz"
        if not vectorizer_path.exists() or not matrix_path.exists():
            return None
        with vectorizer_path.open("rb") as file:
            vectorizer = pickle.load(file)
        matrix = sparse.load_npz(matrix_path)
        return {"chunks": chunks, "vectorizer": vectorizer, "matrix": matrix, "manifest": manifest}

    return None


def load_project_vector_index(repo_dir: Path, index_dir: Path | None = None):
    target_dir = _rag_index_dir(repo_dir.resolve(), index_dir)
    cache_key = _vector_index_cache_key(target_dir)
    if cache_key < 0:
        return None
    try:
        return _load_project_vector_index_cached(str(target_dir), cache_key)
    except Exception:
        return None


def retrieve_project_context_from_vector_index(
    repo_dir: Path,
    question: str,
    top_k: int = 6,
) -> list[KnowledgeChunk]:
    clean_question = question.strip()
    if not clean_question:
        return []

    index = load_project_vector_index(repo_dir)
    if not index:
        return []

    manifest = index.get("manifest", {})
    if manifest.get("retriever") in {"embeddings", "semantic"}:
        embedding_model = manifest.get("embedding_model") or DEFAULT_EMBEDDING_MODEL
        local_only = _env_true("SAWRAP_EMBEDDING_LOCAL_ONLY", default=True)
        embedder = _load_sentence_transformer(str(embedding_model), local_files_only=local_only)
        if embedder is None:
            return []
        query_vector = _encode_with_model(embedder, [f"query: {clean_question}"])
        scores = np.asarray(index["embeddings"] @ query_vector[0], dtype=np.float32)
    else:
        from sklearn.metrics.pairwise import linear_kernel

        query_vector = index["vectorizer"].transform([clean_question])
        scores = linear_kernel(query_vector, index["matrix"]).ravel()

    if scores.size == 0:
        return []

    ranked_indices = scores.argsort()[::-1]
    result = []
    for chunk_index in ranked_indices:
        score = float(scores[chunk_index])
        if score <= 0:
            break
        chunk = index["chunks"][int(chunk_index)]
        result.append(KnowledgeChunk(chunk.source, chunk.title, chunk.text, score))
        if len(result) >= top_k:
            break
    return result


def _retrieve_project_context_keyword(repo_dir: Path, question: str, top_k: int = 6) -> list[KnowledgeChunk]:
    query_tokens = Counter(_tokenize(question))
    if not query_tokens:
        return []

    query_set = set(query_tokens)
    scored: list[KnowledgeChunk] = []
    for chunk in load_project_knowledge(str(repo_dir.resolve())):
        haystack = f"{chunk.title}\n{chunk.source}\n{chunk.text}"
        chunk_tokens = Counter(_tokenize(haystack))
        if not chunk_tokens:
            continue

        overlap = query_set.intersection(chunk_tokens)
        if not overlap:
            continue

        term_score = sum(query_tokens[token] * (1 + min(chunk_tokens[token], 5) / 5) for token in overlap)
        coverage_score = len(overlap) / max(len(query_set), 1)
        title_hits = len(query_set.intersection(_tokenize(chunk.title)))
        source_hits = len(query_set.intersection(_tokenize(chunk.source)))
        score = term_score + coverage_score * 3 + title_hits * 1.5 + source_hits
        scored.append(KnowledgeChunk(chunk.source, chunk.title, chunk.text, score))

    scored.sort(key=lambda item: item.score, reverse=True)
    return scored[:top_k]


def _overview_context_chunks(repo_dir: Path) -> list[KnowledgeChunk]:
    chunks = []
    seen_sources = set()
    for source in OVERVIEW_SOURCES:
        for chunk in load_project_knowledge(str(repo_dir.resolve())):
            if chunk.source != source or chunk.source in seen_sources:
                continue
            chunks.append(KnowledgeChunk(chunk.source, chunk.title, chunk.text, 0.001))
            seen_sources.add(chunk.source)
            break
    return chunks


def _all_available_context_chunks(repo_dir: Path) -> list[KnowledgeChunk]:
    chunks = list(load_project_knowledge(str(repo_dir.resolve())))
    index = load_project_vector_index(repo_dir)
    if index:
        chunks.extend(index.get("chunks", ()))
    return chunks


def _foundation_context_chunks(repo_dir: Path, question: str, top_k: int = 10) -> list[KnowledgeChunk]:
    del question
    selected = []
    seen = set()

    all_chunks = _all_available_context_chunks(repo_dir)
    for required_source, required_title in FOUNDATION_REQUIRED_SECTIONS:
        for chunk in all_chunks:
            if chunk.source != required_source or required_title.lower() not in chunk.title.lower():
                continue
            key = (chunk.source, chunk.title, chunk.text[:180])
            if key in seen:
                continue
            seen.add(key)
            selected.append(KnowledgeChunk(chunk.source, chunk.title, chunk.text, max(chunk.score, 0.003)))
            break
        if len(selected) >= top_k:
            return selected

    for chunk in all_chunks:
        haystack = f"{chunk.source}\n{chunk.title}\n{chunk.text[:500]}".lower()
        if not any(keyword in haystack for keyword in FOUNDATION_TITLE_KEYWORDS):
            continue
        key = (chunk.source, chunk.title, chunk.text[:180])
        if key in seen:
            continue
        seen.add(key)
        selected.append(KnowledgeChunk(chunk.source, chunk.title, chunk.text, max(chunk.score, 0.002)))
        if len(selected) >= top_k:
            break
    return selected


def _merge_ranked_chunks(*chunk_groups: list[KnowledgeChunk], top_k: int = DEFAULT_RAG_TOP_K) -> list[KnowledgeChunk]:
    merged: list[KnowledgeChunk] = []
    seen = set()
    for chunks in chunk_groups:
        for chunk in chunks:
            key = (chunk.source, chunk.title, chunk.text[:180])
            if key in seen:
                continue
            seen.add(key)
            merged.append(chunk)
            if len(merged) >= top_k:
                return merged
    return merged


def retrieve_project_context(repo_dir: Path, question: str, top_k: int = DEFAULT_RAG_TOP_K) -> list[KnowledgeChunk]:
    vector_chunks = retrieve_project_context_from_vector_index(repo_dir, question, top_k=min(top_k, 6))
    keyword_chunks = _retrieve_project_context_keyword(repo_dir, question, top_k=4)
    foundation_chunks = _foundation_context_chunks(repo_dir, question, top_k=10)
    overview_chunks = _overview_context_chunks(repo_dir)
    return _merge_ranked_chunks(keyword_chunks, foundation_chunks, vector_chunks, overview_chunks, top_k=top_k)


def _context_audit(chunks: list[KnowledgeChunk]) -> str:
    titles = {chunk.title.lower() for chunk in chunks}
    sources = {chunk.source for chunk in chunks}

    facts = []
    if any("модели" in title for title in titles):
        facts.append(
            "В контексте есть конкретные модели: LogisticRegression, SVC, KNeighborsClassifier, "
            "DecisionTreeClassifier, RandomForestClassifier, GradientBoostingClassifier, ElasticNet, "
            "SVR, KNeighborsRegressor, DecisionTreeRegressor, RandomForestRegressor, "
            "GradientBoostingRegressor, KaplanMeierFitter, CoxPHSurvivalAnalysis, SurvivalTree, "
            "RandomSurvivalForest, GradientBoostingSurvivalAnalysis, CRAID, ParallelBootstrapCRAID."
        )
    if any("итоговое ранжирование" in title or "интерпретация" in title for title in titles):
        facts.append(
            "В контексте есть результаты экспериментов: лучшие модели по итоговому ранжированию - "
            "ParallelBootstrapCRAID, CRAID и RandomForestRegressor; ParallelBootstrapCRAID занял "
            "1 место в survival-блоке."
        )
    if any("метрики" in title or "формулы" in title for title in titles):
        facts.append(
            "В контексте есть метрики и формулы: AUC_EVENT, LOGLOSS_EVENT, RMSE_EVENT, RMSE_TIME, "
            "R2_TIME, MAPE_TIME, MEDAPE_TIME, SPEARMAN_TIME, RMSLE_TIME, CI, IBS, AUPRC."
        )
    if any(source.startswith("thesis/") for source in sources):
        facts.append("В контексте есть фрагменты диплома из LaTeX-источников thesis/contents/*.tex.")

    if not facts:
        return "Контроль контекста: специальных разделов с моделями, метриками или результатами не найдено."

    return (
        "Контроль контекста перед ответом:\n"
        + "\n".join(f"- {fact}" for fact in facts)
        + "\nНе создавай раздел ограничений и не утверждай, что перечисленных выше данных нет."
    )


def _build_rag_messages(
    question: str,
    chunks: list[KnowledgeChunk],
    history: list[dict[str, Any]] | None = None,
) -> list[dict[str, str]]:
    context_blocks = []
    for index, chunk in enumerate(chunks, start=1):
        context_blocks.append(
            "\n".join(
                [
                    f"[{index}] Источник: {chunk.source}",
                    f"Раздел: {chunk.title}",
                    chunk.text[:2200],
                ]
            )
        )
    joined_context = "\n\n".join(context_blocks)
    messages = [
        {
            "role": "system",
            "content": (
                "Ты сильный ML/RAG-ассистент проекта SAWrap, а не простой пересказчик chunks. "
                "Отвечай на русском языке уверенно и полезно, как старший ML-инженер на защите проекта. "
                "Проектные факты, числа, названия файлов, датасетов, моделей и метрик бери только из "
                "переданного контекста. Общие знания ML/DS можно использовать для объяснений, выводов "
                "и интуиции, но явно отделяй их от фактов проекта фразами вроде 'из этого следует' "
                "или 'как общее ML-объяснение'. Если точного факта нет в источниках, кратко скажи "
                "об этом внутри основного объяснения без отдельного раздела ограничений. "
                "Не выдумывай новые метрики, результаты экспериментов или ссылки. "
                "Перед ответом проверь найденный контекст на разделы с метриками, формулами "
                "и thesis/contents/2_problem_def.tex. Не утверждай, что метрики или формулы не указаны, "
                "если в источниках есть хотя бы их перечень или определения. "
                "Не выводи разделы с названиями 'Ограничения', 'Ограничения/источники', 'Недостатки' "
                "или похожие блоки, если пользователь сам прямо не попросил ограничения. "
                "Формулы пиши в LaTeX с delimiters \\( ... \\) или \\[ ... \\], например "
                "\\(S(t \\mid X)=P(T>t \\mid X)\\). В конце укажи использованные источники в формате [1], [2]."
            ),
        },
        {
            "role": "user",
            "content": (
                "Найденный RAG-контекст:\n\n"
                f"{joined_context}\n\n"
                f"{_context_audit(chunks)}\n\n"
                "Используй этот контекст как фактическую основу ответа. "
                "Структура ответа: короткий прямой вывод, затем объяснение, затем использованные источники. "
                "Не добавляй отдельный блок ограничений."
            ),
        },
    ]
    messages.extend(_normalize_chat_history(history))
    messages.append({"role": "user", "content": question[:1800]})
    return messages


def _local_fallback_answer(
    question: str,
    chunks: list[KnowledgeChunk],
) -> str:
    if not chunks:
        return "Я не нашел релевантных фрагментов в базе знаний проекта."

    source_parts = [f"[{index}] {chunk.source}" for index, chunk in enumerate(chunks[:3], start=1)]
    sources = ", ".join(source_parts)
    first = " ".join(chunks[0].text.split())
    if len(first) > 520:
        first = first[:517].rstrip() + "..."
    return (
        "OPENROUTER_API_KEY не задан, поэтому показываю найденный локальный контекст без LLM-обобщения.\n\n"
        f"Вопрос: {question}\n\n"
        f"Самый релевантный фрагмент: {first}\n\n"
        f"Источники: {sources}"
    )


def build_project_rag_answer(
    repo_dir: Path,
    question: str,
    history: list[dict[str, Any]] | None = None,
    *,
    api_key: str | None = None,
    model: str | None = None,
    timeout: float = 25.0,
    opener=urlopen,
) -> dict[str, Any]:
    clean_question = question.strip()
    selected_model = model or DEFAULT_OPENROUTER_MODEL
    if not clean_question:
        return {
            "enabled": False,
            "provider": "OpenRouter",
            "model": selected_model,
            "text": None,
            "error": "Вопрос пустой.",
            "sources": [],
        }

    chunks = retrieve_project_context(repo_dir, clean_question, top_k=DEFAULT_RAG_TOP_K)
    sources = [chunk.to_public_dict() for chunk in chunks]

    if not chunks:
        return {
            "enabled": False,
            "provider": "OpenRouter",
            "model": selected_model,
            "text": None,
            "error": "Не найден релевантный контекст в knowledge base.",
            "sources": sources,
        }

    answer = _call_openrouter(
        _build_rag_messages(
            clean_question,
            chunks,
            history,
        ),
        api_key=api_key,
        model=model,
        max_tokens=1100,
        temperature=0.25,
        timeout=timeout,
        opener=opener,
    )
    answer["sources"] = sources

    if not answer.get("text") and not answer.get("enabled"):
        answer["text"] = _local_fallback_answer(clean_question, chunks)
        answer["error"] = None

    return answer
