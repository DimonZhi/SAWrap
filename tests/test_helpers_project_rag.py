import json
from pathlib import Path

import numpy as np

import UI.helpers_project_rag as project_rag
from UI.helpers_project_rag import (
    build_project_rag_answer,
    load_project_knowledge,
    load_thesis_knowledge,
    retrieve_project_context,
    write_project_vector_index,
)


class _FakeResponse:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self):
        return b'{"choices":[{"message":{"content":"RAG answer from project context."}}]}'


class _FakeEmbedder:
    def encode(self, texts, **kwargs):
        vectors = []
        for text in texts:
            lower = text.lower()
            vectors.append(
                [
                    1.0 if "parallelbootstrapcraid" in lower else 0.0,
                    1.0 if "survival" in lower or "выживания" in lower else 0.0,
                    1.0 if "openrouter" in lower else 0.0,
                ]
            )
        return np.asarray(vectors, dtype=np.float32)


def _write_knowledge(repo_dir: Path):
    knowledge_dir = repo_dir / "knowledge"
    knowledge_dir.mkdir()
    (knowledge_dir / "project_context.md").write_text(
        "# SAWrap\n\n"
        "SAWrap сравнивает classification, regression и survival analysis через функцию выживания.\n"
        "Лучшая итоговая модель в экспериментах - ParallelBootstrapCRAID.",
        encoding="utf-8",
    )


def test_retrieve_project_context_finds_knowledge_file(tmp_path: Path):
    _write_knowledge(tmp_path)

    chunks = retrieve_project_context(tmp_path, "Какая модель лучшая в экспериментах?")

    assert chunks
    assert chunks[0].source == "knowledge/project_context.md"
    assert "ParallelBootstrapCRAID" in chunks[0].text


def test_build_project_rag_answer_sends_retrieved_context_to_openrouter(tmp_path: Path):
    _write_knowledge(tmp_path)
    captured = {}

    def fake_opener(request, timeout, context):
        captured["payload"] = request.data.decode("utf-8")
        captured["timeout"] = timeout
        captured["context"] = context
        return _FakeResponse()

    answer = build_project_rag_answer(
        tmp_path,
        "Что делает SAWrap?",
        api_key="test-key",
        model="test/model",
        timeout=4,
        opener=fake_opener,
    )

    assert answer["text"] == "RAG answer from project context."
    assert answer["sources"][0]["source"] == "knowledge/project_context.md"
    assert captured["timeout"] == 4
    assert captured["context"] is not None

    payload = json.loads(captured["payload"])
    message_text = "\n".join(message["content"] for message in payload["messages"])
    assert "Что делает SAWrap?" in message_text
    assert "ParallelBootstrapCRAID" in message_text
    assert "knowledge/project_context.md" in message_text
    assert "Общие знания ML/DS можно использовать" in message_text
    assert "\\(S(t \\mid X)=P(T>t \\mid X)\\)" in message_text


def test_build_project_rag_answer_has_local_fallback_without_key(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    _write_knowledge(tmp_path)

    answer = build_project_rag_answer(tmp_path, "Что такое SAWrap?")

    assert answer["text"]
    assert "OPENROUTER_API_KEY" in answer["text"]
    assert answer["sources"]


def test_build_project_rag_answer_includes_pinned_result_context(tmp_path: Path):
    _write_knowledge(tmp_path)
    captured = {}

    def fake_opener(request, timeout, context):
        captured["payload"] = request.data.decode("utf-8")
        return _FakeResponse()

    answer = build_project_rag_answer(
        tmp_path,
        "Почему выбрана эта модель?",
        pinned_context=(
            "Датасет: toy\n"
            "Задача: Анализ выживаемости\n"
            "Рекомендованная модель: StrongSurvivalModel\n"
            "CI: 0.91, IBS: 0.08"
        ),
        pinned_title="Текущий результат: toy · Анализ выживаемости",
        api_key="test-key",
        opener=fake_opener,
    )

    assert answer["text"] == "RAG answer from project context."
    assert answer["sources"][0]["source"] == "current_result"
    assert answer["sources"][0]["title"] == "Текущий результат: toy · Анализ выживаемости"

    payload = json.loads(captured["payload"])
    message_text = "\n".join(message["content"] for message in payload["messages"])
    assert "current_result" in message_text
    assert "StrongSurvivalModel" in message_text
    assert "knowledge/project_context.md" in message_text


def test_retrieve_project_context_uses_vector_index(tmp_path: Path):
    _write_knowledge(tmp_path)
    chunks = load_project_knowledge(str(tmp_path))
    manifest = write_project_vector_index(tmp_path, chunks)

    result = retrieve_project_context(tmp_path, "единое представление survival analysis")

    assert manifest["retriever"] == "tfidf"
    assert manifest["chunk_count"] == len(chunks)
    assert (tmp_path / "UI" / "rag_index" / "chunks.json").exists()
    assert result
    assert result[0].source == "knowledge/project_context.md"
    assert 0 < result[0].score <= 1


def test_load_thesis_knowledge_reads_latex_sources(tmp_path: Path):
    thesis_dir = tmp_path / "thesis"
    contents_dir = thesis_dir / "contents"
    contents_dir.mkdir(parents=True)
    (contents_dir / "1_intro.tex").write_text(
        r"\section{Введение} Научная новизна связана с функцией выживания.",
        encoding="utf-8",
    )

    chunks = load_thesis_knowledge(tmp_path, thesis_dir)

    assert chunks
    assert chunks[0].source == "thesis/contents/1_intro.tex"
    assert "Научная новизна" in chunks[0].text


def test_retrieve_project_context_uses_embedding_index(tmp_path: Path, monkeypatch):
    _write_knowledge(tmp_path)
    chunks = load_project_knowledge(str(tmp_path))
    manifest = write_project_vector_index(
        tmp_path,
        chunks,
        retriever="embeddings",
        embedding_model="fake/e5",
        embedder=_FakeEmbedder(),
    )
    monkeypatch.setattr(project_rag, "_load_sentence_transformer", lambda *args, **kwargs: _FakeEmbedder())

    result = retrieve_project_context(tmp_path, "Какая модель ParallelBootstrapCRAID лучше для survival?")

    assert manifest["retriever"] == "embeddings"
    assert manifest["embedding_model"] == "fake/e5"
    assert manifest["embedding_dim"] == 3
    assert (tmp_path / "UI" / "rag_index" / "embeddings.npy").exists()
    assert result
    assert result[0].source == "knowledge/project_context.md"
    assert 0 < result[0].score <= 1
