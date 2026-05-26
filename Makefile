PYTHON ?= python3
HOST ?= 0.0.0.0
PORT ?= 8000
APP ?= UI.app:app
IMAGE ?= sawrap:local
MPLCONFIGDIR ?= /private/tmp
RAG_RETRIEVER ?= embeddings
SKIP_RECALC ?= 1

.PHONY: help install install-test run test smoke compile ci rank rag rag-tfidf docker-build

help:
	@printf '%s\n' 'SAWrap engineering commands'
	@printf '%s\n' ''
	@printf '%s\n' '  make install       Install runtime dependencies'
	@printf '%s\n' '  make install-test  Install test dependencies'
	@printf '%s\n' '  make run           Run local FastAPI app with reload'
	@printf '%s\n' '  make test          Run unit tests'
	@printf '%s\n' '  make smoke         Check that key web pages render'
	@printf '%s\n' '  make compile       Compile checked Python modules'
	@printf '%s\n' '  make ci            Run compile, unit tests and smoke test'
	@printf '%s\n' '  make rank          Rebuild leaderboard tables'
	@printf '%s\n' '  make rag           Rebuild RAG index with semantic embeddings'
	@printf '%s\n' '  make rag-tfidf     Rebuild RAG index with TF-IDF fallback'
	@printf '%s\n' '  make docker-build  Build local Docker image'

install:
	$(PYTHON) -m pip install -r requirements.txt

install-test:
	$(PYTHON) -m pip install -r requirements-test.txt

run:
	SAWRAP_SKIP_MISSING_RECALC=$(SKIP_RECALC) MPLCONFIGDIR=$(MPLCONFIGDIR) $(PYTHON) -m uvicorn $(APP) --host $(HOST) --port $(PORT) --reload

test:
	MPLCONFIGDIR=$(MPLCONFIGDIR) $(PYTHON) -m pytest -q

smoke:
	SAWRAP_SKIP_MISSING_RECALC=1 MPLCONFIGDIR=$(MPLCONFIGDIR) $(PYTHON) scripts/smoke_test.py

compile:
	MPLCONFIGDIR=$(MPLCONFIGDIR) $(PYTHON) -m compileall UI tests scripts rank.py

ci: compile test smoke

rank:
	MPLCONFIGDIR=$(MPLCONFIGDIR) $(PYTHON) rank.py

rag:
	MPLCONFIGDIR=$(MPLCONFIGDIR) $(PYTHON) scripts/build_rag_index.py --retriever $(RAG_RETRIEVER) $(if $(SAWRAP_THESIS_DIR),--thesis-dir "$(SAWRAP_THESIS_DIR)",)

rag-tfidf:
	$(MAKE) rag RAG_RETRIEVER=tfidf

docker-build:
	docker build -t $(IMAGE) .
