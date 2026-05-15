SAWrap

Локальный запуск:
1. перейти в директорию уровнем выше репозитория
2. запустить:
SAWRAP_SKIP_MISSING_RECALC=1 python3 -m uvicorn SAWrap.UI.app:app --host 0.0.0.0 --port 8000 --reload

Подготовка к GitHub:
1. проверить remote:
git remote -v
2. добавить все файлы:
git add .
3. сделать коммит:
git commit -m "Prepare GitHub and Docker deployment"
4. отправить в GitHub:
git push origin HEAD

Что добавлено для деплоя:
- .gitignore
- .dockerignore
- requirements.txt
- Dockerfile
- deploy/docker-run.example.sh
- deploy/nginx-sawrap-subdomain.conf.example
- deploy/nginx-sawrap-location.conf.example

Локальная проверка Docker:
1. собрать образ:
docker build -t sawrap:local .
2. запустить:
docker run --rm -p 8000:8000 -e SAWRAP_SKIP_MISSING_RECALC=1 sawrap:local
3. открыть:
http://127.0.0.1:8000

Проверки качества:
1. установить тестовые зависимости:
python3 -m pip install -r requirements-test.txt
2. запустить unit-тесты:
python3 -m pytest -q
3. проверить компиляцию ключевых модулей:
python3 -m compileall rank.py UI/helpers_leaderboard.py UI/helpers_ai_advice.py

CI:
- workflow лежит в .github/workflows/ci.yml
- запускается на push и pull request
- job Unit tests ставит requirements-test.txt, компилирует rank.py, UI/helpers_leaderboard.py и UI/helpers_ai_advice.py, запускает pytest
- job Docker build собирает production-образ командой docker build -t sawrap:ci .

AI-интерпретатор:
- форма находится на главной странице
- выбирает датасет и задачу: классификация события, прогноз времени или анализ выживаемости
- читает предрассчитанную таблицу UI/tables/<dataset>.xlsx
- считает итоговую оценку модели по направлению метрик и объясняет, почему выбран топовый метод
- если задан OPENROUTER_API_KEY, дополнительно получает текстовую интерпретацию от OpenRouter
- модель по умолчанию: openai/gpt-4o-mini
- можно заменить модель через OPENROUTER_MODEL

RAG-ассистент по проекту:
- форма находится на главной странице под AI-интерпретатором
- ищет релевантные фрагменты в knowledge/*.md, README, ключевых файлах кода и таблицах UI/tables
- отправляет найденный контекст во внешнюю LLM через OpenRouter
- отвечает на вопросы по диплому, архитектуре, экспериментам, продуктовой части и методологии проекта
- показывает источники, на которые опирался ответ
- если OPENROUTER_API_KEY не задан, показывает найденный локальный контекст без LLM-обобщения

Векторный RAG-индекс:
- строится один раз локально и сохраняется в UI/rag_index
- по умолчанию использует локальные semantic embeddings через sentence-transformers
- fallback-режим использует TF-IDF-векторизацию через scikit-learn без внешних embedding API
- включает knowledge/*.md, README, ключевые .py файлы, таблицы UI/tables и LaTeX-диплом из SAWRAP_THESIS_DIR
- если индекс есть, RAG сначала ищет по векторному индексу; если индекса нет, использует обычный keyword fallback
- UI/rag_index добавлен в .gitignore, чтобы случайно не залить полный текст диплома в GitHub

Установить зависимости для semantic embeddings:
python3 -m pip install -r requirements-embeddings.txt

Собрать индекс:
python3 scripts/build_rag_index.py

Собрать TF-IDF fallback-индекс без sentence-transformers:
python3 scripts/build_rag_index.py --retriever tfidf

Если диплом лежит в другой папке:
SAWRAP_THESIS_DIR="/path/to/ДипломML_SA-3" python3 scripts/build_rag_index.py

Модель embeddings по умолчанию:
intfloat/multilingual-e5-small

Для Docker с embeddings: установить requirements-embeddings.txt в образ или заранее адаптировать Dockerfile, затем собрать индекс и docker build. Папка UI/rag_index попадет в образ вместе с UI.

OpenRouter:
0. пример переменных лежит в .env.example, реальный .env уже игнорируется git
1. локально:
export OPENROUTER_API_KEY="твой_ключ"
export OPENROUTER_MODEL="openai/gpt-4o-mini"
SAWRAP_SKIP_MISSING_RECALC=1 python3 -m uvicorn SAWrap.UI.app:app --host 0.0.0.0 --port 8000 --reload
2. в Docker:
docker run --rm -p 8000:8000 -e SAWRAP_SKIP_MISSING_RECALC=1 -e OPENROUTER_API_KEY="твой_ключ" -e OPENROUTER_MODEL="openai/gpt-4o-mini" sawrap:local

Рекомендуемый деплой на сервер:
1. клонировать репозиторий:
git clone https://github.com/DimonZhi/SAWrap /srv/survival_wrappers
2. перейти в проект:
cd /srv/survival_wrappers
3. запустить пример вручную или адаптировать:
deploy/docker-run.example.sh
4. наружу публиковать через nginx, а не через прямой порт контейнера

Ресурсные ограничения для маленького VPS:
- CPU: 0.40
- RAM: 800m
- pids-limit: 128
- workers: 1
- SAWRAP_SKIP_MISSING_RECALC=1

Публикация наружу:
- лучший вариант: отдельный поддомен через deploy/nginx-sawrap-subdomain.conf.example
- вариант под путём /sawrap/: deploy/nginx-sawrap-location.conf.example
- если публикуешь под /sawrap, добавь SAWRAP_ROOT_PATH=/sawrap
