# Разработка и инженерия

Раздел описывает инженерную часть проекта: Git, Docker, CI, MLOps/DevOps, качество кода и ML-пайплайны.

## Что реализовано

- GitHub-репозиторий хранит код проекта, конфигурацию деплоя и тесты.
- `Dockerfile` позволяет собрать приложение в воспроизводимый контейнер.
- `.github/workflows/ci.yml` запускает unit-тесты, компиляцию ключевых модулей и Docker build.
- `.gitignore` и `.dockerignore` защищают локальные артефакты, `.env` и RAG-индекс от случайной публикации.
- `requirements.txt`, `requirements-test.txt` и `requirements-embeddings.txt` разделяют runtime, test и embedding-зависимости.
- Примеры `deploy/` показывают запуск через Docker, systemd и nginx.

## Архитектура

- `UI/app.py` - FastAPI routes, формы и страницы.
- `UI/helpers_tables.py` - чтение таблиц метрик.
- `UI/helpers_leaderboard.py` - загрузка итоговых лидербордов.
- `UI/helpers_ai_advice.py` - AI-интерпретатор результатов через OpenRouter.
- `UI/helpers_project_rag.py` - RAG-ассистент по проекту, knowledge-базе, коду и диплому.
- `scripts/build_rag_index.py` - сборка локального semantic/TF-IDF индекса.

## ML-пайплайн

Данные и модели проходят цепочку: survival dataset -> адаптеры моделей -> расчет метрик -> таблицы `UI/tables` -> лидерборды -> визуализация на сайте -> AI/RAG-интерпретация.

## Почему это важно

Проект не является одиночным ноутбуком. Он упакован как инженерный ML-сервис: есть backend, UI, Docker, CI, тесты, деплой, предрассчитанные результаты и AI-слой.
