# Knowledge Base для RAG

Эта папка содержит сжатую базу знаний проекта SAWrap. Ее цель - дать LLM контекст о дипломной работе, коде, экспериментах, продуктовой идее и практических вопросах по системе без fine-tuning модели.

## Источники

- Дипломный LaTeX-проект: `/Users/dimonzhi/Downloads/ДипломML_SA-3`
- Код проекта: `/Users/dimonzhi/Documents/proga/SAWrap`
- Таблицы результатов: `UI/tables/*.xlsx`
- Веб-интерфейс и AI-интерпретатор: `UI/app.py`, `UI/helpers_ai_advice.py`, `UI/templates/home.html`

## Что индексировать в RAG

- `knowledge/project_context.md`
- `knowledge/thesis_summary.md`
- `knowledge/experiments_summary.md`
- `knowledge/product_context.md`
- `knowledge/engineering_evidence.md`
- `knowledge/data_science_evidence.md`
- `knowledge/ai_usage_evidence.md`
- `knowledge/product_evidence.md`
- `knowledge/project_faq.md`
- `README.txt`
- ключевые файлы кода: `UI/app.py`, `UI/helpers_ai_advice.py`, `rank.py`, `wrapSA.py`, `run_many_server.py`
- итоговые таблицы из `UI/tables`, особенно `leaderboards_by_task.xlsx`

## Что не индексировать

- `.env` и любые API-ключи
- `.git`
- `.ml` / виртуальные окружения
- большие временные файлы и кэши
- приватные данные, если они появятся в проекте

## Роль AI-ассистента

AI-ассистент должен отвечать как исследовательский помощник по проекту SAWrap. Он объясняет:

- почему survival analysis важен для терминальных событий;
- как классификация, регрессия и анализ выживаемости приводятся к единому виду;
- какие модели сравнивались и по каким метрикам;
- почему выбранная модель подходит под выбранный датасет и задачу;
- как проект выглядит с точки зрения разработки, MLOps, продукта и исследовательской методологии.

Если в базе знаний нет ответа, ассистент должен прямо сказать, что данных недостаточно, а не выдумывать результат.
