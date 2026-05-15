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
