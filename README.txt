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
