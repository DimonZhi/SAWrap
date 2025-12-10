from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/", name="home")
async def home(request: Request):
    return templates.TemplateResponse(
        "home.html",
        {
            "request": request,
            "messages": [],
        },
    )


@app.get("/goal", name="goal_page")
async def goal_page(request: Request):
    return templates.TemplateResponse(
        "goal.html",
        {
            "request": request,
            "messages": [],
        },
    )


@app.get("/link", name="link_page")
async def link_page(request: Request):
    return templates.TemplateResponse(
        "link.html",
        {
            "request": request,
            "messages": [],
        },
    )


@app.get("/methodology", name="methodology_page")
async def methodology_page(request: Request):
    return templates.TemplateResponse(
        "methodology.html",
        {
            "request": request,
            "messages": [],
        },
    )


@app.get("/example", name="example_page")
async def example_page(request: Request):
    return templates.TemplateResponse(
        "example.html",
        {
            "request": request,
            "messages": [],
        },
    )
