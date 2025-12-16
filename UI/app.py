from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import List, Optional
import json
from fastapi import Form
from sklearn.model_selection import train_test_split
from survival_wrappers.metrics_sa import eval_classification_model,eval_regression_model, eval_survival_model
from pathlib import Path
from survival_wrappers.wrapSA import get_bins, SAWrapSA, ClassifWrapSA, RegrWrapSA
from survivors.datasets import load_gbsg_dataset
import survivors.constants as cnt


app = FastAPI()
BASE_DIR = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")


DATASETS = [
    {"id": "gbsg", "label": "GBSG"},
]
PRESETS = [
  {"id":"cls_acc_roc","task":"classification","label":"Классификация: Accuracy и ROC AUC",
   "x_metric":"accuracy","y_metric":"roc_auc","x_label":"Accuracy","y_label":"ROC AUC"},
  {"id":"reg_rmse_r2","task":"regression","label":"Регрессия: RMSE и R2",
   "x_metric":"rmse","y_metric":"r2","x_label":"RMSE","y_label":"R2"},
  {"id":"surv_ci_ibs","task":"survival","label":"Выживаемость: C-index и IBS",
   "x_metric":"c_index","y_metric":"ibs_remain","x_label":"C-index","y_label":"IBS"},
]


MODELS = [
  # classification (sklearn)
  {"id":"sklearn.linear_model.LogisticRegression","label":"LogisticRegression","lib":"sklearn","task":"classification"},
  {"id":"sklearn.ensemble.RandomForestClassifier","label":"RandomForestClassifier","lib":"sklearn","task":"classification"},

  # regression (sklearn)
  {"id":"sklearn.ensemble.RandomForestRegressor","label":"RandomForestRegressor","lib":"sklearn","task":"regression"},

  # survival (lifelines)
  {"id":"lifelines.CoxPHFitter","label":"CoxPHFitter","lib":"lifelines","task":"survival"},
  {"id":"lifelines.WeibullAFTFitter","label":"WeibullAFTFitter","lib":"lifelines","task":"survival"},

  # survival (survivors)
  {"id":"survivors.tree.CRAID","label":"CRAID","lib":"survivors","task":"survival"},
  {"id":"survivors.ensemble.ParallelBootstrapCRAID","label":"ParallelBootstrapCRAID","lib":"survivors","task":"survival"},
]


import importlib

def import_class(path: str):
    module_name, class_name = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)

def wrap_model(raw_model, task: str):
    if task == "classification":
        return ClassifWrapSA(raw_model)
    if task == "regression":
        return RegrWrapSA(raw_model)
    if task == "survival":
        return SAWrapSA(raw_model)
    raise ValueError(task)


@app.get("/", name="home")
async def home(request: Request):
    return templates.TemplateResponse(
        "home.html",
        {
            "request": request,
            "messages": [],
            "datasets": DATASETS,
            "presets": PRESETS,
            "models": MODELS,
            "selected": None,
            "plot_json": None,
        },
    )
@app.post("/compare", name="compare_models")
async def compare_models(
    request: Request,
    dataset_id: str = Form(...),
    preset_id: str = Form(...),
    model_ids: Optional[List[str]] = Form(None),
):
    model_ids = model_ids or []
    selected = {"dataset_id": dataset_id, "preset_id": preset_id, "model_ids": model_ids}

    preset = next((p for p in PRESETS if p["id"] == preset_id), None)
    if preset is None:
        return templates.TemplateResponse("home.html", {
            "request": request, "datasets": DATASETS, "presets": PRESETS, "models": MODELS,
            "selected": selected, "plot_json": None,
            "messages": [{"category":"error","text":"Пресет не найден"}],
        })

    if not model_ids:
        return templates.TemplateResponse("home.html", {
            "request": request, "datasets": DATASETS, "presets": PRESETS, "models": MODELS,
            "selected": selected, "plot_json": None,
            "messages": [{"category":"error","text":"Выбери хотя бы одну модель"}],
        })

    task = preset["task"]
    xk, yk = preset["x_metric"], preset["y_metric"]

    X, y, features, categ, sch_nan = load_gbsg_dataset()
    X_tr, X_tst, y_tr, y_tst = train_test_split(X, y, test_size=0.25, random_state=42)
    bins = None
    if task == "survival":
        bins = cnt.get_bins(time=y_tr[cnt.TIME_NAME], cens=y_tr[cnt.CENS_NAME])

    xs, ys, labels = [], [], []
    errors = []

    for mid in model_ids:
        mcfg = next((m for m in MODELS if m["id"] == mid), None)
        if mcfg is None:
            continue

        try:
            Cls = import_class(mid)
            raw = Cls()  # если параметры нужны — добавь mcfg.get("params", {})
            model = wrap_model(raw, task)

            model.fit(X_tr, y_tr)

            if task == "classification":
                md = eval_classification_model(model, X_tst, y_tst)
            elif task == "regression":
                md = eval_regression_model(model, X_tst, y_tst["time"])
                print(md)
            else:
                md = eval_survival_model(model, X_tr, y_tr, X_tst, y_tst, bins=bins)

            vx = md.get(xk)
            vy = md.get(yk)
            if vx is None or vy is None:
                errors.append(f"{mcfg['label']}: нет метрик {xk}/{yk}")
                continue

            xs.append(float(vx))
            ys.append(float(vy))
            labels.append(mcfg["label"])

        except Exception as e:
            errors.append(f"{mcfg['label']}: {type(e).__name__}: {e}")

    if not xs:
        return templates.TemplateResponse("home.html", {
            "request": request, "datasets": DATASETS, "presets": PRESETS, "models": MODELS,
            "selected": selected, "plot_json": None,
            "messages": [{"category":"error","text":"Не построено ни одной точки: " + (" | ".join(errors[:3]) if errors else "")}],
        })

    plot_data = {
        "x": xs,
        "y": ys,
        "labels": labels,
        "x_metric": preset["x_label"],
        "y_metric": preset["y_label"],
    }

    msgs = [{"category":"error","text":" | ".join(errors[:3])}] if errors else []

    return templates.TemplateResponse("home.html", {
        "request": request, "datasets": DATASETS, "presets": PRESETS, "models": MODELS,
        "selected": selected, "plot_json": json.dumps(plot_data, ensure_ascii=False),
        "messages": msgs,
    })




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
