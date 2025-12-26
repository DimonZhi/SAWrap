from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import List, Optional
import json
from fastapi import Form
from sklearn.model_selection import train_test_split
from survival_wrappers.metrics_sa import eval_classification_model, eval_regression_model
from pathlib import Path
from survival_wrappers.wrapSA import SAWrapSA, ClassifWrapSA, RegrWrapSA
import survivors.datasets as ds
import survivors.constants as cnt
from survival_wrappers.UI.helpers_tables import  get_surv_metrics


app = FastAPI()
BASE_DIR = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")


DATASETS = [
    {"id": "gbsg", "label": "GBSG"},
    {"id": "telco", "label": "Telco"},
    {"id": "wuhan", "label": "Wuhan"},
]

PRESETS = [
  {"id":"cls_acc_roc","task":"classification","label":"Классификация: Accuracy и ROC AUC",
   "x_metric":"accuracy","y_metric":"roc_auc","x_label":"Accuracy","y_label":"ROC AUC"},
  {"id":"reg_rmse_r2","task":"regression","label":"Регрессия: RMSE и R2",
   "x_metric":"rmse","y_metric":"r2","x_label":"RMSE","y_label":"R2"},
  {"id":"surv_ci_ibs","task":"survival","label":"Выживаемость: C-index и IBS",
   "x_metric":"CI","y_metric":"IBS_REMAIN","x_label":"C-index","y_label":"IBS"},
]


MODELS = [
  {"id":"sklearn.linear_model.LogisticRegression","label":"LogisticRegression","lib":"sklearn","task":"classification"},
  {"id":"sklearn.ensemble.RandomForestClassifier","label":"RandomForestClassifier","lib":"sklearn","task":"classification"},
  {"id":"sklearn.ensemble.RandomForestRegressor","label":"RandomForestRegressor","lib":"sklearn","task":"regression"},
  #{"id":"lifelines.CoxPHFitter","label":"CoxPHFitter","lib":"lifelines","task":"survival"},
  #{"id":"lifelines.WeibullAFTFitter","label":"WeibullAFTFitter","lib":"lifelines","task":"survival"},
  #{"id":"survivors.tree.CRAID","label":"CRAID","lib":"survivors","task":"survival", "param_grid": {"criterion": "wilcoxon", "depth": 2, "min_samples_leaf": 0.1, "signif": 0.05, "leaf_model": "base"}},
  {"id":"survivors.ensemble.ParallelBootstrapCRAID","label":"ParallelBootstrapCRAID","lib":"survivors","task":"survival"},
  {"id":"sksurv.linear_model.CoxPHSurvivalAnalysis","label":"CoxPHSurvivalAnalysis","lib":"sksurv","task":"survival", "param_grid": {"alpha":[100, 10, 1, 0.1, 0.01, 0.001],"ties":["breslow"]}},
  {"id":"sksurv.ensemble.RandomSurvivalForest","label":"RandomSurvivalForest","lib":"sksurv","task":"survival", "param_grid": {"n_estimators":[50],"max_depth":[None,20],"min_samples_leaf":[0.001,0.01,0.1,0.25],"random_state":[123]}},
  {"id":"sksurv.tree.SurvivalTree","label":"SurvivalTree","lib":"sksurv","task":"survival", "param_grid": {"max_depth":[None,20],"min_samples_leaf":[1, 10, 20],"max_features":[None,"sqrt"],"random_state":[123]}},
  {"id":"sksurv.ensemble.ComponentwiseGradientBoostingSurvivalAnalysis","label":"ComponentwiseGradientBoostingSurvivalAnalysis","lib":"sksurv","task":"survival", "param_grid": {"loss":["coxph"],"learning_rate":[0.01, 0.05, 0.1, 0.5],"n_estimators":[30,50],"subsample":[0.7, 1.0],"dropout_rate":[0.0, 0.1, 0.5],"random_state":[123]}},
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
    load_fn = getattr(ds, f"load_{dataset_id}_dataset", None)
    if load_fn is None:
        return templates.TemplateResponse("home.html", {
            "request": request, "datasets": DATASETS, "presets": PRESETS, "models": MODELS,
            "selected": selected, "plot_json": None,
            "messages": [{"category":"error","text":f"Нет загрузчика датасета: load_{dataset_id}_dataset()"}],
        })
    X, y, features, categ, sch_nan = load_fn()

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
            vx = None
            vy = None
            if task == "classification" or task == "regression":
                Cls = import_class(mid)
                raw = Cls()
                model = wrap_model(raw, task)
                model.fit(X_tr, y_tr)

                if task == "classification":
                    md = eval_classification_model(model, X_tst, y_tst)
                else:
                    md = eval_regression_model(model, X_tst, y_tst["time"])

                vx = md.get(xk)
                vy = md.get(yk)

            else:
                surv_cfgs = [next((m for m in MODELS if m["id"] == mid2), None) for mid2 in model_ids]
                surv_cfgs = [m for m in surv_cfgs if m is not None and m.get("task") == "survival"]

                vx, vy, _ = get_surv_metrics(
                    base_dir=BASE_DIR,
                    dataset_id=dataset_id,
                    X_tr=X,
                    y_tr=y,
                    bins=bins,
                    model_cfgs=surv_cfgs,
                    model_label=mcfg["label"],
                    x_metric=xk,
                    y_metric=yk,
                )
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
    return templates.TemplateResponse("goal.html", {"request": request, "messages": []})


@app.get("/link", name="link_page")
async def link_page(request: Request):
    return templates.TemplateResponse("link.html", {"request": request, "messages": []})


@app.get("/methodology", name="methodology_page")
async def methodology_page(request: Request):
    return templates.TemplateResponse("methodology.html", {"request": request, "messages": []})


@app.get("/example", name="example_page")
async def example_page(request: Request):
    return templates.TemplateResponse("example.html", {"request": request, "messages": []})
