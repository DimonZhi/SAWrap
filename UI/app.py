from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import List, Optional
import json
from fastapi import Form
from survival_wrappers.metrics_sa import eval_classification_model, eval_regression_model, eval_survival_model
from sklearn.model_selection import train_test_split
from pathlib import Path
from survival_wrappers.wrapSA import SAWrapSA, ClassifWrapSA, RegrWrapSA, get_bins
import survivors.datasets as ds
import survivors.constants as cnt
from survival_wrappers.UI.helpers_tables import list_surv_metrics_from_table, get_surv_metrics


app = FastAPI()
BASE_DIR = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")


DATASETS = [
    {"id": "gbsg", "label": "GBSG"},
    {"id": "telco", "label": "Telco"},
    {"id": "pbc", "label": "PBC"},
    {"id": "rott2", "label": "Rott2"},
    {"id": "smarto", "label": "Smarto"},
    {"id": "support2", "label": "Support2"},
]

PRESETS = [
  {"id":"cls_acc_roc","preset_task":"classification","label":"Классификация: Accuracy и ROC AUC",
   "x_metric":"accuracy","y_metric":"roc_auc","x_label":"Accuracy","y_label":"ROC AUC"},
  {"id":"reg_rmse_r2","preset_task":"regression","label":"Регрессия: RMSE и R2",
   "x_metric":"rmse","y_metric":"r2","x_label":"RMSE","y_label":"R2"},
  {"id":"surv_ci_ibs","preset_task":"survival","label":"Выживаемость: C-index и IBS",
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
            "plot_data": None,
            "metrics_list": [],
        },
    )


@app.post("/compare", name="compare_models")
async def compare_models(
    request: Request,
    dataset_id: str = Form(...),
    preset_id: str = Form(...),
    model_ids: Optional[List[str]] = Form(None),
    x_metric: Optional[str] = Form(None),
    y_metric: Optional[str] = Form(None),
):
    model_ids = model_ids or []
    preset = next((p for p in PRESETS if p["id"] == preset_id), None)
    preset_task = preset["preset_task"]
    xk = x_metric or preset["x_metric"]
    yk = y_metric or preset["y_metric"]
    selected = {
        "dataset_id": dataset_id,
        "preset_id": preset_id,
        "model_ids": model_ids,
        "x_metric": xk,
        "y_metric": yk,
    }

    
    if preset is None:
        return templates.TemplateResponse("home.html", {
            "request": request, "datasets": DATASETS, "presets": PRESETS, "models": MODELS,
            "selected": selected, "plot_json": None,
            "messages": [{"category":"error","text":"Пресет не найден"}],  
            "metrics_list": [],
        })

    if not model_ids:
        return templates.TemplateResponse("home.html", {
            "request": request, "datasets": DATASETS, "presets": PRESETS, "models": MODELS,
            "selected": selected, "plot_json": None,
            "messages": [{"category":"error","text":"Выбери хотя бы одну модель"}], 
            "metrics_list": [],
        })


    load_fn = getattr(ds, f"load_{dataset_id}_dataset", None)
    if load_fn is None:
        return templates.TemplateResponse("home.html", {
            "request": request, "datasets": DATASETS, "presets": PRESETS, "models": MODELS,
            "selected": selected, "plot_json": None,
            "messages": [{"category":"error","text":f"Нет загрузчика датасета: load_{dataset_id}_dataset()"}],
            "metrics_list": [],
        })
    X, y, features, categ, sch_nan = load_fn()
    errors = []
    metrics_list = []
    labels = []
    values_by_label = {}
    metrics_set = set()
    for mid in model_ids:
        mcfg = next((m for m in MODELS if m["id"] == mid), None)
        task = mcfg["task"]
        if mcfg is None:
            continue

        try:
            if task in ("classification", "regression"):
                Cls = import_class(mid)
                raw = Cls()
                model = wrap_model(raw, task)
                model.fit(X, y)

                if preset_task == "classification":
                    md = eval_classification_model(model, X, y)
                elif preset_task == "regression":
                    md = eval_regression_model(model, X, y["time"])
                else:
                    X_tr, X_tst, y_tr, y_tst = train_test_split(X, y, test_size=0.2, random_state=42)
                    bins = get_bins(y, 128)
                    md = eval_survival_model(model, X_tr, y_tr, X_tst, y_tst, bins)

                values_by_label.setdefault(mcfg["label"], {})
                for k, v in md.items():
                    if v is None:
                        continue
                    try:
                        fv = float(v)
                    except Exception:
                        continue
                    values_by_label[mcfg["label"]][f"{k}_mean"] = fv
                    metrics_set.add(k)

                labels.append(mcfg["label"])
                metrics_list = sorted(metrics_set)
            else:
                
                surv_cfgs = [next((m for m in MODELS if m["id"] == mid2), None) for mid2 in model_ids]
                surv_cfgs = [m for m in surv_cfgs if m is not None and m.get("task") == "survival"]
                vx, vy, _ = get_surv_metrics(
                    base_dir=BASE_DIR,
                    dataset_id=dataset_id,
                    X_tr=X,
                    y_tr=y,
                    model_cfgs=surv_cfgs,
                    model_label=mcfg["label"],
                    x_metric=xk,
                    y_metric=yk,
                )
                metrics_list, df = list_surv_metrics_from_table(BASE_DIR, dataset_id)
                r = df[df["method"].astype(str) == str(mcfg["label"])]
                d = r.iloc[0].to_dict()
                values_by_label.setdefault(mcfg["label"], {})
                for k, v in d.items():
                    if isinstance(k, str) and k.endswith("_mean"):
                        try:
                            values_by_label[mcfg["label"]][k] = float(v)
                        except Exception:
                            pass
                labels.append(mcfg["label"])


        except Exception as e:
            errors.append(f"{mcfg['label']}: {type(e).__name__}: {e}")

    if not labels:
        return templates.TemplateResponse("home.html", {
            "request": request, "datasets": DATASETS, "presets": PRESETS, "models": MODELS,
            "selected": selected, "plot_json": None,
            "messages": [{"category":"error","text":"Не построено ни одной точки: " + (" | ".join(errors[:3]) if errors else "")}],
            "metrics_list": [],
        })
    print(values_by_label)
    plot_data = {
        "labels": labels,
        "metrics": metrics_list,        
        "values": values_by_label,
        "x_metric": xk,
        "y_metric": yk,
    }

    msgs = [{"category":"error","text":" | ".join(errors[:3])}] if errors else []

    return templates.TemplateResponse("home.html", {
        "request": request, "datasets": DATASETS, "presets": PRESETS, "models": MODELS,
        "selected": selected, "plot_data": plot_data,
        "messages": msgs, "metrics_list": metrics_list,
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
