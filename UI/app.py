from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import List, Optional
import json
import os
from fastapi import Form
from sklearn.model_selection import train_test_split
from pathlib import Path
from survivors.external import SAWrapSA, ClassifWrapSA, RegrWrapSA
import survivors.datasets as ds
import survivors.constants as cnt
from .helpers_ai_advice import AI_TASKS, build_ai_advice
from .helpers_tables import list_surv_metrics_from_table, get_surv_metrics
from .helpers_leaderboard import load_leaderboard_images, load_overall_leaderboard_rows


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_root_path(name: str = "SAWRAP_ROOT_PATH") -> str:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return ""
    if not raw.startswith("/"):
        raw = "/" + raw
    return raw.rstrip("/")


DISABLE_MISSING_RECALC = _env_flag("SAWRAP_SKIP_MISSING_RECALC", default=False)
ROOT_PATH = _env_root_path()


app = FastAPI(root_path=ROOT_PATH)
BASE_DIR = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
app.mount("/images", StaticFiles(directory=BASE_DIR / "images"), name="images")
templates = Jinja2Templates(directory=BASE_DIR / "templates")


DATASETS = [
    {"id": "actg", "label": "ACTG"},
    {"id": "gbsg", "label": "GBSG"},
    {"id": "pbc", "label": "PBC"},
    {"id": "rott2", "label": "Rott2"},
    {"id": "smarto", "label": "Smarto"},
    {"id": "framingham", "label": "Framingham"},
    {"id": "support2", "label": "Support2"},

]

PRESETS = [
  {"id":"cls_auc_logloss","preset_task":"classification","label":"Классификация: AUC и LogLoss",
   "x_metric":"AUC_EVENT","y_metric":"LOGLOSS_EVENT","x_label":"AUC(event)","y_label":"LogLoss(event)"},
  {"id":"reg_rmse_r2","preset_task":"regression","label":"Регрессия: RMSE и R2",
   "x_metric":"RMSE_TIME","y_metric":"R2_TIME","x_label":"RMSE E[T]","y_label":"R2 E[T]"},
  {"id":"surv_ci_ibs","preset_task":"survival","label":"Выживаемость: C-index и IBS",
   "x_metric":"CI","y_metric":"IBS","x_label":"C-index","y_label":"IBS"},
]



MODELS = [
  # -------- classification --------
  {"id":"sklearn.linear_model.LogisticRegression","label":"LogisticRegression","lib":"sklearn","task":"classification",
   "param_grid":{"penalty":["l2"],"C":[0.01,0.1,1,10],"solver":["liblinear","lbfgs"],"class_weight":[None,"balanced"],"max_iter":[1000]}},

  {"id":"sklearn.neighbors.KNeighborsClassifier","label":"KNeighborsClassifier","lib":"sklearn","task":"classification",
   "param_grid":{"n_neighbors":[5,10,20],"weights":["uniform","distance"]}},

  {"id":"sklearn.tree.DecisionTreeClassifier","label":"DecisionTreeClassifier","lib":"sklearn","task":"classification",
   "param_grid":{"max_depth":[5,10,20],"min_samples_split":[2,10],"min_samples_leaf":[1,5],"criterion":["gini","entropy"]}},

  {"id":"sklearn.ensemble.RandomForestClassifier","label":"RandomForestClassifier","lib":"sklearn","task":"classification",
   "param_grid":{"n_estimators":[100,300],"max_depth":[10,30],"min_samples_split":[2,10],"min_samples_leaf":[1,5]}},

  {"id":"sklearn.ensemble.GradientBoostingClassifier","label":"GradientBoostingClassifier","lib":"sklearn","task":"classification",
   "param_grid":{"n_estimators":[100,300],"learning_rate":[0.05,0.1],"max_depth":[2,3],"subsample":[0.7,1.0]}},

  {"id":"sklearn.svm.SVC","label":"SVC","lib":"sklearn","task":"classification",
   "param_grid":{}},

  # -------- regression --------
  {"id":"sklearn.linear_model.ElasticNet","label":"ElasticNet","lib":"sklearn","task":"regression",
   "param_grid":{"alpha":[0.001,0.01,0.1],"l1_ratio":[0.2,0.5,0.8],"max_iter":[1000,5000]}},

  {"id":"sklearn.tree.DecisionTreeRegressor","label":"DecisionTreeRegressor","lib":"sklearn","task":"regression",
   "param_grid":{"max_depth":[5,10,20],"min_samples_split":[2,10],"min_samples_leaf":[1,5],"criterion":["squared_error","friedman_mse"]}},

  {"id":"sklearn.ensemble.RandomForestRegressor","label":"RandomForestRegressor","lib":"sklearn","task":"regression",
   "param_grid":{"n_estimators":[100,300],"max_depth":[10,30],"min_samples_split":[2,10],"min_samples_leaf":[1,5]}},

  {"id":"sklearn.ensemble.GradientBoostingRegressor","label":"GradientBoostingRegressor","lib":"sklearn","task":"regression",
   "param_grid":{"n_estimators":[100,300],"learning_rate":[0.05,0.1],"max_depth":[2,3],"subsample":[0.7,1.0]}},

  {"id":"sklearn.svm.SVR","label":"SVR","lib":"sklearn","task":"regression",
   "param_grid":{"kernel":["linear","rbf"],"C":[0.1,1,10],"epsilon":[0.1,0.2]}},

  {"id":"sklearn.neighbors.KNeighborsRegressor","label":"KNeighborsRegressor","lib":"sklearn","task":"regression",
   "param_grid":{"n_neighbors":[5,10,20],"weights":["uniform","distance"]}},

  # -------- survival (lifelines / survivors / sksurv) --------
  {"id":"lifelines.fitters.kaplan_meier_fitter.KaplanMeierFitter","label":"KaplanMeierFitter","lib":"lifelines","task":"survival",
   "param_grid":{}},

  {"id":"sksurv.linear_model.CoxPHSurvivalAnalysis","label":"CoxPHSurvivalAnalysis","lib":"sksurv","task":"survival",
   "param_grid":{"alpha":[100,10,1,0.1,0.01,0.001],"ties":["breslow"]}},

  {"id":"sksurv.ensemble.RandomSurvivalForest","label":"RandomSurvivalForest","lib":"sksurv","task":"survival",
   "param_grid":{"n_estimators":[50,100],"max_depth":[None,20],"min_samples_leaf":[0.001,0.01,0.1,0.25],"random_state":[123]}},

  {"id":"sksurv.tree.SurvivalTree","label":"SurvivalTree","lib":"sksurv","task":"survival",
   "param_grid":{"max_depth":[None,20],"min_samples_leaf":[1,10,20],"max_features":[None,"sqrt"],"random_state":[123]}},

  {"id":"sksurv.ensemble.GradientBoostingSurvivalAnalysis","label":"GradientBoostingSurvivalAnalysis","lib":"sksurv","task":"survival",
   "param_grid":{"loss":["coxph"],"learning_rate":[0.01,0.05,0.1,0.5],"n_estimators":[30,50],"subsample":[0.7,1.0],"dropout_rate":[0.0,0.1,0.5],"random_state":[123]}},

  {"id":"survivors.tree.CRAID","label":"CRAID","lib":"survivors","task":"survival",
   "param_grid":{"criterion":["wilcoxon"],"depth":[2,3,4],"min_samples_leaf":[0.05,0.1],"signif":[0.01,0.05],"leaf_model":["base"]}},

  {"id":"survivors.ensemble.ParallelBootstrapCRAID","label":"ParallelBootstrapCRAID","lib":"survivors","task":"survival",
   "param_grid":{"n_estimators":[25,50,100],"random_state":[123]}},
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


def home_context(request: Request, **overrides):
    context = {
        "request": request,
        "messages": [],
        "datasets": DATASETS,
        "presets": PRESETS,
        "models": MODELS,
        "selected": None,
        "plot_data": None,
        "plot_json": None,
        "metrics_list": [],
        "ai_tasks": AI_TASKS,
        "ai_selected": {
            "dataset_id": DATASETS[0]["id"],
            "task_id": "survival",
        },
        "ai_advice": None,
    }
    context.update(overrides)
    return context


@app.get("/", name="home")
async def home(request: Request):
    return templates.TemplateResponse(
        request,
        "home.html",
        home_context(request),
    )


@app.post("/ai-advice", name="ai_advice")
async def ai_advice(
    request: Request,
    ai_dataset_id: str = Form(...),
    ai_task_id: str = Form(...),
):
    advice = build_ai_advice(BASE_DIR, ai_dataset_id, ai_task_id, use_llm=True)
    return templates.TemplateResponse(
        request,
        "home.html",
        home_context(
            request,
            ai_selected={
                "dataset_id": ai_dataset_id,
                "task_id": ai_task_id,
            },
            ai_advice=advice,
        ),
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
    xk = x_metric or (preset["x_metric"] if preset else None)
    yk = y_metric or (preset["y_metric"] if preset else None)
    selected = {
        "dataset_id": dataset_id,
        "preset_id": preset_id,
        "model_ids": model_ids,
        "x_metric": xk,
        "y_metric": yk,
    }

    
    if preset is None:
        return templates.TemplateResponse(request, "home.html", home_context(
            request,
            selected=selected,
            messages=[{"category":"error","text":"Пресет не найден"}],
        ))

    if not model_ids:
        return templates.TemplateResponse(request, "home.html", home_context(
            request,
            selected=selected,
            messages=[{"category":"error","text":"Выбери хотя бы одну модель"}],
        ))

    allow_recompute = not DISABLE_MISSING_RECALC
    selected_cfgs = [m for m in MODELS if m["id"] in model_ids]
    X = y = None

    if allow_recompute:
        load_fn = getattr(ds, f"load_{dataset_id}_dataset", None)
        if load_fn is None:
            return templates.TemplateResponse(request, "home.html", home_context(
                request,
                selected=selected,
                messages=[{"category":"error","text":f"Нет загрузчика датасета: load_{dataset_id}_dataset()"}],
            ))
        X, y, features, categ, sch_nan = load_fn()

    errors = []
    metrics_list = []
    labels = []
    values_by_label = {}
    for mcfg in selected_cfgs:

        vx, vy, _ = get_surv_metrics(
            base_dir=BASE_DIR,
            dataset_id=dataset_id,
            X_tr=X,
            y_tr=y,
            model_cfgs=selected_cfgs,
            model_label=mcfg["label"],
            x_metric=xk,
            y_metric=yk,
            allow_recompute=allow_recompute,
        )

        metrics_list, df = list_surv_metrics_from_table(BASE_DIR, dataset_id)
        r = df[df["method"].astype(str) == str(mcfg["label"])]
        if r.empty:
            msg = f"{mcfg['label']}: нет строки в таблице {dataset_id}.xlsx"
            if not allow_recompute:
                msg += " и автопересчёт отключён"
            errors.append(msg)
            continue

        missing_metrics = []
        if vx is None:
            missing_metrics.append(xk)
        if vy is None and str(yk).lower() != str(xk).lower():
            missing_metrics.append(yk)
        if missing_metrics:
            msg = f"{mcfg['label']}: отсутствуют метрики {', '.join(missing_metrics)}"
            if not allow_recompute:
                msg += " и автопересчёт отключён"
            errors.append(msg)
            continue

        d = r.iloc[0].to_dict()

        values_by_label.setdefault(mcfg["label"], {})
        for k, v in d.items():
            if isinstance(k, str) and k.endswith("_mean"):
                try:
                    values_by_label[mcfg["label"]][k] = float(v)
                except Exception:
                    pass
        labels.append(mcfg["label"])

    if not labels:
        return templates.TemplateResponse(request, "home.html", home_context(
            request,
            selected=selected,
            messages=[{"category":"error","text":"Не построено ни одной точки: " + (" | ".join(errors[:3]) if errors else "")}],
        ))
    plot_data = {
        "labels": labels,
        "metrics": metrics_list,        
        "values": values_by_label,
        "x_metric": xk,
        "y_metric": yk,
    }

    msgs = [{"category":"error","text":" | ".join(errors[:3])}] if errors else []

    return templates.TemplateResponse(request, "home.html", home_context(
        request,
        selected=selected,
        plot_data=plot_data,
        messages=msgs,
        metrics_list=metrics_list,
    ))


@app.get("/goal", name="goal_page")
async def goal_page(request: Request):
    return templates.TemplateResponse(request, "goal.html", {"request": request, "messages": []})


@app.get("/link", name="link_page")
async def link_page(request: Request):
    return templates.TemplateResponse(request, "link.html", {"request": request, "messages": []})


@app.get("/methodology", name="methodology_page")
async def methodology_page(request: Request):
    return templates.TemplateResponse(request, "methodology.html", {"request": request, "messages": []})


@app.get("/example", name="example_page")
async def example_page(request: Request):
    return templates.TemplateResponse(request, "example.html", {"request": request, "messages": []})


@app.get("/leaderboard", name="leaderboard_page")
async def leaderboard_page(request: Request):
    leaderboard_rows = load_overall_leaderboard_rows(BASE_DIR / "tables" / "leaderboards_by_task.xlsx")
    notebook_figures = load_leaderboard_images(BASE_DIR / "images", limit=3)
    top_methods = leaderboard_rows[:3]

    return templates.TemplateResponse(
        request,
        "leaderboard.html",
        {
            "request": request,
            "messages": [],
            "leaderboard_rows": leaderboard_rows,
            "top_methods": top_methods,
            "notebook_figures": notebook_figures,
        },
    )
