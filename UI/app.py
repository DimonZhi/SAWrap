from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
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
from .helpers_project_rag import build_project_rag_answer
from .helpers_tables import list_surv_metrics_from_table, get_surv_metrics
from .helpers_leaderboard import load_leaderboard_images, load_overall_leaderboard_rows
from .helpers_piecewise import load_piecewise_classification_summary


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


PIECEWISE_TIMES = 16
PIECEWISE_BASE_MODELS = [
    {"id": "DecisionTreeClassifier", "label": "DecisionTreeClassifier"},
    {"id": "LogisticRegression", "label": "LogisticRegression"},
    {"id": "RandomForestClassifier", "label": "RandomForestClassifier"},
    {"id": "GradientBoostingClassifier", "label": "GradientBoostingClassifier"},
]
PIECEWISE_OPTIONS = [
    {
        "id": "piecewise_plain",
        "family": "PiecewiseClassifWrapSA",
        "label": "PiecewiseClassifWrapSA",
        "base_field": "piecewise_plain_base",
        "description": "Интервальные классификаторы без отдельной censor-aware коррекции.",
    },
    {
        "id": "piecewise_censor",
        "family": "PiecewiseCensorAwareClassifWrapSA",
        "label": "PiecewiseCensorAwareClassifWrapSA",
        "base_field": "piecewise_censor_base",
        "description": "Интервальные классификаторы с учетом цензурированных наблюдений.",
    },
]
PIECEWISE_DEFAULT_BASE = "DecisionTreeClassifier"


def _valid_piecewise_base(model_name: str) -> str:
    allowed = {model["id"] for model in PIECEWISE_BASE_MODELS}
    return model_name if model_name in allowed else PIECEWISE_DEFAULT_BASE


def build_piecewise_model_cfgs(
    piecewise_families: Optional[List[str]],
    piecewise_plain_base: str,
    piecewise_censor_base: str,
) -> list[dict]:
    selected_families = set(piecewise_families or [])
    base_by_family = {
        "PiecewiseClassifWrapSA": _valid_piecewise_base(piecewise_plain_base),
        "PiecewiseCensorAwareClassifWrapSA": _valid_piecewise_base(piecewise_censor_base),
    }

    cfgs = []
    for option in PIECEWISE_OPTIONS:
        family = option["family"]
        if family not in selected_families:
            continue
        base_model = base_by_family[family]
        label = f"{family}({base_model}, times={PIECEWISE_TIMES})"
        cfgs.append(
            {
                "id": f"piecewise::{family}::{base_model}::{PIECEWISE_TIMES}",
                "label": label,
                "lib": "survivors",
                "task": "classification",
                "param_grid": {},
                "precomputed_only": True,
            }
        )
    return cfgs



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
        "piecewise_options": PIECEWISE_OPTIONS,
        "piecewise_base_models": PIECEWISE_BASE_MODELS,
        "piecewise_times": PIECEWISE_TIMES,
        "piecewise_default_base": PIECEWISE_DEFAULT_BASE,
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


@app.post("/project-rag-chat", name="project_rag_chat")
async def project_rag_chat(request: Request):
    try:
        payload = await request.json()
    except Exception:
        return JSONResponse({"ok": False, "error": "Некорректный JSON-запрос."}, status_code=400)

    question = str(payload.get("question", "")).strip()
    history = payload.get("history") or []

    if not question:
        return JSONResponse({"ok": False, "error": "Вопрос пустой."}, status_code=400)
    if not isinstance(history, list):
        history = []

    answer = build_project_rag_answer(BASE_DIR.parent, question, history)
    if answer.get("text"):
        return {
            "ok": True,
            "answer": answer["text"],
            "provider": answer["provider"],
            "model": answer["model"],
            "sources": answer.get("sources", []),
        }

    return JSONResponse(
        {
            "ok": False,
            "error": answer.get("error") or "Не удалось получить RAG-ответ.",
            "provider": answer.get("provider"),
            "model": answer.get("model"),
            "sources": answer.get("sources", []),
        },
        status_code=502 if answer.get("enabled") else 400,
    )


@app.post("/compare", name="compare_models")
async def compare_models(
    request: Request,
    dataset_id: str = Form(...),
    preset_id: str = Form(...),
    model_ids: Optional[List[str]] = Form(None),
    piecewise_families: Optional[List[str]] = Form(None),
    piecewise_plain_base: str = Form(PIECEWISE_DEFAULT_BASE),
    piecewise_censor_base: str = Form(PIECEWISE_DEFAULT_BASE),
    x_metric: Optional[str] = Form(None),
    y_metric: Optional[str] = Form(None),
):
    model_ids = model_ids or []
    piecewise_families = piecewise_families or []
    piecewise_plain_base = _valid_piecewise_base(piecewise_plain_base)
    piecewise_censor_base = _valid_piecewise_base(piecewise_censor_base)
    preset = next((p for p in PRESETS if p["id"] == preset_id), None)
    xk = x_metric or (preset["x_metric"] if preset else None)
    yk = y_metric or (preset["y_metric"] if preset else None)
    selected = {
        "dataset_id": dataset_id,
        "preset_id": preset_id,
        "model_ids": model_ids,
        "piecewise_families": piecewise_families,
        "piecewise_plain_base": piecewise_plain_base,
        "piecewise_censor_base": piecewise_censor_base,
        "x_metric": xk,
        "y_metric": yk,
    }

    
    if preset is None:
        return templates.TemplateResponse(request, "home.html", home_context(
            request,
            selected=selected,
            messages=[{"category":"error","text":"Пресет не найден"}],
        ))

    piecewise_cfgs = build_piecewise_model_cfgs(
        piecewise_families,
        piecewise_plain_base,
        piecewise_censor_base,
    )
    selected_cfgs = [m for m in MODELS if m["id"] in model_ids] + piecewise_cfgs

    if not selected_cfgs:
        return templates.TemplateResponse(request, "home.html", home_context(
            request,
            selected=selected,
            messages=[{"category":"error","text":"Выбери хотя бы одну модель"}],
        ))

    allow_recompute = not DISABLE_MISSING_RECALC
    can_recompute_selected = allow_recompute and any(
        not mcfg.get("precomputed_only") for mcfg in selected_cfgs
    )
    X = y = None

    if can_recompute_selected:
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
            allow_recompute=can_recompute_selected and not mcfg.get("precomputed_only"),
        )

        row_metrics_list, df = list_surv_metrics_from_table(BASE_DIR, dataset_id, model_label=mcfg["label"])
        metrics_list = sorted(set(metrics_list).union(row_metrics_list))
        r = df[df["method"].astype(str) == str(mcfg["label"])]
        if r.empty:
            msg = f"{mcfg['label']}: нет строки в таблицах для {dataset_id}"
            if not allow_recompute or mcfg.get("precomputed_only"):
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
            if not allow_recompute or mcfg.get("precomputed_only"):
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


@app.get("/overview", name="overview_page")
async def overview_page(request: Request):
    leaderboard_rows = load_overall_leaderboard_rows(BASE_DIR / "tables" / "leaderboards_by_task.xlsx")
    notebook_figures = load_leaderboard_images(BASE_DIR / "images", limit=3)
    piecewise_summary = load_piecewise_classification_summary(BASE_DIR / "tables")
    return templates.TemplateResponse(
        request,
        "overview.html",
        {
            "request": request,
            "messages": [],
            "leaderboard_rows": leaderboard_rows,
            "top_methods": leaderboard_rows[:3],
            "notebook_figures": notebook_figures,
            "piecewise_summary": piecewise_summary,
        },
    )


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
