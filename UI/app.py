from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import List, Optional
import json
import os
import shutil
from fastapi import Form
from sklearn.model_selection import train_test_split
from pathlib import Path
from survivors.external import SAWrapSA, ClassifWrapSA, RegrWrapSA
import survivors.datasets as ds
import survivors.constants as cnt
from .helpers_ai_advice import AI_TASKS, build_ai_advice
from .helpers_demo_predict import (
    MODEL_STORE_DIR,
    build_demo_input_context,
    build_demo_prediction,
)
from .helpers_project_rag import build_project_rag_answer
from .helpers_tables import (
    get_surv_metrics,
    get_piecewise_table_path,
    get_surv_table_path,
    list_surv_metrics_from_table,
    select_best_piecewise_variant,
    select_global_piecewise_time,
)
from .helpers_leaderboard import load_leaderboard_images, load_overall_leaderboard_rows
from .helpers_piecewise import load_piecewise_classification_summary
from .helpers_user_datasets import (
    DatasetUploadError,
    delete_user_dataset,
    is_user_dataset,
    list_user_dataset_options,
    load_user_dataset,
    save_uploaded_dataset,
)


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


PIECEWISE_BASE_MODELS = [
    {"id": "DecisionTreeClassifier", "label": "DecisionTreeClassifier"},
    {"id": "KNeighborsClassifier", "label": "KNeighborsClassifier"},
    {"id": "LogisticRegression", "label": "LogisticRegression"},
    {"id": "RandomForestClassifier", "label": "RandomForestClassifier"},
    {"id": "GradientBoostingClassifier", "label": "GradientBoostingClassifier"},
]
PIECEWISE_OPTIONS = [
    {
        "id": "piecewise_plain",
        "family": "PiecewiseClassifWrapSA",
        "label": "PiecewiseClassifWrapSA",
        "bases_field": "piecewise_plain_bases",
        "description": (
            "Улучшает модели классификации за счет учета временной структуры "
            "события. Временной горизонт делится на равномерные интервалы, "
            "для каждого интервала обучается отдельный классификатор риска. "
            "Вместо одной вероятности события получается survival-кривая, "
            "поэтому модель начинает различать ранние и поздние события."
        ),
    },
    {
        "id": "piecewise_censor",
        "family": "PiecewiseCensorAwareClassifWrapSA",
        "label": "PiecewiseCensorAwareClassifWrapSA",
        "bases_field": "piecewise_censor_bases",
        "description": (
            "Censor-aware версия PiecewiseClassifWrapSA. Она также строит "
            "классификаторы риска на равномерных интервалах и формирует "
            "survival-кривую, но аккуратнее работает с цензурированными "
            "наблюдениями. Это улучшает классификационные модели на неполных "
            "данных: классификатор не просто считает цензурированный объект "
            "отсутствием события, а использует информацию о том, до какого "
            "времени объект точно дожил."
        ),
    },
]
PIECEWISE_DEFAULT_BASE = "DecisionTreeClassifier"
PIECEWISE_RUNTIME_DEFAULT_TIMES = 8


def _valid_piecewise_bases(model_names: Optional[List[str]]) -> list[str]:
    requested = set(model_names or [])
    return [model["id"] for model in PIECEWISE_BASE_MODELS if model["id"] in requested]


def build_piecewise_model_cfgs(
    base_dir: Path,
    dataset_id: str,
    task_id: str,
    piecewise_plain_bases: Optional[List[str]],
    piecewise_censor_bases: Optional[List[str]],
) -> list[dict]:
    base_by_family = {
        "PiecewiseClassifWrapSA": _valid_piecewise_bases(piecewise_plain_bases),
        "PiecewiseCensorAwareClassifWrapSA": _valid_piecewise_bases(piecewise_censor_bases),
    }

    cfgs = []
    classification_cfgs_by_label = {
        model["label"]: model
        for model in MODELS
        if model.get("task") == "classification"
    }
    allow_runtime_piecewise = (not DISABLE_MISSING_RECALC) and is_user_dataset(base_dir, dataset_id)
    for option in PIECEWISE_OPTIONS:
        family = option["family"]
        for base_model in base_by_family[family]:
            label, _ = select_best_piecewise_variant(
                base_dir=base_dir,
                dataset_id=dataset_id,
                family=family,
                base_model=base_model,
                task_id=task_id,
            )
            if not label and allow_runtime_piecewise:
                base_cfg = classification_cfgs_by_label.get(base_model)
                if not base_cfg:
                    continue
                times = (
                    select_global_piecewise_time(base_dir, family, base_model, task_id)
                    or PIECEWISE_RUNTIME_DEFAULT_TIMES
                )
                label = f"{family}({base_model}, times={times})"
                cfgs.append(
                    {
                        "id": base_cfg["id"],
                        "label": label,
                        "lib": base_cfg["lib"],
                        "task": "classification",
                        "param_grid": base_cfg.get("param_grid") or {},
                        "piecewise_family": family,
                        "base_model": base_model,
                        "times": int(times),
                    }
                )
                continue
            if not label:
                continue
            cfgs.append(
                {
                    "id": f"piecewise::{family}::{base_model}::{label}",
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


def preset_for_task(task_id: str) -> dict:
    return next((preset for preset in PRESETS if preset["preset_task"] == task_id), PRESETS[0])


def home_context(request: Request, **overrides):
    explicit_ai_selected = "ai_selected" in overrides
    context = {
        "request": request,
        "messages": [],
        "datasets": DATASETS + list_user_dataset_options(BASE_DIR),
        "presets": PRESETS,
        "models": MODELS,
        "piecewise_options": PIECEWISE_OPTIONS,
        "piecewise_base_models": PIECEWISE_BASE_MODELS,
        "piecewise_default_base": PIECEWISE_DEFAULT_BASE,
        "allow_dataset_upload": not DISABLE_MISSING_RECALC,
        "selected": None,
        "plot_data": None,
        "plot_json": None,
        "metrics_list": [],
        "show_demo": False,
        "demo_context": None,
        "demo_result": None,
        "ai_tasks": AI_TASKS,
        "ai_selected": {
            "dataset_id": DATASETS[0]["id"],
            "task_id": "survival",
        },
        "ai_advice": None,
    }
    context.update(overrides)
    selected_ctx = context.get("selected")
    if selected_ctx:
        context["show_demo"] = is_user_dataset(BASE_DIR, selected_ctx.get("dataset_id"))
        if not explicit_ai_selected:
            preset_id = selected_ctx.get("preset_id")
            preset = next((preset for preset in PRESETS if preset["id"] == preset_id), None)
            context["ai_selected"] = {
                "dataset_id": selected_ctx.get("dataset_id"),
                "task_id": (preset or PRESETS[0]).get("preset_task", "classification"),
            }
    if context.get("demo_context") is None and selected_ctx:
        context["demo_context"] = demo_context_for_dataset(selected_ctx.get("dataset_id"))
    return context


def default_selected(dataset_id: str | None = None, preset_id: str | None = None):
    preset = next((p for p in PRESETS if p["id"] == preset_id), None) or PRESETS[0]
    return {
        "dataset_id": dataset_id or DATASETS[0]["id"],
        "preset_id": preset["id"],
        "model_ids": [],
        "piecewise_plain_bases": [],
        "piecewise_censor_bases": [],
        "x_metric": preset["x_metric"],
        "y_metric": preset["y_metric"],
    }


def load_dataset_for_recompute(dataset_id: str):
    if is_user_dataset(BASE_DIR, dataset_id):
        return load_user_dataset(BASE_DIR, dataset_id)
    load_fn = getattr(ds, f"load_{dataset_id}_dataset", None)
    if load_fn is None:
        return None
    return load_fn()


def delete_dataset_result_files(dataset_id: str) -> list[str]:
    tables_dir = BASE_DIR / "tables"
    candidates = {
        get_surv_table_path(BASE_DIR, dataset_id),
        get_piecewise_table_path(BASE_DIR, dataset_id),
        tables_dir / f"{dataset_id}.xlsx",
        tables_dir / f"{str(dataset_id).lower()}.xlsx",
        tables_dir / f"{str(dataset_id).upper()}.xlsx",
        tables_dir / f"Piecewise_{dataset_id}.xlsx",
        BASE_DIR / f"Piecewise_{dataset_id}.xlsx",
    }

    removed = []
    for path in sorted(candidates, key=lambda item: str(item)):
        if path.exists() and path.is_file():
            path.unlink()
            removed.append(path.name)
    return removed


def delete_dataset_model_store(dataset_id: str) -> bool:
    model_dir = BASE_DIR / MODEL_STORE_DIR / str(dataset_id).strip()
    if not model_dir.exists():
        return False
    shutil.rmtree(model_dir)
    return True


def demo_context_for_dataset(dataset_id: str | None):
    if not dataset_id:
        return None
    if not is_user_dataset(BASE_DIR, dataset_id):
        return None
    try:
        loaded_dataset = load_dataset_for_recompute(dataset_id)
        if loaded_dataset is None:
            return None
        X, y, features, categ, sch_nan = loaded_dataset
    except Exception:
        return None
    return build_demo_input_context(dataset_id, X)


@app.get("/", name="home")
async def home(request: Request):
    return templates.TemplateResponse(
        request,
        "home.html",
        home_context(request),
    )


@app.post("/datasets", name="upload_dataset")
async def upload_dataset(
    request: Request,
    dataset_file: UploadFile = File(...),
    dataset_id: str = Form(DATASETS[0]["id"]),
    preset_id: str = Form(PRESETS[0]["id"]),
):
    selected = default_selected(dataset_id=dataset_id, preset_id=preset_id)
    if DISABLE_MISSING_RECALC:
        return templates.TemplateResponse(
            request,
            "home.html",
            home_context(
                request,
                selected=selected,
                messages=[
                    {
                        "category": "error",
                        "text": "Добавление датасета доступно только при SAWRAP_SKIP_MISSING_RECALC=0.",
                    }
                ],
            ),
        )

    content = await dataset_file.read()
    try:
        manifest = save_uploaded_dataset(BASE_DIR, dataset_file.filename or "dataset.csv", content)
    except DatasetUploadError as exc:
        return templates.TemplateResponse(
            request,
            "home.html",
            home_context(
                request,
                selected=selected,
                messages=[{"category": "error", "text": f"Датасет не добавлен: {exc}"}],
            ),
        )

    selected = default_selected(dataset_id=manifest["id"], preset_id=preset_id)
    detail = f"{manifest['rows']} строк, {manifest['feature_count']} признаков"
    if manifest.get("dropped_rows"):
        detail += f", удалено строк при очистке: {manifest['dropped_rows']}"
    if manifest.get("dropped_feature_count"):
        detail += f", отброшено служебных/пустых признаков: {manifest['dropped_feature_count']}"
    return templates.TemplateResponse(
        request,
        "home.html",
        home_context(
            request,
            selected=selected,
            messages=[
                {
                    "category": "success",
                    "text": f"Датасет «{manifest['label']}» добавлен и приведен к стандартному виду: {detail}.",
                }
            ],
        ),
    )


@app.post("/datasets/delete", name="delete_dataset")
async def delete_dataset(
    request: Request,
    dataset_id: str = Form(...),
    preset_id: str = Form(PRESETS[0]["id"]),
):
    selected = default_selected(preset_id=preset_id)
    user_datasets = list_user_dataset_options(BASE_DIR)
    dataset_label = next(
        (dataset["label"] for dataset in user_datasets if dataset["id"] == dataset_id),
        dataset_id,
    )

    if not is_user_dataset(BASE_DIR, dataset_id):
        return templates.TemplateResponse(
            request,
            "home.html",
            home_context(
                request,
                selected=default_selected(dataset_id=dataset_id, preset_id=preset_id),
                demo_context=False,
                messages=[
                    {
                        "category": "error",
                        "text": "Удалять можно только пользовательские датасеты.",
                    }
                ],
            ),
        )

    try:
        delete_user_dataset(BASE_DIR, dataset_id)
        removed_results = delete_dataset_result_files(dataset_id)
        removed_models = delete_dataset_model_store(dataset_id)
    except FileNotFoundError as exc:
        return templates.TemplateResponse(
            request,
            "home.html",
            home_context(
                request,
                selected=selected,
                demo_context=False,
                messages=[{"category": "error", "text": str(exc)}],
            ),
        )

    details = ""
    if removed_results:
        details = f" Удалены результаты: {', '.join(removed_results)}."
    if removed_models:
        details += " Удалены сохраненные демо-модели."
    return templates.TemplateResponse(
        request,
        "home.html",
        home_context(
            request,
            selected=selected,
            demo_context=False,
            messages=[
                {
                    "category": "success",
                    "text": f"Датасет «{dataset_label}» удален вместе с сохраненными результатами.{details}",
                }
            ],
        ),
    )


@app.post("/ai-advice", name="ai_advice")
async def ai_advice(
    request: Request,
    ai_dataset_id: str = Form(...),
    ai_task_id: str = Form(...),
):
    advice = build_ai_advice(BASE_DIR, ai_dataset_id, ai_task_id, use_llm=True)
    selected = default_selected(
        dataset_id=ai_dataset_id,
        preset_id=preset_for_task(ai_task_id)["id"],
    )
    return templates.TemplateResponse(
        request,
        "home.html",
        home_context(
            request,
            selected=selected,
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


@app.post("/demo-predict", name="demo_predict")
async def demo_predict(request: Request):
    form = await request.form()
    dataset_id = str(form.get("dataset_id", "")).strip()
    preset_id = str(form.get("preset_id", PRESETS[0]["id"])).strip() or PRESETS[0]["id"]
    feature_values = {
        str(key)[len("feature__") :]: str(value)
        for key, value in form.multi_items()
        if str(key).startswith("feature__")
    }
    selected = default_selected(dataset_id=dataset_id, preset_id=preset_id)

    if not is_user_dataset(BASE_DIR, dataset_id):
        result = {
            "ok": False,
            "error": "Демо-прогноз доступен только для загруженных пользовательских датасетов.",
        }
    else:
        result = build_demo_prediction(
            base_dir=BASE_DIR,
            dataset_id=dataset_id,
            raw_values=feature_values,
            model_cfgs=MODELS,
            load_dataset=load_dataset_for_recompute,
        )

    wants_json = "application/json" in str(request.headers.get("accept", "")).lower()
    if wants_json or str(form.get("_ajax", "")).strip() == "1":
        return JSONResponse(result)

    messages = []
    if not result.get("ok"):
        messages.append({"category": "error", "text": result.get("error") or "Демо-прогноз не построен."})

    return templates.TemplateResponse(
        request,
        "home.html",
        home_context(
            request,
            selected=selected,
            demo_result=result,
            messages=messages,
        ),
    )


@app.post("/compare", name="compare_models")
async def compare_models(
    request: Request,
    dataset_id: str = Form(...),
    preset_id: str = Form(...),
    model_ids: Optional[List[str]] = Form(None),
    piecewise_plain_bases: Optional[List[str]] = Form(None),
    piecewise_censor_bases: Optional[List[str]] = Form(None),
    x_metric: Optional[str] = Form(None),
    y_metric: Optional[str] = Form(None),
):
    model_ids = model_ids or []
    piecewise_plain_bases = _valid_piecewise_bases(piecewise_plain_bases)
    piecewise_censor_bases = _valid_piecewise_bases(piecewise_censor_bases)
    preset = next((p for p in PRESETS if p["id"] == preset_id), None)
    xk = x_metric or (preset["x_metric"] if preset else None)
    yk = y_metric or (preset["y_metric"] if preset else None)
    selected = {
        "dataset_id": dataset_id,
        "preset_id": preset_id,
        "model_ids": model_ids,
        "piecewise_plain_bases": piecewise_plain_bases,
        "piecewise_censor_bases": piecewise_censor_bases,
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
        BASE_DIR,
        dataset_id,
        "classification",
        piecewise_plain_bases,
        piecewise_censor_bases,
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
        loaded_dataset = load_dataset_for_recompute(dataset_id)
        if loaded_dataset is None:
            return templates.TemplateResponse(request, "home.html", home_context(
                request,
                selected=selected,
                messages=[{"category":"error","text":f"Нет загрузчика датасета: load_{dataset_id}_dataset()"}],
            ))
        X, y, features, categ, sch_nan = loaded_dataset

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
