from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


DATASETS = [
    {"id": "gbsg", "label": "GBSG"},
]
PRESETS = [
    {
        "id": "cls_acc_roc", "label": "Классификация: Accuracy и ROC AUC",
        "x_metric": "accuracy",
        "y_metric": "roc_auc",
        "x_label": "Accuracy",
        "y_label": "ROC AUC",
    },
    {
        "id": "reg_rmse_r2", "label": "Регрессия: RMSE и R2",
        "x_metric": "rmse",
        "y_metric": "r2",
        "x_label": "RMSE",
        "y_label": "R2",
    },
    {
        "id": "cls_ci_ibs", "label": "Выживаемость: C-index и IBS",
        "x_metric": "cindex",
        "y_metric": "ibs_remain",
        "x_label": "C-index",
        "y_label": "IBS",
    },
]
MODELS =[
    # sklearn
    {"id": "sk_logreg", "label": "LogisticRegression (sklearn)", "lib": "sklearn"},
    {"id": "sk_rf_cls", "label": "RandomForestClassifier (sklearn)", "lib": "sklearn"},
    {"id": "sk_rf_reg", "label": "RandomForestRegressor (sklearn)", "lib": "sklearn"},

    # lifelines
    {"id": "ll_cox", "label": "CoxPHFitter (lifelines)", "lib": "lifelines"},
    {"id": "ll_weibull", "label": "WeibullAFTFitter (lifelines)", "lib": "lifelines"},

    # survivors
    {"id": "sv_cr", "label": "CRAID (survivors)", "lib": "survivors"},
    {"id": "sv_rsf", "label": "RandomSurvivalForest (survivors)", "lib": "survivors"},
]
DUMMY_SCORES = {
    "sk_logreg": {
        "roc_auc": 0.81, "f1": 0.73,
        "mse": 0.52, "r2": 0.58,
        "c_index": 0.79, "ibs": 0.18,
    },
    "sk_rf_cls": {
        "roc_auc": 0.86, "f1": 0.77,
        "mse": 0.49, "r2": 0.62,
        "c_index": 0.82, "ibs": 0.16,
    },
    "sk_rf_reg": {
        "roc_auc": 0.75, "f1": 0.68,
        "mse": 0.41, "r2": 0.69,
        "c_index": 0.78, "ibs": 0.17,
    },
    "ll_cox": {
        "roc_auc": 0.79, "f1": 0.71,
        "mse": 0.55, "r2": 0.54,
        "c_index": 0.83, "ibs": 0.15,
    },
    "ll_weibull": {
        "roc_auc": 0.77, "f1": 0.70,
        "mse": 0.53, "r2": 0.56,
        "c_index": 0.81, "ibs": 0.16,
    },
    "sv_cr": {
        "roc_auc": 0.84, "f1": 0.76,
        "mse": 0.48, "r2": 0.63,
        "c_index": 0.85, "ibs": 0.14,
    },
    "sv_rsf": {
        "roc_auc": 0.88, "f1": 0.79,
        "mse": 0.46, "r2": 0.65,
        "c_index": 0.87, "ibs": 0.13,
    },
}

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
    dataset_id: str,
    preset_id: str,
    model_ids: list[str] | None = None,
):    # Здесь будет логика сравнения моделей и генерации графика
    plot_json = "{}"
    
    return templates.TemplateResponse(
        "home.html",
        {
            "request": request,
            "messages": [],
            "datasets": DATASETS,
            "presets": PRESETS,
            "models": MODELS,
            "selected": {
                "dataset_id": dataset_id,
                "preset_id": preset_id,
                "model_ids": model_ids or [],
            },
            "plot_json": plot_json,
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
