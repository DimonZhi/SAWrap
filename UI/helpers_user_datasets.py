import io
import json
import re
import shutil
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


USER_DATASETS_DIR = "user_datasets"
DATA_FILE = "data.csv"
MANIFEST_FILE = "manifest.json"
MAX_UPLOAD_BYTES = 128 * 1024 * 1024
HIGH_MISSING_FEATURE_THRESHOLD = 0.99
HIGH_CARDINALITY_TEXT_UNIQUE_RATIO = 0.9
HIGH_CARDINALITY_TEXT_MIN_UNIQUE = 50

TIME_ALIASES = (
    "time",
    "event_time",
    "duration",
    "dtime",
    "rfst",
    "futime",
    "survivaltime",
    "survival_time",
    "days",
    "days_to_event",
    "tte",
    "tenure",
    "tenure_months",
)
EVENT_ALIASES = (
    "cens",
    "event",
    "status",
    "death",
    "dead",
    "outcome",
    "observed",
    "event_observed",
    "censor",
    "censored",
    "churn",
    "churn_value",
)
METADATA_COLUMN_KEYS = {
    "customerid",
    "date",
    "id",
    "latlong",
    "serialnumber",
    "timerow",
    "unnamed0",
}
TRUE_TOKENS = {
    "1",
    "true",
    "t",
    "yes",
    "y",
    "event",
    "observed",
    "dead",
    "death",
    "deceased",
    "case",
    "recurrence",
    "relapse",
}
FALSE_TOKENS = {
    "0",
    "false",
    "f",
    "no",
    "n",
    "censored",
    "censor",
    "alive",
    "none",
    "noevent",
    "no_event",
    "control",
}


class DatasetUploadError(ValueError):
    pass


def _storage_root(base_dir: Path) -> Path:
    return base_dir / USER_DATASETS_DIR


def _dataset_dir(base_dir: Path, dataset_id: str) -> Path:
    dataset_id = str(dataset_id or "").strip()
    if not re.fullmatch(r"[a-z0-9_]+", dataset_id):
        raise FileNotFoundError(f"Пользовательский датасет не найден: {dataset_id}")

    root = _storage_root(base_dir).resolve()
    dataset_dir = (root / dataset_id).resolve()
    if dataset_dir.parent != root:
        raise FileNotFoundError(f"Пользовательский датасет не найден: {dataset_id}")
    return dataset_dir


def _key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def _safe_slug(value: str, fallback: str = "dataset") -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")
    return slug or fallback


def _unique_dataset_id(root: Path, filename: str) -> str:
    stem = Path(filename or "").stem
    base = f"custom_{_safe_slug(stem)}"
    dataset_id = base
    counter = 2
    while (root / dataset_id).exists():
        dataset_id = f"{base}_{counter}"
        counter += 1
    return dataset_id


def _unique_columns(columns: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    unique = []
    for raw in columns:
        name = _safe_slug(raw, "feature")
        if name in {"time", "cens", "event", "status"}:
            name = f"feature_{name}"
        count = seen.get(name, 0)
        seen[name] = count + 1
        unique.append(name if count == 0 else f"{name}_{count + 1}")
    return unique


def _find_column(columns: list[str], aliases, kind: str) -> str | None:
    by_key = {_key(col): col for col in columns}
    for alias in aliases:
        hit = by_key.get(_key(alias))
        if hit:
            return hit

    for col in columns:
        norm = _key(col)
        if kind == "time" and ("duration" in norm or norm.endswith("time")):
            return col
        if kind == "event" and ("event" in norm or "cens" in norm):
            return col
    return None


def _read_table(filename: str, content: bytes) -> pd.DataFrame:
    if len(content) > MAX_UPLOAD_BYTES:
        max_mb = MAX_UPLOAD_BYTES // (1024 * 1024)
        raise DatasetUploadError(f"Файл слишком большой: максимум {max_mb} МБ.")

    suffixes = [suffix.lower() for suffix in Path(filename or "").suffixes]
    suffix = suffixes[-1] if suffixes else ""
    buffer = io.BytesIO(content)
    try:
        if suffix in {".xlsx", ".xlsm", ".xls"}:
            return pd.read_excel(buffer)
        if suffix == ".zip":
            return _read_zip_csv(buffer)
        if suffix == ".gz" and ".csv" in suffixes:
            return pd.read_csv(buffer, compression="gzip")
        if suffix == ".csv" or not suffix:
            try:
                return pd.read_csv(buffer, sep=None, engine="python")
            except Exception:
                buffer.seek(0)
                return pd.read_csv(buffer)
    except Exception as exc:
        raise DatasetUploadError(f"Не удалось прочитать файл: {exc}") from exc

    raise DatasetUploadError("Поддерживаются CSV, CSV.GZ, ZIP с CSV и XLSX.")


def _read_zip_csv(buffer: io.BytesIO) -> pd.DataFrame:
    try:
        with zipfile.ZipFile(buffer) as archive:
            csv_names = [
                name
                for name in archive.namelist()
                if not name.endswith("/") and Path(name).suffix.lower() == ".csv"
            ]
            if not csv_names:
                raise DatasetUploadError("В ZIP не найден CSV-файл.")
            with archive.open(csv_names[0]) as csv_file:
                return pd.read_csv(csv_file)
    except DatasetUploadError:
        raise
    except Exception as exc:
        raise DatasetUploadError(f"Не удалось прочитать ZIP: {exc}") from exc


def _coerce_time(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    non_missing = series.notna()
    if numeric[non_missing].notna().all():
        return numeric

    timedeltas = pd.to_timedelta(series, errors="coerce")
    if timedeltas[non_missing].notna().any():
        return timedeltas.dt.total_seconds() / 86400.0

    return numeric


def _coerce_event(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    non_missing = series.notna()
    if numeric[non_missing].notna().all():
        return pd.Series(np.where(numeric > 0, 1, 0), index=series.index).where(numeric.notna())

    values = []
    for value in series:
        if pd.isna(value):
            values.append(np.nan)
            continue
        token = re.sub(r"[\s\-]+", "_", str(value).strip().lower())
        compact = token.replace("_", "")
        if token in TRUE_TOKENS or compact in TRUE_TOKENS:
            values.append(1)
        elif token in FALSE_TOKENS or compact in FALSE_TOKENS:
            values.append(0)
        else:
            number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
            values.append(1 if pd.notna(number) and float(number) > 0 else np.nan)
    return pd.Series(values, index=series.index)


def _is_high_cardinality_text(series: pd.Series) -> bool:
    numeric = pd.to_numeric(series, errors="coerce")
    non_missing_count = int(series.notna().sum())
    if non_missing_count == 0:
        return False
    numeric_count = int(numeric.notna().sum())
    if numeric_count / non_missing_count >= 0.9:
        return False
    unique_count = int(series.dropna().astype(str).nunique())
    return (
        unique_count >= HIGH_CARDINALITY_TEXT_MIN_UNIQUE
        and unique_count / non_missing_count >= HIGH_CARDINALITY_TEXT_UNIQUE_RATIO
    )


def _select_feature_columns(
    df: pd.DataFrame,
    excluded_columns: set[str],
) -> tuple[list[str], list[str]]:
    feature_columns = []
    dropped_features = []
    for column in df.columns:
        if column in excluded_columns:
            continue
        key = _key(column)
        series = df[column]
        if key in METADATA_COLUMN_KEYS or key.startswith("unnamed"):
            dropped_features.append(column)
            continue
        if float(series.isna().mean()) >= HIGH_MISSING_FEATURE_THRESHOLD:
            dropped_features.append(column)
            continue
        if _is_high_cardinality_text(series):
            dropped_features.append(column)
            continue
        feature_columns.append(column)
    return feature_columns, dropped_features


def _prepare_features(df: pd.DataFrame, feature_columns: list[str]) -> tuple[pd.DataFrame, list[str], list[str]]:
    prepared = []
    prepared_names = _unique_columns(feature_columns)
    categorical_features: list[str] = []

    for source_name, target_name in zip(feature_columns, prepared_names):
        series = df[source_name]
        if series.notna().sum() == 0:
            continue

        numeric = pd.to_numeric(series, errors="coerce")
        non_missing_count = int(series.notna().sum())
        numeric_count = int(numeric.notna().sum())
        if non_missing_count and numeric_count / non_missing_count >= 0.9:
            fill_value = numeric.median()
            if pd.isna(fill_value):
                fill_value = 0.0
            prepared.append((target_name, numeric.fillna(fill_value).astype(float)))
            continue

        categorical = pd.Categorical(series.astype("string").fillna("__missing__"))
        categorical_features.append(target_name)
        prepared.append((target_name, pd.Series(categorical.codes, index=series.index).astype(float)))

    if not prepared:
        raise DatasetUploadError("После удаления time/cens не осталось признаков для обучения.")

    return pd.DataFrame({name: values for name, values in prepared}), [name for name, _ in prepared], categorical_features


def standardize_uploaded_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    if df is None or df.empty:
        raise DatasetUploadError("Файл не содержит строк.")

    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]
    time_col = _find_column(list(df.columns), TIME_ALIASES, "time")
    event_col = _find_column(list(df.columns), EVENT_ALIASES, "event")
    if not time_col or not event_col:
        raise DatasetUploadError("Нужны колонки времени и события: например time и event/cens.")
    if time_col == event_col:
        raise DatasetUploadError("Колонки времени и события должны быть разными.")

    time = _coerce_time(df[time_col])
    event = _coerce_event(df[event_col])
    valid_mask = time.notna() & event.notna() & np.isfinite(time) & (time >= 0)
    dropped_rows = int((~valid_mask).sum())
    if valid_mask.sum() < 5:
        raise DatasetUploadError("После очистки осталось меньше 5 строк.")

    clean_df = df.loc[valid_mask].reset_index(drop=True)
    clean_time = time.loc[valid_mask].reset_index(drop=True).astype(float)
    clean_event = event.loc[valid_mask].reset_index(drop=True).astype(int)
    if clean_time.min() == 0:
        clean_time = clean_time + 1.0
    if clean_time.nunique() < 2:
        raise DatasetUploadError("Колонка времени должна содержать хотя бы два разных значения.")
    if clean_event.nunique() < 2:
        raise DatasetUploadError("В колонке события нужны оба класса: 0 для цензуры и 1 для события.")

    feature_columns, dropped_features = _select_feature_columns(clean_df, {time_col, event_col})
    X, feature_names, categorical_features = _prepare_features(clean_df, feature_columns)
    standard = pd.concat(
        [
            pd.DataFrame({"time": clean_time, "cens": clean_event.astype(bool)}),
            X.reset_index(drop=True),
        ],
        axis=1,
    )
    manifest = {
        "rows": int(len(standard)),
        "features": feature_names,
        "feature_count": int(len(feature_names)),
        "categorical_features": categorical_features,
        "time_column": time_col,
        "event_column": event_col,
        "dropped_rows": dropped_rows,
        "dropped_feature_count": int(len(dropped_features)),
        "dropped_features_sample": [str(name) for name in dropped_features[:20]],
    }
    return standard, manifest


def save_uploaded_dataset(base_dir: Path, filename: str, content: bytes) -> dict:
    root = _storage_root(base_dir)
    root.mkdir(parents=True, exist_ok=True)

    raw = _read_table(filename, content)
    standard, manifest = standardize_uploaded_dataset(raw)
    dataset_id = _unique_dataset_id(root, filename)
    label = Path(filename or dataset_id).stem or dataset_id
    dataset_dir = root / dataset_id
    dataset_dir.mkdir()

    manifest.update(
        {
            "id": dataset_id,
            "label": label,
            "source_filename": filename,
            "standard_columns": ["time", "cens"] + manifest["features"],
        }
    )
    standard.to_csv(dataset_dir / DATA_FILE, index=False)
    (dataset_dir / MANIFEST_FILE).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def list_user_dataset_options(base_dir: Path) -> list[dict]:
    root = _storage_root(base_dir)
    if not root.exists():
        return []

    options = []
    for manifest_path in sorted(root.glob(f"*/{MANIFEST_FILE}")):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        dataset_id = str(manifest.get("id") or manifest_path.parent.name)
        label = str(manifest.get("label") or dataset_id)
        options.append(
            {
                "id": dataset_id,
                "label": f"{label} · пользовательский",
                "custom": True,
            }
        )
    return options


def is_user_dataset(base_dir: Path, dataset_id: str) -> bool:
    try:
        return (_dataset_dir(base_dir, dataset_id) / DATA_FILE).exists()
    except FileNotFoundError:
        return False


def load_user_dataset(base_dir: Path, dataset_id: str):
    dataset_dir = _dataset_dir(base_dir, dataset_id)
    data_path = dataset_dir / DATA_FILE
    manifest_path = dataset_dir / MANIFEST_FILE
    if not data_path.exists():
        raise FileNotFoundError(f"Пользовательский датасет не найден: {dataset_id}")

    df = pd.read_csv(data_path)
    if "time" not in df.columns or "cens" not in df.columns:
        raise DatasetUploadError("Сохраненный датасет поврежден: нет time/cens.")

    X = df.drop(columns=["time", "cens"])
    cens = _coerce_event(df["cens"])
    if cens.isna().any():
        raise DatasetUploadError("Сохраненный датасет поврежден: некорректный cens.")
    y = np.array(
        list(zip(cens.astype(bool), df["time"].astype(float))),
        dtype=[("cens", "?"), ("time", "<f8")],
    )
    features = list(X.columns)
    categorical_features: list[str] = []
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            categorical_features = list(manifest.get("categorical_features") or [])
        except Exception:
            categorical_features = []
    return X, y, features, categorical_features, None


def delete_user_dataset(base_dir: Path, dataset_id: str) -> Path:
    dataset_dir = _dataset_dir(base_dir, dataset_id)
    data_path = dataset_dir / DATA_FILE
    if not data_path.exists():
        raise FileNotFoundError(f"Пользовательский датасет не найден: {dataset_id}")
    shutil.rmtree(dataset_dir)
    return dataset_dir
