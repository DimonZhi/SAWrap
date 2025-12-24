import pandas as pd
from pathlib import Path

def _norm(s: str) -> str:
    return str(s).strip().lower()

def load_surv_table(base_dir: Path, cfg: dict):
    rel = cfg.get("table")
    if not rel:
        return None
    p = (base_dir / rel).resolve()
    if not p.exists():
        return None

    df = pd.read_excel(p)
    df = df.rename(columns={c: _norm(c) for c in df.columns})

    if "method" not in df.columns:
        return None

    df["__method_norm__"] = df["method"].map(_norm)
    return df

def lookup_metric(df, model_label: str, metric_key: str):
    if df is None:
        return None

    row = df[df["__method_norm__"] == _norm(model_label)]
    if row.empty:
        return None

    col = _norm(metric_key) + "_mean"
    if col not in df.columns:
        return None

    try:
        return float(row.iloc[0][col])
    except:
        return None
