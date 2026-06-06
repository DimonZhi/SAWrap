from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from UI.app import BASE_DIR, DATASETS, MODELS, load_dataset_for_recompute
from UI.helpers_demo_predict import build_demo_input_context, build_demo_prediction, find_demo_candidates
from UI.helpers_user_datasets import list_user_dataset_options


def _dataset_options(include_user: bool) -> list[dict]:
    options = list(DATASETS)
    if include_user:
        options.extend(list_user_dataset_options(BASE_DIR))
    return options


def _raw_values_from_context(context: dict | None) -> dict[str, str]:
    if not context:
        return {}
    return {
        str(feature["name"]): str(feature.get("value", ""))
        for feature in context.get("features", [])
    }


def build_demo_models(dataset_ids: set[str] | None = None, include_user: bool = False) -> list[dict]:
    results = []
    for dataset in _dataset_options(include_user):
        dataset_id = str(dataset["id"])
        if dataset_ids and dataset_id not in dataset_ids:
            continue

        try:
            loaded = load_dataset_for_recompute(dataset_id)
            if loaded is None:
                raise RuntimeError("датасет недоступен")
            X, y, *_ = loaded
            if not isinstance(X, pd.DataFrame):
                X = pd.DataFrame(X)

            context = build_demo_input_context(dataset_id, X)
            raw_values = _raw_values_from_context(context)
            candidates = find_demo_candidates(BASE_DIR, dataset_id, MODELS)
            result = build_demo_prediction(
                base_dir=BASE_DIR,
                dataset_id=dataset_id,
                raw_values=raw_values,
                model_cfgs=MODELS,
                load_dataset=load_dataset_for_recompute,
            )
            if not result.get("ok"):
                detail = result.get("error") or "демо не построено"
                if result.get("errors"):
                    detail += ": " + " | ".join(str(error) for error in result["errors"][:3])
                raise RuntimeError(detail)

            card = (result.get("cards") or [{}])[0]
            results.append(
                {
                    "dataset_id": dataset_id,
                    "ok": True,
                    "candidate": candidates[0] if candidates else None,
                    "method": card.get("method"),
                    "category": card.get("category_label"),
                    "source": card.get("source"),
                }
            )
        except Exception as exc:
            results.append(
                {
                    "dataset_id": dataset_id,
                    "ok": False,
                    "error": str(exc),
                }
            )
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Train and cache ParallelBootstrapCRAID demo models.")
    parser.add_argument(
        "--dataset",
        action="append",
        dest="datasets",
        help="Dataset id to build. Can be passed multiple times. Default: all built-in datasets.",
    )
    parser.add_argument(
        "--include-user",
        action="store_true",
        help="Also include uploaded user datasets.",
    )
    args = parser.parse_args()

    results = build_demo_models(
        dataset_ids=set(args.datasets) if args.datasets else None,
        include_user=bool(args.include_user),
    )
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0 if all(item.get("ok") for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
