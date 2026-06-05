import gzip
from pathlib import Path

from fastapi.testclient import TestClient

import UI.app as app_module


def _write_user_dataset(base_dir: Path, dataset_id: str = "custom_demo"):
    dataset_dir = base_dir / "user_datasets" / dataset_id
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "data.csv").write_text(
        "time,cens,x\n1,1,0\n2,0,1\n3,1,2\n4,0,3\n5,1,4\n",
        encoding="utf-8",
    )
    (dataset_dir / "manifest.json").write_text(
        '{"id":"custom_demo","label":"demo"}',
        encoding="utf-8",
    )
    return dataset_dir


def test_delete_dataset_removes_user_dataset_and_results(tmp_path: Path, monkeypatch):
    dataset_dir = _write_user_dataset(tmp_path)
    tables_dir = tmp_path / "tables"
    tables_dir.mkdir()
    result_path = tables_dir / "custom_demo.xlsx"
    piecewise_path = tables_dir / "Piecewise_custom_demo.xlsx"
    result_path.write_bytes(b"result")
    piecewise_path.write_bytes(b"piecewise")
    monkeypatch.setattr(app_module, "BASE_DIR", tmp_path)

    client = TestClient(app_module.app, raise_server_exceptions=False)
    response = client.post(
        "/datasets/delete",
        data={"dataset_id": "custom_demo", "preset_id": "cls_auc_logloss"},
    )

    assert response.status_code == 200
    assert not dataset_dir.exists()
    assert not result_path.exists()
    assert not piecewise_path.exists()
    assert "удален" in response.text


def test_delete_dataset_rejects_builtin_dataset(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(app_module, "BASE_DIR", tmp_path)

    client = TestClient(app_module.app, raise_server_exceptions=False)
    response = client.post(
        "/datasets/delete",
        data={"dataset_id": "actg", "preset_id": "cls_auc_logloss"},
    )

    assert response.status_code == 200
    assert "Удалять можно только пользовательские датасеты" in response.text


def test_upload_dataset_accepts_exp_csv_gz(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(app_module, "BASE_DIR", tmp_path)
    monkeypatch.setattr(app_module, "DISABLE_MISSING_RECALC", False)
    content = gzip.compress(
        (
            "serial_number,manufacturer,model,smart_1_raw,event,event_time\n"
            "disk_1,A,2,10,False,471\n"
            "disk_2,A,2,20,True,202\n"
            "disk_3,B,3,30,False,203\n"
            "disk_4,B,3,40,True,204\n"
            "disk_5,C,4,50,False,205\n"
            "disk_6,C,4,60,True,206\n"
        ).encode()
    )

    client = TestClient(app_module.app, raise_server_exceptions=False)
    response = client.post(
        "/datasets",
        data={"dataset_id": "actg", "preset_id": "cls_auc_logloss"},
        files={"dataset_file": ("Cut_Alibaba.csv.gz", content, "application/gzip")},
    )

    assert response.status_code == 200
    assert "добавлен и приведен к стандартному виду" in response.text
    assert (tmp_path / "user_datasets" / "custom_cut_alibaba_csv" / "data.csv").exists()
