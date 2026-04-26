import cgi
import csv
import json
import mimetypes
import os
from typing import Callable, Iterable, Tuple
from wsgiref.simple_server import make_server

from steganalysis_service import SteganalysisService


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(ROOT_DIR, "webui", "static")
INDEX_PATH = os.path.join(ROOT_DIR, "webui", "index.html")
EXPERIMENTS_PATH = os.path.join(ROOT_DIR, "webui", "experiments.html")
EVALUATION_DIR = os.path.join(ROOT_DIR, "evaluation_outputs")

service = SteganalysisService(ROOT_DIR)


def _safe_float(value: str):
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _parse_experiment_name(experiment_name: str):
    lower_name = experiment_name.lower()
    if lower_name.startswith("suni"):
        algorithm = "S-UNIWARD"
        rate = lower_name.replace("suni", "")
    elif lower_name.startswith("wow"):
        algorithm = "WOW"
        rate = lower_name.replace("wow", "")
    else:
        algorithm = "UNKNOWN"
        rate = ""
    return algorithm, rate


def load_experiment_results():
    summary_csv = os.path.join(ROOT_DIR, "evaluation_outputs", "log_plots", "experiment_summary.csv")
    eval_root = os.path.join(ROOT_DIR, "evaluation_outputs")
    experiments = []

    if not os.path.exists(summary_csv):
        return experiments

    with open(summary_csv, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            exp_name = (row.get("experiment") or "").strip()
            if not exp_name:
                continue

            algorithm, embedding_rate = _parse_experiment_name(exp_name)
            eval_json_path = os.path.join(eval_root, f"{exp_name}_eval", "metrics_summary.json")
            eval_metrics = {}
            if os.path.exists(eval_json_path):
                with open(eval_json_path, "r", encoding="utf-8") as metrics_handle:
                    eval_metrics = json.load(metrics_handle)

            conf = eval_metrics.get("confusion_matrix", {}) if isinstance(eval_metrics, dict) else {}

            experiments.append(
                {
                    "id": exp_name,
                    "algorithm": algorithm,
                    "embedding_rate": embedding_rate,
                    "epochs": int(_safe_float(row.get("epochs")) or 0),
                    "best_valid_accuracy": _safe_float(row.get("best_valid_accuracy")),
                    "best_test_accuracy": _safe_float(row.get("best_test_accuracy")),
                    "final_test_accuracy": _safe_float(row.get("final_test_accuracy")),
                    "final_test_loss": _safe_float(row.get("final_test_loss")),
                    "training_time_seconds": _safe_float(row.get("training_time_seconds")),
                    "accuracy": eval_metrics.get("accuracy"),
                    "precision": eval_metrics.get("precision"),
                    "recall": eval_metrics.get("recall"),
                    "f1_score": eval_metrics.get("f1_score"),
                    "auc": eval_metrics.get("auc"),
                    "average_loss": eval_metrics.get("average_loss"),
                    "sample_count": eval_metrics.get("sample_count"),
                    "confusion_matrix": {
                        "tn": conf.get("tn"),
                        "fp": conf.get("fp"),
                        "fn": conf.get("fn"),
                        "tp": conf.get("tp"),
                    },
                    "images": {
                        "roc_curve": f"/evaluation_outputs/{exp_name}_eval/roc_curve.png",
                        "confusion_matrix": f"/evaluation_outputs/{exp_name}_eval/confusion_matrix.png",
                        "probability_histogram": f"/evaluation_outputs/{exp_name}_eval/probability_histogram.png",
                    },
                }
            )

    experiments.sort(key=lambda item: (item["algorithm"], item["embedding_rate"]))
    return experiments


def json_response(start_response: Callable, status: str, payload: dict) -> Iterable[bytes]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = [
        ("Content-Type", "application/json; charset=utf-8"),
        ("Content-Length", str(len(body))),
    ]
    start_response(status, headers)
    return [body]


def file_response(start_response: Callable, file_path: str, content_type: str) -> Iterable[bytes]:
    with open(file_path, "rb") as handle:
        body = handle.read()
    headers = [
        ("Content-Type", content_type),
        ("Content-Length", str(len(body))),
    ]
    start_response("200 OK", headers)
    return [body]


def not_found(start_response: Callable) -> Iterable[bytes]:
    return json_response(start_response, "404 Not Found", {"error": "Resource not found."})


def internal_error(start_response: Callable, message: str) -> Iterable[bytes]:
    return json_response(start_response, "500 Internal Server Error", {"error": message})


def parse_form_data(environ) -> Tuple[str, bytes]:
    form = cgi.FieldStorage(fp=environ["wsgi.input"], environ=environ, keep_blank_values=True)
    model_id = form.getfirst("model_id", "").strip()
    upload = form["image"] if "image" in form else None

    if not model_id:
        raise ValueError("请先选择检测模型。")
    if upload is None or not getattr(upload, "file", None):
        raise ValueError("请上传待检测图片。")

    image_bytes = upload.file.read()
    if not image_bytes:
        raise ValueError("上传图片为空，请重新选择文件。")

    return model_id, image_bytes


def app(environ, start_response):
    method = environ.get("REQUEST_METHOD", "GET").upper()
    path = environ.get("PATH_INFO", "/")

    try:
        if path == "/" and method == "GET":
            return file_response(start_response, INDEX_PATH, "text/html; charset=utf-8")

        if path in ("/experiments", "/achievements") and method == "GET":
            return file_response(start_response, EXPERIMENTS_PATH, "text/html; charset=utf-8")

        if path == "/static/styles.css" and method == "GET":
            return file_response(start_response, os.path.join(STATIC_DIR, "styles.css"), "text/css; charset=utf-8")

        if path == "/static/app.js" and method == "GET":
            return file_response(start_response, os.path.join(STATIC_DIR, "app.js"), "application/javascript; charset=utf-8")

        if path == "/static/experiments.js" and method == "GET":
            return file_response(
                start_response,
                os.path.join(STATIC_DIR, "experiments.js"),
                "application/javascript; charset=utf-8",
            )

        if path.startswith("/evaluation_outputs/") and method == "GET":
            relative_path = path[len("/evaluation_outputs/") :]
            file_path = os.path.abspath(os.path.join(EVALUATION_DIR, relative_path))
            if not file_path.startswith(os.path.abspath(EVALUATION_DIR)):
                return not_found(start_response)
            if not os.path.exists(file_path) or not os.path.isfile(file_path):
                return not_found(start_response)
            content_type, _ = mimetypes.guess_type(file_path)
            return file_response(start_response, file_path, content_type or "application/octet-stream")

        if path == "/api/models" and method == "GET":
            return json_response(start_response, "200 OK", {"models": service.list_models()})

        if path == "/api/experiments" and method == "GET":
            return json_response(start_response, "200 OK", {"experiments": load_experiment_results()})

        if path == "/api/predict" and method == "POST":
            model_id, image_bytes = parse_form_data(environ)
            result = service.predict(model_id, image_bytes)
            return json_response(start_response, "200 OK", result)

        return not_found(start_response)
    except ValueError as exc:
        return json_response(start_response, "400 Bad Request", {"error": str(exc)})
    except Exception as exc:
        return internal_error(start_response, f"推理失败: {exc}")


def main():
    host = "127.0.0.1"
    port = 8000
    print(f"Steganalysis web app is running at http://{host}:{port}")
    with make_server(host, port, app) as server:
        server.serve_forever()


if __name__ == "__main__":
    main()
