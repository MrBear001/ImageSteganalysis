import cgi
import json
import os
from typing import Callable, Iterable, Tuple
from wsgiref.simple_server import make_server

from steganalysis_service import SteganalysisService


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(ROOT_DIR, "webui", "static")
INDEX_PATH = os.path.join(ROOT_DIR, "webui", "index.html")

service = SteganalysisService(ROOT_DIR)


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

        if path == "/static/styles.css" and method == "GET":
            return file_response(start_response, os.path.join(STATIC_DIR, "styles.css"), "text/css; charset=utf-8")

        if path == "/static/app.js" and method == "GET":
            return file_response(start_response, os.path.join(STATIC_DIR, "app.js"), "application/javascript; charset=utf-8")

        if path == "/api/models" and method == "GET":
            return json_response(start_response, "200 OK", {"models": service.list_models()})

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
