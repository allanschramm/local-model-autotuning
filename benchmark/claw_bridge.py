"""One task's upstream FastAPI apps over private JSON-lines stdin/stdout."""

import contextlib
import importlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from urllib.parse import urlsplit


def main():
    import yaml
    from fastapi.testclient import TestClient

    root = Path(sys.argv[1]).resolve()
    task_id = sys.argv[2]
    task = yaml.safe_load(
        (root / "upstream/tasks" / task_id / "task.yaml").read_text(encoding="utf-8")
    )
    live = any(s["name"] == "web_real" for s in task["services"])
    if live and (sys.argv[3] != "live" or not os.environ.get("SERP_DEV_KEY")):
        raise RuntimeError(
            "live_web_disabled: this task requires explicit --allow-live-web and SERP_DEV_KEY"
        )
    os.environ["ERROR_RATE"] = "0"
    os.environ["WEB_REAL_CACHE_DIR"] = str(Path.cwd() / "web-cache")
    if task.get("environment", {}).get("mock_today"):
        os.environ["MOCK_TODAY"] = task["environment"]["mock_today"]
    sys.path.insert(0, str(root / "upstream"))
    spec = importlib.util.spec_from_file_location("legacy_score", root / "legacy_score.py")
    scoring = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scoring)
    endpoints = {e["tool_name"]: e for e in task["tool_endpoints"]}
    calls = []
    clients = {}
    with contextlib.ExitStack() as stack:
        for service in task["services"]:
            for key, value in service.get("env", {}).items():
                # Fixture references are relative to the frozen upstream root.
                path = root / "upstream" / str(value)
                if not path.is_file():
                    raise FileNotFoundError(f"Missing declared fixture: {value}")
                os.environ[key] = str(path)
            sys.path.insert(0, str(root / "upstream/mock_services" / service["name"]))
            module = importlib.import_module(f"mock_services.{service['name']}.server")
            client = stack.enter_context(TestClient(module.app, raise_server_exceptions=False))
            clients[urlsplit(service["health_check"]).netloc] = client
            response = client.request(
                service.get("health_check_method", "GET"),
                urlsplit(service["health_check"]).path,
                headers={"X-Health-Check": "1"},
            )
            if response.status_code >= 500:
                raise RuntimeError(
                    f"Service health failed: {service['name']} {response.status_code}"
                )
            if service.get("reset_endpoint"):
                response = client.post(urlsplit(service["reset_endpoint"]).path, json={})
                if response.status_code >= 400:
                    raise RuntimeError(f"Service reset failed: {service['name']}")
        emit({"ok": True, "ready": True})
        for line in sys.stdin:
            request = json.loads(line)
            if request["op"] == "call":
                endpoint = endpoints[request["name"]]
                url = urlsplit(endpoint["url"])
                if url.netloc not in clients or url.query or url.fragment:
                    raise ValueError("Tool endpoint is outside declared services")
                response = clients[url.netloc].request(
                    endpoint.get("method", "POST"), url.path, json=request["arguments"]
                )
                if response.status_code >= 500:
                    raise RuntimeError(f"Mock service failed: HTTP {response.status_code}")
                result = response.json()
                calls.append(
                    {
                        "tool": request["name"],
                        "arguments": request["arguments"],
                        "result": result,
                        "status": response.status_code,
                    }
                )
                emit({"ok": True, "result": calls[-1]})
            elif request["op"] == "score":
                result = scoring.score_task(
                    task, request["text"], calls, root / "upstream/tasks" / task_id
                )
                names = {c["tool"] for c in calls}
                checks = task.get("safety_checks", [])
                result["safety_violations"] = [
                    c
                    for c in checks
                    if c["type"] in ("tool_not_called", "wrong_action") and c["tool_name"] in names
                ]
                result["unevaluated_safety_checks"] = [
                    c for c in checks if c["type"] not in ("tool_not_called", "wrong_action")
                ]
                emit({"ok": True, "result": result})
            else:
                raise ValueError("Unknown bridge operation")


def emit(value):
    print(json.dumps(value, ensure_ascii=True), file=PROTOCOL_OUT, flush=True)


if __name__ == "__main__":
    PROTOCOL_OUT = sys.stdout
    try:
        with contextlib.redirect_stdout(sys.stderr):
            main()
    except Exception as exc:
        emit({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
        sys.exit(1)
