"""Serve user-defined decisions from one resident saved adapter on localhost."""

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import secrets
import threading
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
WEB = Path(__file__).with_name("web")
EXAMPLE = ROOT / "examples/general-decisions.json"
MAX_REQUEST_BYTES = 256 * 1024
ASSETS = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/snake.mjs": ("snake.mjs", "text/javascript; charset=utf-8"),
    "/style.css": ("style.css", "text/css; charset=utf-8"),
}


def strict_json(data):
    """Reject ambiguous objects and nonfinite numbers, including float overflow."""
    def constant(_value):
        raise ValueError("JSON numbers must be finite")

    def floating(value):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("JSON numbers must be finite")
        return number

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON object keys are not allowed")
            result[key] = value
        return result

    try:
        if isinstance(data, bytes):
            data = data.decode("utf-8")
        return json.loads(data, parse_constant=constant, parse_float=floating,
                          object_pairs_hook=unique)
    except RecursionError as error:
        raise ValueError("JSON nesting is too deep") from error


def read_receipt(run):
    receipt = strict_json((Path(run) / "run.json").read_bytes())
    if not isinstance(receipt, dict):
        raise ValueError("The saved run receipt must be an object")
    return receipt


def checkpoint_metadata(receipt):
    """Describe the selected adapter, rather than the run's latest update."""
    status = receipt.get("status", "unknown")
    if not isinstance(status, str) or not status:
        raise ValueError("The run receipt needs a valid status")
    reinforcement = ("selected_update" in receipt or "optimizer_steps" in receipt
                     or "PPO" in str(receipt.get("method", "")))
    key = "selected_update" if reinforcement else "best_step"
    step = receipt.get(key)
    if step is not None and (type(step) is not int or step < 0):
        raise ValueError(f"The run receipt has an invalid {key}")

    preview = status not in ("complete", "bounded_stop") or step is None
    if reinforcement and step == 0:
        kind = "supervised"
        label = "Supervised starting checkpoint (selected update 0; no RL updates selected)"
    elif reinforcement:
        kind = "reinforcement"
        label = (f"Reinforcement checkpoint, update {step}" if step is not None
                 else "Reinforcement run checkpoint; selected update not recorded")
    else:
        kind = "supervised"
        label = (f"Supervised checkpoint, step {step}" if step is not None
                 else "Supervised checkpoint; selected step not recorded")
    if preview:
        kind = "preview"
        label = "Training preview: " + label[0].lower() + label[1:]
    elif status == "bounded_stop":
        label += " (bounded run)"
    return {"run_status": status, "step": step, "kind": kind, "label": label}


class Demo:
    def __init__(self, predictor, receipt=None):
        self.predictor = predictor
        if receipt is None:
            if predictor.run is None:
                raise ValueError("A saved run is required")
            receipt = read_receipt(predictor.run)
        actual = predictor.spec
        recorded = receipt.get("model", {})
        if any(not isinstance(actual.get(key), str) or not actual[key]
               or actual[key] != recorded.get(key) for key in ("id", "revision")):
            raise ValueError("Loaded model and saved run receipt do not match")
        checkpoint = checkpoint_metadata(receipt)
        selected_step = getattr(predictor, "selected_step", None)
        if "selected_update" not in receipt and selected_step is not None and selected_step != checkpoint["step"]:
            raise ValueError("Loaded checkpoint and saved run receipt do not match")
        if type(predictor.max_tokens) is not int or predictor.max_tokens <= 0:
            raise ValueError("Token limit must be positive")
        # Snapshot metadata once: a later edit to run.json must not relabel the
        # adapter already resident in memory. No submitted state is retained.
        self.metadata = {"model": actual["id"], "model_revision": actual["revision"],
                         "checkpoint": checkpoint, "device": str(predictor.device)}
        self.limits = {"max_questions": 32, "max_options": 36, "max_tokens": predictor.max_tokens}
        self.csrf_token = secrets.token_urlsafe(32)
        self.gate = threading.Lock()

    def status(self):
        return {"ready": True, **self.metadata, "limits": self.limits,
                "csrf_token": self.csrf_token}

    def answer(self, payload):
        from .interface import answer
        return {**answer(payload, self.predictor), **self.metadata}


def handler_for(demo):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, _format, *args):
            # Access logs can contain caller-controlled paths. Do not log
            # submitted state, questions, or model exception messages.
            pass

        def reply(self, status, body, content_type="application/json; charset=utf-8"):
            if not isinstance(body, bytes):
                body = json.dumps(body, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'self'; script-src 'self'; object-src 'none'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def local_request(self):
            allowed = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
            hosts = self.headers.get_all("Host", [])
            origins = self.headers.get_all("Origin", [])
            return (len(hosts) == 1 and hosts[0] in allowed and len(origins) <= 1
                    and (not origins or origins[0] in {"http://" + host for host in allowed}))

        def route(self):
            if not self.path.startswith("/") or self.path.startswith("//"):
                return None
            try:
                return urlsplit(self.path).path
            except ValueError:
                return None

        def do_GET(self):
            if not self.local_request():
                self.reply(403, {"error": "Use the local demo address"})
                return
            path = self.route()
            if path in ASSETS:
                name, content_type = ASSETS[path]
                try:
                    self.reply(200, (WEB / name).read_bytes(), content_type)
                except OSError:
                    self.reply(404, {"error": "Demo asset is unavailable"})
            elif path == "/api/status":
                self.reply(200, demo.status())
            elif path == "/api/example":
                self.reply(200, EXAMPLE.read_bytes())
            else:
                self.reply(404, {"error": "Not found"})

        def do_POST(self):
            if not self.local_request():
                self.reply(403, {"error": "Use the local demo address"})
                return
            if self.route() != "/api/answer":
                self.reply(404, {"error": "Not found"})
                return
            tokens = self.headers.get_all("X-CSRF-Token", [])
            if len(tokens) != 1 or not tokens[0].isascii() or not secrets.compare_digest(tokens[0], demo.csrf_token):
                self.reply(403, {"error": "Refresh the page to obtain a valid request token"})
                return
            types = self.headers.get_all("Content-Type", [])
            if len(types) != 1 or self.headers.get_content_type() != "application/json":
                self.reply(415, {"error": "Send application/json"})
                return
            try:
                lengths = self.headers.get_all("Content-Length", [])
                if self.headers.get_all("Transfer-Encoding") or len(lengths) != 1 or not lengths[0].isascii() or not lengths[0].isdigit():
                    raise ValueError("Send one Content-Length and no Transfer-Encoding")
                size = int(lengths[0])
                if not 0 < size <= MAX_REQUEST_BYTES:
                    raise ValueError("Request must contain 1–262144 bytes")
                self.connection.settimeout(10)
                body = self.rfile.read(size)
                if len(body) != size:
                    raise ValueError("Request body is incomplete")
                payload = strict_json(body)
                if not isinstance(payload, dict) or set(payload) != {"state", "questions"}:
                    raise ValueError("Provide an object containing state and questions")
            except (ValueError, TimeoutError, OSError) as error:
                self.reply(400, {"error": str(error)})
                return
            if not demo.gate.acquire(blocking=False):
                self.reply(429, {"error": "The model is busy. Try again shortly."})
                return
            try:
                self.reply(200, demo.answer(payload))
            except ValueError as error:
                self.reply(400, {"error": str(error)})
            except Exception:
                self.reply(500, {"error": "Prediction failed. Try again or restart the local demo."})
            finally:
                demo.gate.release()

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--model", default="qwen35-9b")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--max-tokens", type=int, default=1536)
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error("Use a port between 1024 and 65535")
    if args.max_tokens <= 0:
        parser.error("Token limit must be positive")
    from scale_lab.common import MODELS
    if args.model not in MODELS:
        parser.error("Choose a supported model: " + ", ".join(MODELS))
    receipt = read_receipt(args.run)
    from scale_lab.infer import Predictor
    predictor = Predictor(args.model, args.run, args.max_tokens, args.device)
    demo = Demo(predictor, receipt)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler_for(demo))
    print(f"General decision demo ready at http://127.0.0.1:{args.port}", flush=True)
    print(demo.metadata["checkpoint"]["label"], flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
