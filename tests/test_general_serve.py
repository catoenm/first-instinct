"""Exercise the local decision API without loading a model or opening a socket."""

import contextlib
import copy
from http.client import HTTPResponse
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from general_lab.serve import (Demo, MAX_REQUEST_BYTES, checkpoint_metadata,
                               handler_for, main, strict_json)


class FakePredictor:
    spec = {"id": "fixture/model", "revision": "pinned-test-revision"}
    max_tokens = 1536
    device = "cpu"
    selected_step = 12
    run = None

    def __init__(self):
        self.events = []
        self.failure = None

    def validate(self, item):
        self.events.append(("validate", copy.deepcopy(item)))
        if item["question"] == "TOO_LONG":
            raise ValueError("Input has 1537 tokens; limit 1536; no truncation")
        return [1]

    def predict(self, item):
        self.events.append(("predict", copy.deepcopy(item)))
        if self.failure:
            raise self.failure
        options = item["options"]
        return {"probabilities": {o["id"]: 1 / len(options) for o in options},
                "choice": options[0]["id"], "milliseconds": 2.5}


class MemoryConnection:
    """Socket-shaped byte streams let BaseHTTPRequestHandler parse real HTTP."""
    def __init__(self, data):
        self.input = io.BytesIO(data)
        self.output = io.BytesIO()

    def makefile(self, _mode, *_args):
        return self.input

    def sendall(self, data):
        self.output.write(data)

    def settimeout(self, _seconds):
        pass


def receipt(**changes):
    return {"model": dict(FakePredictor.spec), "status": "complete", "best_step": 12, **changes}


def payload():
    return {"state": {"open_issues": 5}, "questions": {
        "ready": {"type": "binary", "instructions": "Are there five open issues?"}}}


class GeneralServeTests(unittest.TestCase):
    def setUp(self):
        self.predictor = FakePredictor()
        self.demo = Demo(self.predictor, receipt())
        self.handler = handler_for(self.demo)
        self.server = type("Server", (), {"server_port": 8766})()

    def request(self, method="GET", path="/api/status", body=None, headers=None, raw=None):
        body = json.dumps(body).encode() if raw is None and body is not None else (raw or b"")
        chosen = {"Host": "127.0.0.1:8766"}
        if method == "POST":
            chosen.update({"Content-Type": "application/json", "Content-Length": str(len(body)),
                           "X-CSRF-Token": self.demo.csrf_token})
        for key, value in (headers or {}).items():
            if value is None:
                chosen.pop(key, None)
            else:
                chosen[key] = value
        request = f"{method} {path} HTTP/1.1\r\n".encode()
        request += "".join(f"{key}: {value}\r\n" for key, value in chosen.items()).encode("latin-1")
        connection = MemoryConnection(request + b"\r\n" + body)
        self.handler(connection, ("127.0.0.1", 12345), self.server)
        response = HTTPResponse(MemoryConnection(connection.output.getvalue()))
        response.begin()
        data = response.read()
        if response.getheader("Content-Type", "").startswith("application/json"):
            data = json.loads(data)
        return response.status, response.headers, data

    def post(self, value=None, **kwargs):
        return self.request("POST", "/api/answer", payload() if value is None else value, **kwargs)

    def test_status_identifies_loaded_model_and_selected_checkpoint(self):
        status, headers, data = self.request()
        self.assertEqual(status, 200)
        self.assertEqual(data["model"], self.predictor.spec["id"])
        self.assertEqual(data["model_revision"], self.predictor.spec["revision"])
        self.assertEqual(data["checkpoint"], {
            "run_status": "complete", "step": 12, "kind": "supervised",
            "label": "Supervised checkpoint, step 12"})
        self.assertEqual(data["limits"], {"max_questions": 32, "max_options": 36, "max_tokens": 1536})
        self.assertTrue(data["ready"])
        self.assertEqual(data["device"], "cpu")
        self.assertGreaterEqual(len(data["csrf_token"]), 32)
        self.assertEqual(headers["Cache-Control"], "no-store")
        self.assertEqual(headers["X-Content-Type-Options"], "nosniff")
        self.assertIn("frame-ancestors 'none'", headers["Content-Security-Policy"])
        self.assertNotIn("Access-Control-Allow-Origin", headers)

    def test_tokens_are_unique_to_each_resident_demo(self):
        self.assertNotEqual(self.demo.csrf_token, Demo(FakePredictor(), receipt()).csrf_token)

    def test_host_and_origin_restrictions_include_read_endpoints(self):
        for headers in ({"Host": "evil.example:8766"}, {"Host": None},
                        {"Origin": "https://evil.example"}, {"Origin": "null"},
                        {"Origin": "http://localhost:8765"},
                        {"Host": "127.0.0.1:8766\r\nHost: evil.example"},
                        {"Origin": "http://localhost:8766\r\nOrigin: http://localhost:8766"}):
            with self.subTest(headers=headers):
                self.assertEqual(self.request(headers=headers)[0], 403)
                self.assertEqual(self.post(headers=headers)[0], 403)
        self.assertEqual(self.request(headers={"Host": "localhost:8766", "Origin": "http://localhost:8766"})[0], 200)
        self.assertFalse(self.predictor.events)

    def test_answer_requires_one_matching_csrf_token(self):
        for token in (None, "wrong", "é", self.demo.csrf_token + "\r\nX-CSRF-Token: extra"):
            with self.subTest(token=token):
                self.assertEqual(self.post(headers={"X-CSRF-Token": token})[0], 403)
        self.assertFalse(self.predictor.events)

    def test_json_only_and_unambiguous_length(self):
        self.assertEqual(self.post(headers={"Content-Type": "text/plain"})[0], 415)
        self.assertEqual(self.post(headers={"Content-Type": "application/json\r\nContent-Type: text/plain"})[0], 415)
        for headers in ({"Content-Length": None}, {"Content-Length": "-1"},
                        {"Content-Length": "0"}, {"Content-Length": str(MAX_REQUEST_BYTES + 1)},
                        {"Content-Length": "5\r\nContent-Length: 5"},
                        {"Content-Length": "999"}, {"Transfer-Encoding": "chunked"}):
            with self.subTest(headers=headers):
                self.assertEqual(self.post(headers=headers)[0], 400)
        self.assertFalse(self.predictor.events)

    def test_malformed_ambiguous_nonfinite_or_deep_json_never_predicts(self):
        invalid = [b"{", b"[]", b"null", b"\xff", b"{}",
                   b'{"state":"x","state":"y","questions":{}}',
                   b'{"state":{"n":NaN},"questions":{}}',
                   b'{"state":{"n":Infinity},"questions":{}}',
                   b'{"state":{"n":-Infinity},"questions":{}}',
                   b'{"state":{"n":1e999},"questions":{}}',
                   b"[" * 2000 + b"]" * 2000]
        for raw in invalid:
            with self.subTest(raw=raw[:80]):
                self.assertEqual(self.post(raw=raw)[0], 400)
        self.assertFalse(self.predictor.events)

    def test_invalid_state_question_counts_and_option_counts_never_predict(self):
        values = [[], {"state": "", "questions": {}}, payload() | {"extra": True},
                  {"state": "x", "questions": {str(i): {"type": "binary", "instructions": "Q"} for i in range(33)}}]
        for count in (1, 37):
            values.append({"state": "x", "questions": {"q": {
                "type": "choice", "instructions": "Q", "criteria": {str(i): "Option" for i in range(count)}}}})
        values.append({"state": "x", "questions": {"q": {"type": "binary", "instructions": ""}}})
        for value in values:
            with self.subTest(value=value):
                self.assertEqual(self.post(value)[0], 400)
        self.assertFalse(self.predictor.events)

    def test_later_overlength_question_prevents_all_predictions(self):
        value = payload()
        value["questions"]["later"] = {"type": "binary", "instructions": "TOO_LONG"}
        status, _, data = self.post(value)
        self.assertEqual(status, 400)
        self.assertIn("no truncation", data["error"])
        self.assertEqual([kind for kind, _ in self.predictor.events], ["validate", "validate"])

    def test_model_is_serialized_and_busy_returns_429(self):
        self.demo.gate.acquire()
        try:
            status, _, data = self.post()
            self.assertEqual(status, 429)
            self.assertIn("busy", data["error"])
            self.assertEqual(self.request()[0], 200)
            self.assertFalse(self.predictor.events)
        finally:
            self.demo.gate.release()
        self.assertEqual(self.post()[0], 200)

    def test_inference_failure_releases_gate_without_logging_prompts(self):
        self.predictor.failure = RuntimeError("PRIVATE_USER_STATE")
        logs = io.StringIO()
        with contextlib.redirect_stderr(logs), contextlib.redirect_stdout(logs):
            status, _, data = self.post()
        self.assertEqual(status, 500)
        self.assertNotIn("PRIVATE_USER_STATE", json.dumps(data))
        self.assertEqual(logs.getvalue(), "")
        self.predictor.failure = None
        self.assertEqual(self.post()[0], 200)

    def test_independent_user_questions_preserve_answer_schema(self):
        initial = payload()
        status, _, first = self.post(initial)
        self.assertEqual(status, 200)
        first_prompt = next(item for kind, item in self.predictor.events if kind == "predict")
        self.predictor.events.clear()
        expanded = copy.deepcopy(initial)
        expanded["questions"].update({
            "level": {"type": "score", "instructions": "How many open issues?", "criteria": ["None", "One to five", "Six or more"]},
            "route": {"type": "choice", "instructions": "Choose a route.", "criteria": {"custom_north": "North", "custom_south": "South"}},
        })
        status, _, result = self.post(expanded)
        self.assertEqual(status, 200)
        predictions = [item for kind, item in self.predictor.events if kind == "predict"]
        self.assertEqual(predictions[0], first_prompt)
        self.assertEqual(result["answers"]["ready"], first["answers"]["ready"])
        self.assertEqual(result["answers"]["ready"]["probability_yes"], .5)
        self.assertEqual(result["answers"]["level"]["score"], 1.)
        self.assertEqual(result["answers"]["level"]["selected_level"], 0)
        self.assertEqual(len(result["answers"]["level"]["legend"]), 3)
        self.assertEqual(result["answers"]["route"]["choice"], "custom_north")
        self.assertEqual([kind for kind, _ in self.predictor.events][:3], ["validate"] * 3)
        self.assertEqual(result["model_revision"], self.predictor.spec["revision"])
        self.assertEqual(result["checkpoint"], first["checkpoint"])
        self.assertIn("not established", result["probability_note"])
        self.assertIn("Independent prompts", result["execution"])
        self.assertNotIn("csrf_token", result)
        self.assertNotIn("state", result)

    def test_example_and_fixed_asset_allowlist(self):
        status, _, example = self.request(path="/api/example")
        self.assertEqual(status, 200)
        self.assertEqual(set(example), {"state", "questions"})
        with tempfile.TemporaryDirectory() as directory:
            for name in ("index.html", "app.js", "style.css", "snake.mjs"):
                (Path(directory) / name).write_text(name)
            with patch("general_lab.serve.WEB", Path(directory)):
                for path in ("/", "/app.js", "/style.css", "/snake.mjs"):
                    self.assertEqual(self.request(path=path)[0], 200)
                for path in ("/../run.json", "/run.json", "/index.html", "//evil.example/", "http://evil.example/"):
                    self.assertEqual(self.request(path=path)[0], 404)
        self.assertEqual(self.request("POST", "/api/other", payload())[0], 404)

    def test_receipt_and_actual_loaded_model_must_match(self):
        for key in ("id", "revision"):
            changed = receipt()
            changed["model"][key] = "not-the-loaded-model"
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "do not match"):
                Demo(FakePredictor(), changed)
        with self.assertRaisesRegex(ValueError, "checkpoint.*do not match"):
            Demo(FakePredictor(), receipt(best_step=99))

    def test_receipt_is_snapshotted_and_does_not_expose_filesystem_paths(self):
        saved = receipt(config={"output": "/PRIVATE/RUN/PATH"})
        demo = Demo(FakePredictor(), saved)
        saved["status"] = "training"
        saved["best_step"] = 88
        saved["model"]["revision"] = "new-revision"
        self.assertEqual(demo.status()["checkpoint"]["step"], 12)
        self.assertEqual(demo.status()["checkpoint"]["run_status"], "complete")
        self.assertEqual(demo.status()["model_revision"], "pinned-test-revision")
        self.assertNotIn("PRIVATE", json.dumps(demo.status()))

    def test_preview_and_reinforcement_selection_labels_are_truthful(self):
        preview = checkpoint_metadata(receipt(status="training_preview", best_step=420))
        self.assertEqual(preview["kind"], "preview")
        self.assertEqual(preview["step"], 420)
        self.assertIn("preview", preview["label"].lower())
        baseline = checkpoint_metadata(receipt(selected_update=0, updates=25, optimizer_steps=100))
        self.assertEqual(baseline["kind"], "supervised")
        self.assertEqual(baseline["step"], 0)
        self.assertIn("Supervised starting checkpoint", baseline["label"])
        self.assertIn("no RL updates selected", baseline["label"])
        selected = checkpoint_metadata(receipt(selected_update=7, updates=25))
        self.assertEqual(selected["kind"], "reinforcement")
        self.assertEqual(selected["step"], 7)
        self.assertNotIn("25", selected["label"])
        bounded = checkpoint_metadata(receipt(status="bounded_stop"))
        self.assertEqual(bounded["kind"], "supervised")
        self.assertIn("bounded", bounded["label"])
        missing = checkpoint_metadata(receipt(best_step=None))
        self.assertEqual(missing["kind"], "preview")
        self.assertIsNone(missing["step"])
        for step in (True, -1, "12", 2.5):
            with self.subTest(step=step), self.assertRaises(ValueError):
                checkpoint_metadata(receipt(best_step=step))

    def test_main_loads_one_predictor_and_binds_localhost_new_port(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "run.json").write_text(json.dumps(receipt()))
            with patch("sys.argv", ["serve", "--run", directory]), \
                    patch("scale_lab.infer.Predictor", return_value=self.predictor) as load, \
                    patch("general_lab.serve.ThreadingHTTPServer") as server, \
                    contextlib.redirect_stdout(io.StringIO()):
                main()
            load.assert_called_once_with("qwen35-9b", Path(directory), 1536, "auto")
            self.assertEqual(server.call_args.args[0], ("127.0.0.1", 8766))
            server.return_value.serve_forever.assert_called_once_with()
            server.return_value.server_close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
