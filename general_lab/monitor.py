"""Read existing training receipts into TensorBoard without touching a trainer.

The snapshot command uses only the standard library, including on the rental.
The export command needs requirements-monitor.txt on the monitoring computer.
Only training and validation measurements are charted; held-out results remain
in the evaluation artifacts. Event wall times are ingestion times: use Step as
the horizontal axis and progress/elapsed_minutes for recorded training duration.
"""

import argparse
import csv
import json
import math
from pathlib import Path
import subprocess
import time


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def compact(value):
    if isinstance(value, dict):
        return {k: compact(v) for k, v in value.items()
                if k not in {"by_task", "pure_policy_language_gradient"}}
    if isinstance(value, list):
        return [compact(v) for v in value]
    return value


def trace_rows(path):
    try:
        contents = path.read_text()
    except FileNotFoundError:
        return []
    lines = contents.splitlines(keepends=True)
    # A writer may currently be appending the last record. Retry it next time.
    if lines and not lines[-1].endswith("\n"):
        lines.pop()
    return [compact(json.loads(line)) for line in lines if line.strip()]


def snapshot(root, gpus=False):
    root = Path(root)
    result = {"observed_at": time.time(), "runs": {}, "pipeline": read_json(root / "pipeline.json")}
    for folder in sorted((root / "runs").glob("*")):
        if not folder.is_dir() or folder.name.startswith("pilot"):
            continue
        receipt = read_json(folder / "run.json")
        baseline = read_json(folder / "baseline-metrics.json")
        if "validation" in baseline:
            baseline = {"validation": baseline["validation"]}
        # No prompt, target, replay, checkpoint tensor or provider credential.
        result["runs"][folder.name] = {
            "receipt": {k: receipt[k] for k in ("status", "config", "model", "training_rows", "effective_batch",
                        "trainable_parameters", "trainable_language_parameters", "steps", "updates",
                        "visits", "tokens", "seconds", "best_step", "selected_update") if k in receipt},
            "baseline": compact(baseline),
            "training": trace_rows(folder / "training.jsonl"),
        }
    if gpus:
        command = ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu", "--format=csv,noheader,nounits"]
        try:
            output = subprocess.check_output(command, text=True, timeout=10)
            result["gpus"] = [dict(zip(("index", "utilization_percent", "memory_mib", "total_memory_mib", "temperature_c"),
                                      [float(x.strip()) for x in row])) for row in csv.reader(output.splitlines())]
        except (OSError, subprocess.SubprocessError, ValueError) as error:
            result["gpu_error"] = type(error).__name__
    return result


def validation_scalars(measured):
    output = {}
    for source, name in (("macro_log_loss", "loss/validation"), ("macro_accuracy", "accuracy/validation_macro")):
        if source in measured:
            output[name] = measured[source]
    for source, name in (("policy_expected_reward", "reward/validation_expected"),
                         ("oracle_planner_expected_reward", "reward/validation_oracle"),
                         ("expected_regret_to_oracle_planner", "reward/validation_regret")):
        if source in measured.get("policy", {}):
            output[name] = measured["policy"][source]
    for source, name in (("brier", "forecast/validation_brier"), ("log_loss", "forecast/validation_log_loss"),
                         ("mean_squared_error_to_exact_posterior", "forecast/validation_posterior_error")):
        if source in measured.get("forecast", {}):
            output[name] = measured["forecast"][source]
    return output


def training_scalars(event, receipt, previous=None):
    output = {}
    mapping = {"loss": "loss/train", "grad_norm": "optimization/gradient_norm",
               "learning_rate": "optimization/learning_rate", "visits": "progress/examples_seen",
               "tokens": "progress/input_tokens_seen", "mean_sampled_return": "reward/sampled_return",
               "transitions": "progress/transitions_in_update"}
    for source, target in mapping.items():
        if source in event:
            output[target] = event[source]
    if "seconds" in event:
        output["progress/elapsed_minutes"] = event["seconds"] / 60
    if "visits" in event and receipt.get("training_rows"):
        planned = receipt["training_rows"] * receipt.get("config", {}).get("epochs", 1)
        output["progress/percent"] = 100 * event["visits"] / planned
    if "update" in event and receipt.get("config", {}).get("max_updates"):
        output["progress/percent"] = 100 * event["update"] / receipt["config"]["max_updates"]
    if previous and event.get("seconds", 0) > previous.get("seconds", 0):
        duration = event["seconds"] - previous["seconds"]
        for key in ("visits", "tokens"):
            if key in event and key in previous:
                output["throughput/" + ("examples_per_second" if key == "visits" else "tokens_per_second")] = (event[key] - previous[key]) / duration
    epochs = event.get("epochs", [])
    for key in ("policy_loss", "value_loss", "forecast_loss", "replay_loss", "entropy", "clip_fraction", "approximate_kl", "gradient_norm"):
        values = [row[key] for row in epochs if key in row]
        if values:
            output["optimization/" + key] = sum(values) / len(values)
    output.update(validation_scalars(event.get("validation", {})))
    return output


class Exporter:
    def __init__(self, logdir):
        from tensorboard.summary.writer.event_file_writer import EventFileWriter
        self.factory = EventFileWriter
        self.root = Path(logdir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.state_path = self.root / "export-state.json"
        self.state = read_json(self.state_path)
        self.writers = {}

    def writer(self, name):
        if name not in self.writers:
            if Path(name).name != name or name in ("", ".", ".."):
                raise ValueError("Unsafe run name")
            self.writers[name] = self.factory(str(self.root / name))
        return self.writers[name]

    def scalars(self, name, values, step, wall_time):
        from tensorboard.compat.proto.event_pb2 import Event
        from tensorboard.compat.proto.summary_pb2 import Summary
        selected = [Summary.Value(tag=key, simple_value=float(value)) for key, value in values.items()
                    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)]
        if selected:
            self.writer(name).add_event(Event(wall_time=wall_time, step=int(step), summary=Summary(value=selected)))

    def text(self, name, tag, value, step, wall_time):
        from tensorboard.compat.proto.event_pb2 import Event
        from tensorboard.compat.proto.summary_pb2 import Summary, SummaryMetadata
        from tensorboard.compat.proto.tensor_pb2 import TensorProto
        from tensorboard.compat.proto.tensor_shape_pb2 import TensorShapeProto
        metadata = SummaryMetadata(plugin_data=SummaryMetadata.PluginData(plugin_name="text"))
        tensor = TensorProto(dtype=7, string_val=[value.encode()],
                             tensor_shape=TensorShapeProto(dim=[TensorShapeProto.Dim(size=1)]))
        self.writer(name).add_event(Event(wall_time=wall_time, step=int(step),
            summary=Summary(value=[Summary.Value(tag=tag, metadata=metadata, tensor=tensor)])))

    def export(self, data):
        observed = data.get("observed_at", time.time())
        for name, run in data.get("runs", {}).items():
            receipt = run["receipt"]
            state = self.state.setdefault(name, {"last_step": -1})
            baseline = run.get("baseline", {})
            if baseline and not state.get("baseline"):
                self.scalars(name, validation_scalars(baseline.get("validation", baseline)), 0, observed)
                state["baseline"] = True
            previous = None
            for event in run.get("training", []):
                step = event.get("step", event.get("update"))
                if step is not None and step > state["last_step"]:
                    self.scalars(name, training_scalars(event, receipt, previous), step, observed)
                    state["last_step"] = step
                previous = event
            status = receipt.get("status", "starting")
            if status != state.get("status"):
                self.text(name, "status/run", status, max(0, state["last_step"]), observed)
                state["status"] = status
        machine = self.state.setdefault("machine", {"poll": 0})
        machine["poll"] += 1
        values = {}
        for gpu in data.get("gpus", []):
            prefix = f"gpu_{int(gpu['index'])}/"
            values.update({prefix + "utilization_percent": gpu["utilization_percent"],
                           prefix + "memory_gib": gpu["memory_mib"] / 1024,
                           prefix + "temperature_c": gpu["temperature_c"]})
        billing = data.get("billing", {})
        if billing:
            values["compute/estimated_gpu_usd_excludes_storage"] = billing["estimated_compute_usd"]
            values["compute/rental_elapsed_minutes"] = billing["elapsed_seconds"] / 60
        self.scalars("machine", values, machine["poll"], observed)
        phase = data.get("pipeline", {}).get("phase", "unknown")
        self.text("machine", "status/pipeline", f"{phase}\n\nLast successful refresh: {time.strftime('%Y-%m-%d %H:%M:%S %Z', time.localtime(observed))}", machine["poll"], observed)
        self.text("machine", "status/connection", "Metrics refreshed successfully.", machine["poll"], observed)
        self.flush()

    def flush(self):
        for writer in self.writers.values():
            writer.flush()
        temporary = self.state_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.state, indent=2) + "\n")
        temporary.replace(self.state_path)

    def close(self):
        self.flush()
        for writer in self.writers.values():
            writer.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    read = commands.add_parser("snapshot")
    read.add_argument("--root", type=Path, required=True)
    read.add_argument("--gpus", action="store_true")
    export = commands.add_parser("export")
    export.add_argument("--snapshot", type=Path, required=True)
    export.add_argument("--logdir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "snapshot":
        print(json.dumps(snapshot(args.root, args.gpus), allow_nan=False))
    else:
        exporter = Exporter(args.logdir)
        try:
            exporter.export(json.loads(args.snapshot.read_text()))
        finally:
            exporter.close()


if __name__ == "__main__":
    main()
