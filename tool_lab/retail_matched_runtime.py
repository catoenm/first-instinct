"""Explicit environment-only correction of the preserved zero-call import failure."""
import argparse
import json
from pathlib import Path
import subprocess

from scale_lab.common import file_hash, write_json
from tool_lab.retail_matched import execute
from tool_lab.telecom_hidden_causes import verify_sources


def invoked_python(path):
    # Resolving the executable symlink loses Python's virtual-environment lookup.
    path = path.absolute()
    if not path.is_file() or not (path.parent.parent / "pyvenv.cfg").is_file():
        raise ValueError("An existing virtual-environment entry point is required")
    return path


def runtime_identity(python):
    command = """import importlib.metadata as m,json,sys
names=['toml','pydantic','loguru','litellm']
print(json.dumps(dict(executable=sys.executable,prefix=sys.prefix,base_prefix=sys.base_prefix,
version=sys.version,required={n:m.version(n) for n in names},
packages=sorted((d.metadata['Name'],d.version) for d in m.distributions()))))"""
    result = json.loads(subprocess.check_output([str(python), "-c", command], text=True, timeout=30))
    if result["prefix"] == result["base_prefix"] or Path(result["executable"]).absolute() != python:
        raise ValueError("Worker is not using the invoked virtual environment")
    return result


def prepare(previous, output, python):
    if output.exists():
        raise ValueError("Preserve earlier outputs")
    plan = json.loads((previous / "freeze-private.json").read_text())
    verify_sources(plan)
    failure = json.loads((previous / "failure.json").read_text())
    attempts = json.loads((previous / "attempts.json").read_text())
    logs = list(previous.glob("*.log"))
    if (failure.get("type") != "RuntimeError" or len(attempts) != 1 or
            attempts[0].get("status") == "complete" or
            not any("ModuleNotFoundError: No module named 'toml'" in p.read_text() for p in logs)):
        raise ValueError("Not the preserved initial import failure")
    if any(p.read_text().strip() for p in previous.glob("*-journal.jsonl")):
        raise ValueError("Earlier attempt executed a command")
    python = invoked_python(python)
    runtime = runtime_identity(python)
    plan.update(worker_python=str(python), runtime_identity=runtime,
                correction="preserve_virtual_environment_entry_point",
                previous_failed_freeze_sha256=file_hash(previous / "freeze-private.json"))
    for path in [Path(__file__), Path("docs/retail-matched-v1-runtime-correction.md"),
                 Path("tests/test_retail_matched_runtime.py"), python.parent.parent / "pyvenv.cfg",
                 previous / "freeze-private.json", previous / "failure.json", previous / "attempts.json", *logs]:
        plan["paths"][str(path.absolute())] = file_hash(path)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "freeze-private.json", plan)
    return dict(status="runtime_correction_frozen", prior_executed_tool_calls=0,
                same_reset_schedule=True, dependency_installations=0,
                freeze_sha256=file_hash(output / "freeze-private.json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "execute"))
    parser.add_argument("--previous", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python", type=Path)
    args = parser.parse_args()
    if args.mode == "prepare":
        print(json.dumps(prepare(args.previous, args.output, args.python), indent=2))
    else:
        plan = json.loads((args.output / "freeze-private.json").read_text())
        verify_sources(plan)
        python = invoked_python(Path(plan["worker_python"]))
        if runtime_identity(python) != plan["runtime_identity"]:
            raise ValueError("Worker environment changed")
        execute(args.output)
