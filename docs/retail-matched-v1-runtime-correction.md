# Preserve the virtual environment when launching retail workers

The first matched-distribution attempt failed during its first worker import:
`ModuleNotFoundError: toml`. It used the base Python executable instead of the
existing qualified retail virtual environment. The worker's journal contains
zero tool calls, no actor decision was made, and no outcome label was produced.
Keep that attempt, its logs and its original source/data freeze unchanged.

The retry uses the existing `.local/tau2-venv/bin/python` path. Do not resolve its
symbolic link to the base interpreter: Python chooses its virtual environment
from the invoked path. Bind the absolute invoked path, interpreter identity,
virtual-environment configuration and installed package versions in a new manifest.
Check that identity again before execution. No dependency installation, upstream
code change, new curriculum, changed label or enlarged execution ceiling occurs.

The correction wrapper accepts only a preserved import failure with no completed
episode and no started tool call. It refuses to overwrite either output directory.
It inherits and verifies the original frozen protocol, controls and schedule,
records the prior failed freeze hash, and writes a separate corrected manifest
before executing. All 252 planned resets must still pass the independent audit.
There is no automatic retry loop. A new failure stops and remains evidence.
