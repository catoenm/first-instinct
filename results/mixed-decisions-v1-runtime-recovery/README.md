# Runtime qualification and recovery

This is infrastructure evidence for the existing mixed-decisions-v1 experiment,
not a model-performance result. The same six-arm comparison continues from the
original supervised 9B checkpoint. Training questions, case schedules and the
learning recipe are unchanged.

The first rental failed because its upload omitted two startup test modules;
its recovery is recorded in `../mixed-decisions-v1-startup-recovery`. The second
passed all 11 tests but stopped at the first SQLite reference during execution
qualification. Neither attempted a 9B optimizer update. The second archive's
164 files were recovered and verified before its pod was deleted. Combined
estimated compute for both failures was $0.8994, excluding storage. The third
attempt shares the original shutdown deadline and $40 pilot allocation, with
$2 reserved for the failed attempts and at most $38 remaining for this attempt.

The old checker did not preserve the actual mismatching cloud receipt. A local
reproduction across SQLite 3.46.1 and 3.53.1 found the same kind of checksum
mismatch while every decoded value, observation and reward agreed. This is
supporting evidence, not a recovered record of that failed cloud execution.

SQLite records its writer version in four header bytes, at offsets 96–99.
[SQLite file-format specification](https://www.sqlite.org/fileformat.html#the_database_header).
The new comparison allows only those bytes to differ after a successful
database mutation. It reads the actual file, verifies its raw checksum against
the receipt, and hashes an in-memory copy with the reference writer version.
The resulting full-file checksum must match the reference. Every other byte,
decoded value, observation, reward, initial file and protected file stays exact.
The audit never modifies the real database. Read-only branches get no exception.

Local qualification replayed 36 existing SQLite alternatives, executing 108
commands. All passed; 32 had only this writer-version difference. Negative tests
reject changes to other metadata, rows, protected files, rewards and read-only
branches. The exact 213-file upload was extracted and passed all 15 startup tests
before the next rental.

The third cloud attempt also passed all 15 tests, 252 shell alternatives with
744 commands, and 138 application alternatives with 1,062 top-level calls.
The shell check recorded 32 writer-version-only differences. These are repeated
qualification executions of existing examples, not new underlying tasks or
training questions. The cloud files here are a verified qualification snapshot;
they do not claim that the active training run's full archive has been recovered.

`freeze.json` and `execution-attempt.json` preserve the exact source, data and
checkpoint lineage. `hashes.json` covers the copied evidence files. Current
training status is tracked separately in `../mixed-decisions-v1-launch.json`;
model results require optimizer ledgers and held-out evaluation.

The independent `tool_lab.mixed_accounting` reporter was added after the active
upload and does not change its frozen training code. Four tests cover rollback,
repeated presentations, partial batches and missing or duplicated receipts.
`local-mechanics-accounting.json` checks it against the previously completed tiny
network experiment: one accepted update per arm. These are local engineering
counts, not 9B training counts. Run the reporter against recovered cloud artifacts
to obtain the latter; prepared schedules are never substituted for actual use.
