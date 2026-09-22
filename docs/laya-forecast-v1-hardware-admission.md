# Pre-scoring L40S memory admission correction

The qualified protocol requests a nominal 48-GB-or-larger GPU. The rented L40S
advertises 48 GB, but PyTorch reports 47,665,709,056 usable bytes (about 44.4 GiB).
The original runner incorrectly requires at least 45 GiB. This was detected by
reading device properties before any checkpoint load or task prediction.

Keep the original input archive and all frozen files unchanged. The explicit
admission entrypoint generates a retained source overlay that changes exactly one
byte: the minimum memory constant from 45 to 44 GiB. It accepts only the witnessed
L40S device, verifies the original source hash, refuses an existing scoring run,
records both source hashes and the actual memory, then executes the overlay.

Models, data, native calibration, numeric checks, cohort, timing, deadlines and
the $6 stage hold are unchanged. This is a preflight hardware-reader correction
within the existing rental, not a new rental, failed model-run retry, or relaxed
quality gate. Both original and executed sources remain in the recovered evidence.
