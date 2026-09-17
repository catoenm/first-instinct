# First four-billion-parameter cloud pilot

The pilot completed 100 adapter updates on one H100 PCIe with 80 GB of memory.
It visited 3,200 training questions from the 13,939-question pilot pool and
processed 622,137 real input tokens. It did not train on all 153,031 questions
in the separately prepared larger dataset.

| Update | Validation accuracy | Acceptable-set log loss |
| ---: | ---: | ---: |
| Pretrained baseline | 78.57% | 0.5378 |
| 25 | 91.43% | 0.2829 |
| 50 | 95.00% | 0.2088 |
| 75 | 93.57% | 0.2221 |
| 100 | 94.29% | 0.2000 |

All checkpoints use the same 140 validation questions, 20 per task. Related
questions can share a source record. The predeclared selection criterion is
validation log loss, so update 100 is selected even though update 50 has higher
accuracy. These are selection-set results, not a final generalization claim.
The test and challenge splits have not been evaluated for this model.

Training took 552 seconds including model loading, initial validation and kernel
warm-up, but excluding machine setup and artifact packaging. Peak allocated GPU
memory was 19,244,545,536 bytes. The effective batch was 32 questions, implemented
as two batches of 16. An earlier five-update setup run with batches of two was
stopped after measuring poor throughput; it is retained locally.

The training code matches commit `85132154ca98729728e6a043d4156a8d96805a3f`.
`run.json` records the model revision, trainable parameter count, package
versions, data manifest checksum and individual code hashes. `artifacts.json`
checksums the public receipts and predictions. The trained adapter and tokenizer
were copied back and all 18 artifact checksums verified. Model weights remain
local pending further evaluation; this does not change the public v0.2.0 release.

The rental was stopped and deleted after verification. The larger supervised
run was paused while the project revisits environment-backed data and
reinforcement learning for calibrated decisions. This pilot is supervised
learning and does not establish calibrated success probabilities or Jev parity.
