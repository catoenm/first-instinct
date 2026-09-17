# Running the larger experiment on Runpod

The first rental uses one H100 PCIe with 80 GB of graphics memory. It is a
personal Runpod rental; it does not use an employer's infrastructure. The
observed GPU rate on September 17, 2026 was US$2.89/hour. Check the rate shown
for your own rental; storage is billed separately.

The working image is pinned by digest:

```text
runpod/pytorch@sha256:23770e0dc9d8340b2f806e19256c00b357d235789c8daebd642f56545c27e093
```

The running image reports Python 3.12.13, PyTorch 2.13.0 with CUDA 12.9, and
provides a CUDA compiler. The rental has a 40 GB container disk and a 50 GB
volume mounted at `/workspace`. Keep model caches and the environment on the
container disk: the attached volume was substantially slower for installing
thousands of package files. Save checkpoints under `/workspace`.

## Prepare locally

Follow [the larger-model guide](larger-model.md) to build and tokenize data,
then package only the required files:

```bash
python -m scale_lab.bundle --data output/scale-pilot-qwen35 \
  --output output/scale-pilot-job.tar.gz
```

Copy that archive to the Pod over SSH. Verify its SHA-256 hash against the
companion manifest before extracting it. Keep account keys in a private local
file or credential manager; never put them in a bundle or repository.

## Install on the rental

Create `/workspace/first-instinct`, extract the verified archive there, and run:

```bash
cd /workspace/first-instinct
python -m venv --system-site-packages /opt/first-instinct-venv
source /opt/first-instinct-venv/bin/activate
python -m pip install -r requirements-scale-cuda.txt
export HF_HOME=/opt/hf-cache
export TOKENIZERS_PARALLELISM=false
```

The image contains `pygobject` without its `pycairo` dependency. On this image
the following repaired that inherited dependency before `pip check`:

```bash
apt-get update
apt-get install -y --no-install-recommends libcairo2-dev pkg-config
python -m pip install pycairo
python -m pip check
```

The optional optimized linear-attention kernels can be installed separately:

```bash
MAX_JOBS=8 python -m pip install flash-linear-attention==0.5.2 \
  causal-conv1d==1.7.0 --no-build-isolation -c requirements-scale-cuda.txt
python -m pip check
```

Building these kernels from source takes several minutes. Do not start a
training process while the same environment is still being modified.

## Start with a measured pilot

```bash
python -m scale_lab.train --data data --output runs/pilot-01 \
  --device cuda --max-steps 100 --max-hours 0.5 \
  --batch-size 2 --accumulation 16 --validation-per-task 20 --eval-every 25
```

Check `run.json`, `training.jsonl` and the baseline/validation predictions.
The maximum time includes model loading and the initial evaluation. The initial
checkpoint remains eligible when training fails to improve validation loss.
Save the full log: first-call kernel compilation can obscure steady training
throughput. A successful process exit alone does not establish improvement.

Set a separate provider stop deadline **before** launching training. The
training command does not control rental billing. The bundled `runpodctl`
command is not necessarily authenticated inside a custom image; verify any
provider stop mechanism with a read-only request before relying on it.

Copy checkpoints, predictions, manifests and logs back locally, verify their
hashes, then stop the Pod. Delete the rental's storage only after verifying those
copies. A stopped Pod may continue to incur storage charges.

This is adapter training on a pretrained model. The current larger-model
pipeline does not yet run reinforcement learning or certify calibrated
success probabilities.
