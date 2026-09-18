"""Actual two-process distributed gradients, not an algebra-only mock.

No downloads or accelerator are needed. A small causal transformer follows the
same batch/score/partial-batch/no_sync path as general_lab.train. Dropout is zero
so its gradients and one AdamW update must match a single global-batch model.
"""

from collections import namedtuple
from contextlib import nullcontext
import copy
from datetime import timedelta
import json
from pathlib import Path
import tempfile
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from general_lab.train import partition, per_row_loss
from scale_lab.model import batch, score


Output = namedtuple("Output", "logits")


class TinyCausalTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(41, 16)
        self.positions = torch.nn.Embedding(32, 16)
        layer = torch.nn.TransformerEncoderLayer(
            d_model=16, nhead=4, dim_feedforward=32, dropout=0., batch_first=True)
        self.encoder = torch.nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
        self.output = torch.nn.Linear(16, 41)

    def forward(self, input_ids, attention_mask, use_cache=False, logits_to_keep=1):
        # Position numbering depends on real tokens, independent of left padding
        # and therefore consistent between a global batch and smaller shards.
        positions = (attention_mask.cumsum(-1) - 1).clamp_min(0)
        hidden = self.embedding(input_ids) + self.positions(positions)
        length = input_ids.shape[1]
        causal = torch.ones(length, length, dtype=torch.bool, device=input_ids.device).triu(1)
        hidden = self.encoder(hidden, mask=causal, src_key_padding_mask=~attention_mask.bool())
        return Output(self.output(hidden[:, -logits_to_keep:, :]))


def _rows(count):
    rows = []
    for i in range(count):
        size = 2 + i % 4
        valid = [i % size]
        if i % 3 == 0 and size > 2:
            valid = sorted(set(valid + [(i + 1) % size]))
        rows.append({"id": str(i), "task": "tiny", "group_id": str(i),
                     "input_ids": [1 + (i * 7 + j) % 39 for j in range(2 + i % 7)],
                     "option_ids": [str(j) for j in range(size)], "target_indices": valid})
    return rows


def _worker(rank, rendezvous, report_path):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", rank=rank, world_size=2,
                            init_method="file://" + rendezvous, timeout=timedelta(seconds=90))
    reports = []
    try:
        # n=1 gives one rank no real examples. n=7/13 require both accumulation
        # and padding. n=8 is an exactly full global batch.
        for count in (1, 3, 7, 8, 13):
            torch.manual_seed(413)
            # Double precision avoids AdamW magnifying roundoff in theoretically
            # zero attention-key-bias gradients. score() still uses the actual
            # production float32 constrained-logit loss path.
            raw = TinyCausalTransformer().double()
            reference = copy.deepcopy(raw)
            distributed = DistributedDataParallel(raw, broadcast_buffers=False)
            optimizer = torch.optim.AdamW(raw.parameters(), lr=.003, weight_decay=.01)
            ref_optimizer = torch.optim.AdamW(reference.parameters(), lr=.003, weight_decay=.01)
            rows = _rows(count)
            label_ids, pad_id = [2, 4, 6, 8, 10], 0

            if rank == 0:
                inputs, labels, option_mask, valid = batch(rows, label_ids, pad_id, "cpu")
                ref_loss = per_row_loss(score(reference, inputs, labels, option_mask), valid).mean()
                ref_loss.backward()

            local = partition(list(range(count)), 2, 2)[rank]
            total_loss = torch.zeros(())
            for offset in range(0, len(local), 2):
                items = local[offset:offset + 2]
                chunk = [rows[i] for i, _ in items]
                weights = torch.tensor([weight for _, weight in items])
                inputs, labels, option_mask, valid = batch(chunk, label_ids, pad_id, "cpu")
                synchronize = offset + 2 >= len(local)
                with distributed.no_sync() if not synchronize else nullcontext():
                    losses = per_row_loss(score(distributed, inputs, labels, option_mask), valid)
                    (losses.mul(weights).sum() * 2 / count).backward()
                total_loss += losses.detach().mul(weights).sum() / count
            dist.all_reduce(total_loss)

            if rank == 0:
                max_gradient_error = max((a.grad - b.grad).abs().max().item()
                                         for a, b in zip(raw.parameters(), reference.parameters()))
                for actual, expected in zip(raw.parameters(), reference.parameters()):
                    torch.testing.assert_close(actual.grad, expected.grad, atol=2e-6, rtol=2e-5)
                torch.testing.assert_close(total_loss, ref_loss.detach(), atol=2e-6, rtol=2e-5)
                torch.nn.utils.clip_grad_norm_(reference.parameters(), 1., error_if_nonfinite=True)
                ref_optimizer.step()
            torch.nn.utils.clip_grad_norm_(raw.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()

            flat = torch.cat([p.detach().reshape(-1) for p in raw.parameters()])
            gathered = [torch.empty_like(flat) for _ in range(2)]
            dist.all_gather(gathered, flat)
            torch.testing.assert_close(gathered[0], gathered[1], atol=0, rtol=0)
            if rank == 0:
                max_update_error = max((a - b).abs().max().item()
                                       for a, b in zip(raw.parameters(), reference.parameters()))
                # Check the actual optimizer update as well as its gradients.
                for actual, expected in zip(raw.parameters(), reference.parameters()):
                    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
                reports.append({"rows": count, "max_gradient_error": max_gradient_error,
                                "max_update_error": max_update_error,
                                "ranks_bit_identical": True})
            dist.barrier()
            del distributed
        if rank == 0:
            Path(report_path).write_text(json.dumps(reports))
    finally:
        dist.destroy_process_group()


@unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "Requires CPU Gloo support")
class RealDistributedTrainingTest(unittest.TestCase):
    def test_two_process_transformer_matches_single_global_update(self):
        with tempfile.TemporaryDirectory(prefix="first-instinct-gloo-") as directory:
            root = Path(directory)
            mp.spawn(_worker, args=(str(root / "rendezvous"), str(root / "report.json")),
                     nprocs=2, join=True)
            reports = json.loads((root / "report.json").read_text())
            self.assertEqual([report["rows"] for report in reports], [1, 3, 7, 8, 13])
            self.assertTrue(all(report["ranks_bit_identical"] for report in reports))


if __name__ == "__main__":
    unittest.main()
