"""Preserve parent behavior without treating predictions as outcome labels."""
import torch


def reference_kl(logits, rows, references):
    terms=[]
    for i,row in enumerate(rows):
        n=len(row['option_ids']);q=torch.tensor(references[row['id']],dtype=torch.float32,device=logits.device)
        if q.shape!=(n,) or not torch.isfinite(q).all() or (q<0).any() or abs(float(q.sum())-1)>1e-5:
            raise ValueError('Invalid frozen reference distribution')
        logp=logits[i,:n].float().log_softmax(-1)
        if not torch.isfinite(logp).all():raise ValueError('Nonfinite offered logits')
        terms.append((q*(q.clamp_min(1e-30).log()-logp)).sum())
    return torch.stack(terms)
