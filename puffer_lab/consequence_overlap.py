"""Post-hoc exact-prompt overlap audit; never selects or retrains a checkpoint."""

import argparse
import json
from pathlib import Path

from scale_lab.common import digest, read_rows, write_json
from .consequence_train import summarize
from .consequence_report import matched_predictions


def partition(training, validation):
    by_prompt = {}
    for row in training:
        key = digest(row['input_ids'])
        if key in by_prompt and by_prompt[key]['soft_target'] != row['soft_target']:
            raise ValueError('Identical training prompts have conflicting probability targets')
        by_prompt[key] = row
    overlap, unique = [], []
    for row in validation:
        key = digest(row['input_ids'])
        if key in by_prompt:
            if row['soft_target'] != by_prompt[key]['soft_target']:
                raise ValueError('Identical validation/training prompts have conflicting targets')
            overlap.append(row)
        else:
            unique.append(row)
    return overlap, unique


def audit(root):
    train = read_rows(root/'data/train.jsonl')
    validation = read_rows(root/'data/validation.jsonl')
    receipt = json.loads((root/'run/run.json').read_text())
    step = receipt['best_step']
    before = matched_predictions(validation, root/'run/baseline-consequence.jsonl')
    after = matched_predictions(validation, root/'run'/f'consequence-step-{step}.jsonl') if step else before
    predictions = {r['id']:(a,b) for r,a,b in zip(validation,before,after)}
    overlap, unique = partition(train,validation)
    result = dict(role='post_hoc_diagnostic_only', selected_step_unchanged=step,
                  training_rows=len(train), unique_training_prompts=len({digest(r['input_ids']) for r in train}),
                  validation_rows=len(validation), exact_overlap_rows=len(overlap),
                  exact_overlap_fraction=len(overlap)/len(validation), no_exact_overlap_rows=len(unique),
                  explanation='History groups include fee profiles that are not fully rendered in consequence questions. '
                              'Different groups can therefore yield identical input tokens. Filtering after training '
                              'does not undo their influence on checkpoint selection or establish independent mechanism transfer.',
                  overlap_validation_ids=[r['id'] for r in overlap])
    for name, rows in [('exact_overlap',overlap),('no_exact_overlap',unique)]:
        result[name] = dict(n=len(rows), groups=len({r['group_id'] for r in rows}))
        if rows:
            result[name].update(baseline=summarize(rows,[predictions[r['id']][0] for r in rows],True),
                                selected=summarize(rows,[predictions[r['id']][1] for r in rows],True))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    result=audit(args.root);write_json(args.output,result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('exact_overlap','no_exact_overlap','overlap_validation_ids')},indent=2))
