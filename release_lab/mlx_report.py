"""Recompute complete Mac conversion metrics from saved predictions without MLX."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics

from release_lab.pilot_metrics import summarize
from scale_lab.common import file_hash, read_rows, write_json


def audit(output):
    plan = json.loads((output/'plan-private.json').read_text())
    report = json.loads((output/'report.json').read_text())
    if report['plan_sha256'] != file_hash(output/'plan-private.json'):
        raise ValueError('Qualification plan changed')
    for name, sha in plan['paths'].items():
        if file_hash(name) != sha:
            raise ValueError('Frozen conversion source changed: ' + name)
    data_paths = [Path(p) for p in plan['paths'] if Path(p).name == 'development.jsonl']
    if len(data_paths) != 1:
        raise ValueError('Expected one development source')
    rows = read_rows(data_paths[0])
    ids = {row['id'] for row in rows}
    if len(rows) != 6072 or len(ids) != 6072 or any(r['role'] != 'development' for r in rows):
        raise ValueError('Incorrect evaluation ownership or coverage')
    saved = read_rows(output/'development-predictions-private.jsonl')
    predictions = {row['id']: row for row in saved}
    if len(saved) != len(predictions) or set(predictions) != ids:
        raise ValueError('Missing, repeated or additional predictions')
    references = []
    for name in plan['paths']:
        if Path(name).name.startswith('0-') and Path(name).name.endswith('-predictions.jsonl'):
            references.extend(read_rows(name))
    baseline = {row['id']: row['probabilities'] for row in references}
    if len(references) != len(baseline) or set(baseline) != ids:
        raise ValueError('Reference predictions differ')
    suites = defaultdict(list)
    differences = []; agreed = 0; times = []
    for row in rows:
        saved_row = predictions[row['id']]
        if saved_row['suite'] != row['suite'] or saved_row['tokens'] != len(row['input_ids']):
            raise ValueError('Wrong source attribution or context length')
        elapsed = saved_row['seconds']
        if not math.isfinite(elapsed) or elapsed <= 0:
            raise ValueError('Invalid latency')
        p, q = saved_row['probabilities'], baseline[row['id']]
        if len(p) != len(q):
            raise ValueError('Output cardinality differs')
        differences.append(max(abs(a-b) for a,b in zip(p,q)))
        agreed += max(range(len(p)), key=p.__getitem__) == max(range(len(q)), key=q.__getitem__)
        times.append(elapsed); suites[row['suite']].append(row)
    measured = {s:summarize(rs,[predictions[r['id']]['probabilities'] for r in rs]) for s,rs in suites.items()}
    original = {s:summarize(rs,[baseline[r['id']] for r in rs]) for s,rs in suites.items()}
    if measured != report['metrics'] or original != report['reference_metrics']:
        raise ValueError('Reported metrics do not reproduce')
    scalars = dict(choice_agreement=agreed/len(rows), mean_max_delta=statistics.mean(differences),
                   worst_max_delta=max(differences), median_seconds=statistics.median(times))
    if any(report[k] != v for k,v in scalars.items()):
        raise ValueError('Reported conversion diagnostics do not reproduce')
    checks = {}
    for suite in ('general','tools'):
        a,b = measured[suite]['macro'],original[suite]['macro']
        checks[suite] = a['accuracy'] >= b['accuracy']-.005 and a['log_loss'] <= b['log_loss']+.02
    checks['product_slices'] = all(v['accuracy'] >= original['general']['slices'][s]['accuracy']-.01 for s,v in measured['general']['slices'].items())
    checks['application_decisions'] = measured['application_decisions']['macro']['accuracy'] >= original['application_decisions']['macro']['accuracy']
    checks['outcomes'] = all(v['brier'] <= original['outcomes']['by_group'][g]['brier']+.005 and v['log_loss'] <= original['outcomes']['by_group'][g]['log_loss']+.01 for g,v in measured['outcomes']['by_group'].items())
    checks['probability_drift'] = scalars['choice_agreement'] >= .99 and scalars['mean_max_delta'] <= .01 and scalars['worst_max_delta'] <= .20
    checks['inference_memory'] = report['peak_inference_bytes'] <= 10*1024**3
    expected_status = 'regression_passed' if all(checks.values()) else 'regression_failed'
    if checks != report['checks'] or report['status'] != expected_status:
        raise ValueError('Reported gates do not reproduce')
    return dict(status='saved_predictions_and_gates_verified', regression_status=expected_status,
                questions=len(rows), by_suite={s:len(rs) for s,rs in suites.items()},
                checks=checks, **scalars, package_sha256=report['package_sha256'],
                report_sha256=file_hash(output/'report.json'),
                predictions_sha256=file_hash(output/'development-predictions-private.jsonl'),
                note='Metrics and coverage independently recomputed. Memory and timing are instrumentation records, not independent hardware measurements.')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--qualification', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError('Preserve earlier audits')
    result = audit(args.qualification)
    write_json(args.output, result)
    print(json.dumps(result, indent=2))
