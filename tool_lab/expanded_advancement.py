"""Closed-study access check and the prospective joint advancement criterion."""
import json
import math

METHODS=('outcome','reward','hybrid')
SEEDS=(1507,1609)
REQUIRED={f'{method}-{seed}' for method in METHODS for seed in SEEDS}|{'original-test'}


def require_closed_recovery(root):
    """Read completion metadata only; call before opening any final predictions."""
    read=lambda p:json.loads(p.read_text())
    receipt=read(root/'cloud-collection.json')
    if receipt.get('pipeline_status')!='complete' or receipt.get('pod_deleted') is not True:
        raise ValueError('Final scores remain sealed: study/recovery not complete')
    if read(root/'launch.json').get('status')!='complete' or read(root/'run/pipeline.json').get('status')!='complete':
        raise ValueError('Final scores remain sealed: controller not complete')
    states={name:read(root/'run'/name/'run.json') for name in sorted(REQUIRED)}
    if any(r.get('status')!='complete' for r in states.values()):
        raise ValueError('Final scores remain sealed: an arm has not completed')
    for name,r in states.items():
        expected='baseline' if name=='original-test' else name.rsplit('-',1)[0]
        if r.get('arm')!=expected or (name!='original-test' and r.get('seed')!=int(name.rsplit('-',1)[1])):
            raise ValueError('Arm completion receipt has wrong identity')
    return states


def advancement(arms,criteria):
    missing=sorted(REQUIRED-arms.keys())
    if missing:return dict(status='pending',missing=missing,passed_methods=[])
    original=arms['original-test'];control=original['transfer'];methods={}
    def number(x):
        if isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x):raise ValueError('Nonfinite or invalid metric')
        return x
    def at_least(x,y):return number(x)>=number(y)-1e-12
    def retained(measured,baseline):
        accuracy=number(baseline['macro_accuracy'])-number(measured['macro_accuracy'])
        loss=number(measured['macro_log_loss'])-number(baseline['macro_log_loss'])
        return dict(accuracy_drop=accuracy,log_loss_increase=loss,
            passed=at_least(criteria['retention_and_general_transfer_accuracy_drop'],accuracy) and
                   at_least(criteria['retention_and_general_transfer_log_loss_increase'],loss))
    if criteria['both_seeds'] is not True:raise ValueError('Both seeds are required')
    for method in METHODS:
        seeds=[]
        for seed in SEEDS:
            arm=arms[f'{method}-{seed}'];selected=arm['transfer']
            reward=number(selected['reward'])-number(control['reward'])
            brier=number(control['forecast']['macro']['expected_brier'])-number(selected['forecast']['macro']['expected_brier'])
            retention=retained(selected['retention'],arm['baseline_retention'])
            transfer=retained(arm['general_transfer'],original['general_transfer'])
            seeds.append(dict(seed=seed,macro_calendar_return_gain=reward,macro_expected_brier_reduction=brier,
                general_retention=retention,general_transfer=transfer,
                passed=at_least(reward,criteria['macro_calendar_return_gain']) and
                       at_least(brier,criteria['macro_expected_brier_reduction']) and retention['passed'] and transfer['passed']))
        methods[method]=dict(passed=all(s['passed'] for s in seeds),seeds=seeds)
    return dict(status='complete',methods=methods,passed_methods=[m for m in METHODS if methods[m]['passed']],
        comparison_tolerance=1e-12,
        limits='Descriptive gate on one authored transfer mechanism, three task structures and two seeds. Passing does not establish broad generalization or identify Jev implementation.')
