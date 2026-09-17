"""Show a saved program that passed visible checks but failed a private check."""
import argparse
import json
from pathlib import Path

from .data import canonical


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,default=Path('results/executable-evidence-v1'))
    p.add_argument('--case',help='Optional candidate id from analysis.json');a=p.parse_args()
    rows=json.loads((a.run/'analysis.json').read_text())['counterexamples']
    if a.case:rows=[r for r in rows if r['id']==a.case]
    if not rows:raise SystemExit('No matching counterexample.')
    row=min(rows,key=lambda r:(len(r['code']),r['id']))
    print('Saved execution example: '+row['id'])
    print('\nContract: '+row['contract']+'\n\n'+row['code'])
    for i,check in enumerate(row['visible_checks'],1):
        print(f'Visible check {i}: {canonical(check["input"])} -> {canonical(check["observed"])}; PASS')
    failure=row['first_private_failure']
    print('\nA private check exposes the mistake:')
    print('Input: '+canonical(failure['input']))
    print('Expected: '+canonical(failure['expected']))
    print('Observed: '+('raised '+failure['error'] if failure['error'] else canonical(failure['observed'])))
    print(f'\n{row["private_failure_count"]} of the 32 private checks failed.')
    print('These are saved receipts, not a fresh model prediction. Use evidence_lab.verify to execute them again.')


if __name__=='__main__':main()
