#!/bin/sh
set -eu
python3 -c 'import csv,json; from pathlib import Path; rows=list(csv.DictReader(Path('"'"'/app/data/sales.csv'"'"').open())); Path('"'"'/app/data/report.json'"'"').write_text(json.dumps({'"'"'region'"'"':'"'"'east'"'"','"'"'paid_total'"'"':sum(int(r['"'"'amount'"'"']) for r in rows if r['"'"'region'"'"']=='"'"'east'"'"' and r['"'"'status'"'"']=='"'"'paid'"'"')}))'
