#!/bin/sh
set -eu
python3 -c 'import json; from pathlib import Path; p=Path('"'"'/app/data/services.json'"'"'); d=json.loads(p.read_text()); d['"'"'services'"'"']['"'"'east'"'"']['"'"'workers'"'"']=6; p.write_text(json.dumps(d,indent=2)+'"'"'\n'"'"')'
