"""No model inference: replay completed arms, preserving failures explicitly."""
import hashlib,json,shutil
from pathlib import Path
from games_lab.replay_audit import audit
root=Path('/workspace/first-instinct');out=Path('/workspace/mixed-game-replay-audits');out.mkdir(exist_ok=True)
shutil.copyfile(root/'games_lab/replay_audit.py',out/'auditor.py')
shutil.copyfile(Path(__file__),out/'audit-runner.py')
statuses={}
for name in ('reward','hybrid'):
 target=out/(name+'.json');arm=root/'run'/('rl-'+name)
 try:
  if not (arm/'run.json').exists() or json.loads((arm/'run.json').read_text())['status']!='complete':
   result={'status':'arm_not_complete'}
  else:result=audit(root/'data',arm,target)
 except Exception as error:result={'status':'failed','error':{'type':type(error).__name__,'detail':str(error)}}
 target.write_text(json.dumps(result,indent=2)+'\n');statuses[name]=result['status']
names=['auditor.py','audit-runner.py','reward.json','hybrid.json']
manifest={'files':{},'statuses':statuses}
for name in names:
 with (out/name).open('rb') as f:manifest['files'][name]=hashlib.file_digest(f,'sha256').hexdigest()
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(manifest))
