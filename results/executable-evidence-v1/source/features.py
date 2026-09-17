"""Frozen language features, with strict length checks and no label inputs."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from .data import VIEWS,file_sha,read_rows,render,sha,write_json


def encode(rows,output,device='mps',batch_size=16):
    from transformers import AutoModel,AutoTokenizer
    from decision_model import MODEL_ID,MODEL_REVISION,pool_tokens
    torch.set_num_threads(1)
    tokenizer=AutoTokenizer.from_pretrained(MODEL_ID,revision=MODEL_REVISION,local_files_only=True,trust_remote_code=False)
    encoder=AutoModel.from_pretrained(MODEL_ID,revision=MODEL_REVISION,local_files_only=True,
            trust_remote_code=False,use_safetensors=False,dtype=torch.float32).to(device).eval()
    encoder.requires_grad_(False)
    texts=[render(r,v) for r in rows for v in VIEWS]
    tokens=tokenizer(texts,padding=False,truncation=False)
    lengths=[len(x) for x in tokens['input_ids']]
    if max(lengths)>512:
        i=int(np.argmax(lengths));raise ValueError(f'No truncation: {rows[i//4]["id"]}/{VIEWS[i%4]} has {lengths[i]} tokens')
    started=time.perf_counter();vectors=[]
    with torch.inference_mode():
        for start in range(0,len(texts),batch_size):
            batch=tokenizer.pad({k:v[start:start+batch_size] for k,v in tokens.items()},padding=True,return_tensors='pt').to(device)
            vectors.append(pool_tokens(encoder,batch).cpu().numpy())
            if start%(batch_size*10)==0:print(f'Encoded {min(start+batch_size,len(texts))}/{len(texts)}',flush=True)
    x=np.concatenate(vectors).reshape(len(rows),len(VIEWS),-1)
    path=Path(output);np.savez_compressed(path,features=x,ids=np.array([r['id'] for r in rows]))
    write_json(path.with_suffix('.json'),{'model_id':MODEL_ID,'revision':MODEL_REVISION,'frozen':True,
        'parameters':sum(p.numel() for p in encoder.parameters()),'device':device,'dtype':'float32',
        'contexts':len(texts),'maximum_tokens':max(lengths),'median_tokens':float(np.median(lengths)),
        'text_sha256':sha('\n'.join(texts)),'features_sha256':file_sha(path),'seconds':time.perf_counter()-started,
        'torch':torch.__version__,'feature_function':'attention-mask mean pooling then layer normalization'})
    return x


def load(path,rows):
    path=Path(path);meta=json.loads(path.with_suffix('.json').read_text())
    if file_sha(path)!=meta['features_sha256']:raise ValueError('Feature file checksum mismatch')
    if sha('\n'.join(render(r,v) for r in rows for v in VIEWS))!=meta['text_sha256']:
        raise ValueError('Features do not match the current rendered evidence')
    z=np.load(path,allow_pickle=False)
    if list(z['ids'])!=[r['id'] for r in rows]:raise ValueError('Feature/candidate order mismatch')
    return z['features'].astype(np.float64)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--pool',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='mps',choices=['cpu','mps'])
    a=p.parse_args();encode(read_rows(a.pool/'cases.jsonl.gz'),a.output,a.device)


if __name__=='__main__':main()
