"""Local eight-bit package for the existing released decision model.

This custom package scores supplied options; it is not a text-generation export.
"""
import argparse
import importlib.metadata
import json
from pathlib import Path
import shutil

from scale_lab.common import MODELS,file_hash,write_json
from release_lab.mlx_scorer import PARENT,adapter_pairs,load


def export(base,adapter,reference,output,license_path):
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from mlx_lm.utils import quantize_model
    if output.exists():raise ValueError('Preserve earlier packages')
    result=json.loads((reference/'report.json').read_text())
    plan=json.loads((reference/'probe-plan-private.json').read_text())
    if result['status']!='smoke_passed':raise ValueError('Eight-bit source has not passed')
    for path,sha in plan['sources'].items():
        if file_hash(path)!=sha:raise ValueError('Prior serving source changed')
    identity=result['summary']['identity'];labels=identity['native_label_ids']
    model,_=load(base,adapter,labels)
    config=json.loads((base/'config.json').read_text())
    model,_=quantize_model(model,config,64,8,quant_predicate=lambda path,module:path!='language_model.lm_head')
    model.eval();mx.eval(model.parameters())
    pairs=adapter_pairs(mx.load(str(adapter/'adapter_model.safetensors')))
    weights={};removed=0
    for key,array in tree_flatten(model.parameters()):
        matched=False
        for name in pairs:
            if key in (name+'.a',name+'.b'):
                removed+=1;matched=True;break
            if key.startswith(name+'.linear.'):
                key=name+key[len(name+'.linear'):];break
        if matched:continue
        if key in weights:raise ValueError('Duplicate canonical base tensor')
        weights[key]=array
    if removed!=496 or weights['language_model.lm_head.weight'].shape!=(36,4096):
        raise ValueError('Incomplete adapter separation or output rows')
    output.mkdir(parents=True,exist_ok=False)
    shard={};size=0;index={};number=0
    def save():
        nonlocal shard,size,number
        if not shard:return
        number+=1;name=f'model-{number:05d}.safetensors'
        mx.save_safetensors(str(output/name),shard)
        index.update({key:name for key in shard});shard={};size=0
    for key in sorted(weights):
        array=weights[key]
        if array.nbytes>1_500_000_000:raise ValueError('Tensor exceeds portable shard cap')
        if size+array.nbytes>1_500_000_000:save()
        shard[key]=array;size+=array.nbytes
    save()
    write_json(output/'weights.index.json',index)
    write_json(output/'config.json',config)
    for name in ('adapter_config.json','adapter_model.safetensors'):
        shutil.copyfile(adapter/name,output/name)
    for name in ('tokenizer.json','tokenizer_config.json','chat_template.jinja','merges.txt','vocab.json','added_tokens.json','special_tokens_map.json'):
        if (base/name).is_file():shutil.copyfile(base/name,output/name)
    if 'Apache License' not in license_path.read_text():raise ValueError('Pinned foundation license missing')
    shutil.copyfile(license_path,output/'LICENSE-QWEN')
    (output/'requirements.txt').write_text('mlx==0.32.2\nmlx-lm==0.31.3\ntransformers==5.17.0\n')
    (output/'README.md').write_text('''# First Instinct — experimental Mac package

This is the existing released Qwen3.5-9B decision adapter at supervised step 2742,
with an eight-bit text base, unchanged float32 internal adapters, and the exact
36 native output rows in float32. No new training happened during conversion.
Vision weights and the unused vocabulary output rows are omitted.

Load this custom package with `release_lab.mlx_package.load_package` in the First
Instinct repository. It scores caller-supplied answer options; it is not a normal
text-generation checkpoint. Keep the supplied tokenizer and label mapping.

Status at export: full regression and fresh-process reload are pending. Do not
substitute it into a public service based on this export alone. The prototype was
measured on an M5 Max with 128 GB, not a 16-GB Mac mini.

The foundation is Qwen/Qwen3.5-9B, revision
c202236235762e1c871ad0ccb60c8ee5ba337b9a. Its Apache 2.0 license is included.
Training and experimental evidence: https://github.com/catoenm/first-instinct
''')
    manifest=dict(version='first-instinct-mlx-eight-bit-v1',model=MODELS['qwen35-9b'],
        adapter_sha256=PARENT,native_label_ids=labels,base_bits=8,group_size=64,max_tokens=4096,
        native_output_rows=36,adapter_tensors=496,adapter_dtype='float32',native_head_dtype='float32',
        prior_report_sha256=file_hash(reference/'report.json'),new_training_presentations=0,optimizer_steps=0,
        public_serving_qualified=False,versions={n:importlib.metadata.version(n) for n in ('mlx','mlx-lm','transformers')},
        exporter_sha256=file_hash(__file__),files={p.name:file_hash(p) for p in sorted(output.iterdir()) if p.is_file()})
    manifest['artifact_bytes']=sum(p.stat().st_size for p in output.iterdir() if p.is_file())
    write_json(output/'package.json',manifest)
    return manifest


def load_package(path):
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_unflatten
    from mlx_lm.models.qwen3_5 import Model,ModelArgs
    path=Path(path);manifest=json.loads((path/'package.json').read_text())
    if (manifest['version']!='first-instinct-mlx-eight-bit-v1' or manifest['model']!=MODELS['qwen35-9b'] or
            manifest['adapter_sha256']!=PARENT or manifest['base_bits']!=8 or manifest['group_size']!=64 or
            manifest['native_output_rows']!=36 or len(set(manifest['native_label_ids']))!=36):
        raise ValueError('Unsupported package identity')
    for name,sha in manifest['files'].items():
        if Path(name).name!=name or file_hash(path/name)!=sha:
            raise ValueError('Package file integrity differs: '+name)
    if file_hash(path/'adapter_model.safetensors')!=PARENT:raise ValueError('Released adapter changed')
    index=json.loads((path/'weights.index.json').read_text());weights={}
    for name in sorted(set(index.values())):
        if name not in manifest['files']:raise ValueError('Unlisted weight shard')
        part=mx.load(str(path/name))
        if any(key in weights or index.get(key)!=name for key in part):raise ValueError('Weight index differs')
        weights.update(part)
    if set(weights)!=set(index):raise ValueError('Missing base weights')
    config=json.loads((path/'config.json').read_text())
    if config.get('model_file') or config.get('model_type')!='qwen3_5' or config.get('tie_word_embeddings') is not False:
        raise ValueError('Unexpected model configuration')
    model=Model(ModelArgs.from_dict(config))
    model.language_model.lm_head=nn.Linear(4096,36,bias=False)
    nn.quantize(model,group_size=64,bits=8,mode='affine',
                class_predicate=lambda name,module:name+'.scales' in weights)
    # Weights are already in the MLX layout; do not sanitize or shift norms again.
    model.load_weights(list(weights.items()),strict=True)
    tensors=mx.load(str(path/'adapter_model.safetensors'));pairs=adapter_pairs(tensors)
    if len(tensors)!=496 or sum(v.size for v in tensors.values())!=43278336:
        raise ValueError('Incomplete trained tensors')
    class Adapter(nn.Module):
        def __init__(self,base,a,b):
            super().__init__();self.base=base;self.a=a;self.b=b
        def __call__(self,x):
            y=self.base(x);delta=(x.astype(self.a.dtype)@self.a.T)@self.b.T
            return (y.astype(delta.dtype)+2*delta).astype(y.dtype)
    modules=dict(model.named_modules());updates=[]
    for name,pair in pairs.items():
        layer=modules[name];a,b=tensors[pair['A']],tensors[pair['B']]
        if not isinstance(layer,(nn.Linear,nn.QuantizedLinear)):raise ValueError('Missing language projection')
        out,inp=layer.weight.shape
        if isinstance(layer,nn.QuantizedLinear):inp=inp*32//layer.bits
        if a.shape!=(16,inp) or b.shape!=(out,16) or a.dtype!=mx.float32 or b.dtype!=mx.float32:
            raise ValueError('Adapter shape or precision differs')
        updates.append((name,Adapter(layer,a,b)))
    model.update_modules(tree_unflatten(updates));model.eval();mx.eval(model.parameters());mx.clear_cache()
    return model,manifest


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('base','adapter','reference','output','license-path'):p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();m=export(**vars(args));print(json.dumps({k:v for k,v in m.items() if k!='files'},indent=2))
