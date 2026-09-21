"""Experimental Mac scorer using the same internal adapters and native label rows.

Imports MLX only when used. This module is not wired into the public demo.
"""
import json
from pathlib import Path

from scale_lab.common import file_hash

PARENT='882323ebe8a7c33edf89b7dd2938a977b00dfd7cb38551e8d68db4ae36128e1a'


def adapter_pairs(keys):
    pairs={}
    for key in keys:
        prefix='base_model.model.model.language_model.'
        if not key.startswith(prefix):
            raise ValueError('Adapter is not a language projection')
        suffix=next((x for x in ('lora_A.weight','lora_B.weight') if key.endswith('.'+x)),None)
        if suffix is None:
            raise ValueError('Unsupported adapter tensor')
        stem=key[len(prefix):-(len(suffix)+1)]
        name='language_model.model.'+stem
        pairs.setdefault(name,{})[suffix[5]]=key
    if not pairs or any(set(pair)!={'A','B'} for pair in pairs.values()):
        raise ValueError('Unpaired adapter tensors')
    return pairs


def load(base, adapter, label_ids, *, bits=None, expected_adapter=PARENT):
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_unflatten
    from mlx_lm.utils import load_model,quantize_model
    base,adapter=Path(base),Path(adapter)
    if file_hash(adapter/'adapter_model.safetensors')!=expected_adapter:
        raise ValueError('Adapter checkpoint identity differs')
    config=json.loads((base/'config.json').read_text())
    if (config.get('model_type')!='qwen3_5' or config.get('model_file') or
            config.get('tie_word_embeddings') is not False or config['text_config'].get('tie_word_embeddings',False)):
        raise ValueError('Require the pinned dense, untied Qwen3.5 text model')
    if len(label_ids)!=36 or len(set(label_ids))!=36 or bits not in (None,4):
        raise ValueError('Unexpected labels or quantization recipe')
    spec=json.loads((adapter/'adapter_config.json').read_text())
    if (spec['r']!=16 or spec['lora_alpha']!=32 or spec['bias']!='none' or spec.get('use_dora') or
            spec.get('use_rslora') or spec.get('rank_pattern') or spec.get('alpha_pattern') or
            spec.get('modules_to_save') or spec.get('fan_in_fan_out')):
        raise ValueError('Unsupported adapter configuration')
    model,loaded_config=load_model(base,lazy=True,strict=True)
    head=model.language_model.lm_head
    if head.weight.shape!=(248320,4096):
        raise ValueError('Unexpected native projection shape')
    native=head.weight[mx.array(label_ids)].astype(mx.float32)
    mx.eval(native)
    # Replace the large unused vocabulary projection with its exact native rows.
    small=nn.Linear(4096,36,bias=False);small.weight=native
    model.language_model.lm_head=small
    if bits:
        model,loaded_config=quantize_model(model,loaded_config,64,bits,
            quant_predicate=lambda path,module:path!='language_model.lm_head')
    tensors=mx.load(str(adapter/'adapter_model.safetensors'))
    pairs=adapter_pairs(tensors)
    if len(pairs)!=248 or sum(x.size for x in tensors.values())!=43278336:
        raise ValueError('Incomplete trained adapter')
    modules=dict(model.named_modules());converted=[]
    class PeftLinear(nn.Module):
        def __init__(self,linear,a,b):
            super().__init__();self.linear=linear;self.a=a;self.b=b
        def __call__(self,x):
            y=self.linear(x)
            delta=(x.astype(self.a.dtype)@self.a.T)@self.b.T
            return (y.astype(delta.dtype)+2.0*delta).astype(y.dtype)
    for name,pair in pairs.items():
        linear=modules.get(name)
        if not isinstance(linear,(nn.Linear,nn.QuantizedLinear)):
            raise ValueError('Adapter projection not found: '+name)
        a,b=tensors[pair['A']],tensors[pair['B']]
        out,inp=linear.weight.shape
        if isinstance(linear,nn.QuantizedLinear):inp=inp*32//linear.bits
        if a.shape!=(16,inp) or b.shape!=(out,16) or a.dtype!=mx.float32 or b.dtype!=mx.float32:
            raise ValueError('Adapter tensor shape or precision differs: '+name)
        converted.append((name,PeftLinear(linear,a,b)))
    model.update_modules(tree_unflatten(converted))
    model.eval();mx.eval(model.parameters())
    return model,dict(adapter_sha256=expected_adapter,adapter_tensors=len(tensors),adapted_projections=len(pairs),
                      native_label_ids=label_ids,bits=bits,native_head_bytes=native.nbytes,
                      active_memory_bytes=mx.get_active_memory())


def probabilities(model,input_ids,option_count):
    import mlx.core as mx
    if not 1<=len(input_ids)<=4096 or not 2<=option_count<=36:
        raise ValueError('Unsupported context or option count')
    inputs=mx.array([input_ids])
    hidden=model.language_model.model(inputs)[:,-1,:].astype(mx.float32)
    scores=model.language_model.lm_head(hidden)[0,:option_count]
    p=mx.softmax(scores)
    mx.eval(p)
    return p.tolist()
