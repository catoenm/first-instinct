"""Losslessly share public evidence and plans; decode before fixed controllers."""
import copy
import json

from tool_lab.appworld_controller_diagnostic import COST_FIELDS
from tool_lab.shared_json import canonical, expand_part, shared_pack

FORMAT = ('A {"$ref": n} means exactly entry n in $catalog, recursively. '
          'Expand references before interpreting records, result paths or commands. '
          'Each expanded API command incurs the declared per-call cost, including repeated commands. '
          'The catalog contains public data shared across the offered plans, without outcomes.')
ENCODING_FIELDS = {'$catalog', 'sharing_encoding'}
STOP = 'Stop now without another call.'


def encode_history(rows):
    if not rows: return []
    bases, plans = {}, {}
    for row in rows:
        state = json.loads(row['input']['state'])
        base = {k: v for k, v in state.items() if k not in COST_FIELDS | {'proposed_calls'}}
        bases[canonical(base)] = base
        candidates = ([state['proposed_calls']] if 'proposed_calls' in state else
                      [[] if o['description'] == STOP else json.loads(o['description'])
                       for o in row['input']['options']])
        for script in candidates: plans[canonical(script)] = script
    if len(bases) != 1: raise ValueError('Paired rows have different public base evidence')
    ordered = sorted(plans)
    bundle = {'base': next(iter(bases.values())),
              **{f'plan_{i}': plans[key] for i, key in enumerate(ordered)}}
    catalog, packed = shared_pack(bundle)
    if not isinstance(packed['base'], dict) or '$ref' in packed['base']:
        raise ValueError('Unexpected shared base envelope')
    encoded_plans = {key: packed[f'plan_{i}'] for i, key in enumerate(ordered)}
    output = []
    for row in rows:
        original_state = json.loads(row['input']['state'])
        state = {**packed['base'], '$catalog': catalog, 'sharing_encoding': FORMAT,
                 **{k: original_state[k] for k in COST_FIELDS if k in original_state}}
        item = copy.deepcopy(row['input'])
        if 'proposed_calls' in original_state:
            state['proposed_calls'] = encoded_plans[canonical(original_state['proposed_calls'])]
        else:
            for option in item['options']:
                if option['description'] != STOP:
                    option['description'] = canonical(encoded_plans[canonical(json.loads(option['description']))])
        item['state'] = canonical(state)
        if canonical(decode_input(item)) != canonical(normalize_input(row['input'])):
            raise ValueError('A public row changed during sharing')
        output.append({**row, 'input': item})
    return output


def normalize_input(item):
    result = copy.deepcopy(item)
    state = json.loads(item['state'])
    result['state'] = canonical(state)
    if 'proposed_calls' not in state:
        for option in result['options']:
            if option['description'] != STOP:
                option['description'] = canonical(json.loads(option['description']))
    return result


def decode_input(item):
    result = copy.deepcopy(item)
    state = json.loads(result['state'])
    if state.get('sharing_encoding') != FORMAT: raise ValueError('Unknown sharing contract')
    catalog = state['$catalog']
    base = {k: v for k, v in state.items() if k not in ENCODING_FIELDS}
    restored = expand_part(catalog, base)
    result['state'] = canonical(restored)
    if 'proposed_calls' not in state:
        for option in result['options']:
            if option['description'] != STOP:
                option['description'] = canonical(expand_part(catalog, json.loads(option['description'])))
    return result


def decode_row(row):
    return {**row, 'input': decode_input(row['input'])}
