"""Independent source reconstruction and nonredundant tool-history admission."""
import argparse
from collections import Counter, defaultdict
import copy
import json
from pathlib import Path
import time

from . import toucan as t
from scale_lab.common import digest, encode, file_hash, messages, write_json

QUESTION = "Given the user's request and the tool results observed so far, which available tool should be called next?"
QWEN_PREFIX = '# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n'
QWEN_SUFFIX = '\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call>'
KIMI_PREFIX = '<|im_system|>tool_declare<|im_middle|>'
KIMI_SUFFIX = '<|im_end|>'


def definitions(raw):
    tools = t.parse(raw['available_tools'])
    return [x.get('function', x) for x in tools]


def system_review(systems, raw):
    expected = sorted(digest(x) for x in definitions(raw))
    kinds = []
    for value in systems:
        if value.startswith(QWEN_PREFIX) and value.endswith(QWEN_SUFFIX):
            content = value[len(QWEN_PREFIX):-len(QWEN_SUFFIX)]
            declared = [t.parse(line) for line in content.splitlines() if line.strip()]
            kind = 'qwen_tool_declarations_only'
        elif value.startswith(KIMI_PREFIX) and value.endswith(KIMI_SUFFIX):
            declared = t.parse(value[len(KIMI_PREFIX):-len(KIMI_SUFFIX)])
            kind = 'kimi_tool_declarations_only'
        else:
            raise t.Exclude('unrecognized_source_system_content')
        if not isinstance(declared, list) or any(not isinstance(x, dict) for x in declared):
            raise t.Exclude('invalid_source_system_declarations')
        if sorted(digest(x.get('function', x)) for x in declared) != expected:
            raise t.Exclude('source_system_menu_mismatch')
        kinds.append(kind)
    return kinds


def source_events(raw):
    """Independent serial parser: return only closed source call/response pairs."""
    records = t.parse(raw['messages'])
    tools = {x['name']: x for x in definitions(raw)}
    _, remote = t.remote_schemas(raw)
    attributed = t.offered_tools(raw, remote)
    validators = {x['name']: t.checked_validator(json.dumps(x['parameters'], sort_keys=True)) for x in attributed}
    systems, user, pending, history, events, identities = [], None, None, [], [], set()
    for position, message in enumerate(records):
        role = message.get('role')
        if role == 'system':
            if user is not None or not isinstance(message.get('content'), str):
                raise t.Exclude('invalid_system_position')
            systems.append(message['content'])
        elif role == 'user':
            if user is not None or not isinstance(message.get('content'), str):
                raise t.Exclude('unsupported_user_history')
            user = message['content']
            if t.request_key(user) != t.request_key(raw['question']):
                raise t.Exclude('source_question_mismatch')
        elif role == 'assistant':
            legacy, modern = message.get('function_call'), message.get('tool_calls')
            if legacy and modern:
                raise t.Exclude('ambiguous_calls')
            if not legacy and not modern:
                continue
            if pending is not None or user is None or (modern and len(modern) != 1):
                raise t.Exclude('nonserial_calls')
            call = legacy if legacy else modern[0]['function']
            name = call.get('name')
            if name not in tools:
                raise t.Exclude('undeclared_call')
            arguments = t.parse(call.get('arguments'))
            if not isinstance(arguments, dict) or not validators[name].is_valid(arguments):
                raise t.Exclude('invalid_arguments')
            identity = modern[0].get('id') if modern else None
            if modern:
                if not isinstance(identity, str) or not identity or identity in identities:
                    raise t.Exclude('invalid_call_identity')
                identities.add(identity)
            pending = dict(position=position, name=name, arguments=arguments, identity=identity,
                           modern=bool(modern), history=copy.deepcopy(history))
        elif role in ('function', 'tool'):
            if pending is None:
                raise t.Exclude('orphan_response')
            if pending['modern']:
                matches = role == 'tool' and message.get('tool_call_id') == pending['identity']
            else:
                matches = role == 'function' and message.get('name') == pending['name']
            if not matches or not isinstance(message.get('content'), (str, dict, list)):
                raise t.Exclude('invalid_response_binding')
            if pending['history']:
                events.append(dict(message_index=pending['position'], target=pending['name'],
                                   history=pending['history'], response_sha256=digest(message)))
            history.append(dict(tool=pending['name'], arguments=pending['arguments'], response=message['content']))
            pending = None
        else:
            raise t.Exclude('unknown_message_role')
    if pending is not None or user is None:
        raise t.Exclude('incomplete_trajectory')
    return user, systems, events


def render(raw, user, systems, event, *, original=False):
    state = dict(request=user, observed_tool_history=event['history'])
    if original:
        state['source_system_messages'] = systems
    packed = json.dumps(state, sort_keys=True, ensure_ascii=False)
    if len(packed.encode()) > 256*1024:
        raise t.Exclude('public_history_over_256k_bytes')
    options = [dict(id=d['name'], description=json.dumps(
        {k: d.get(k, '') for k in ('name', 'description', 'parameters')}, sort_keys=True, ensure_ascii=False)) for d in definitions(raw)]
    options.sort(key=lambda o: digest(dict(seed=t.SEED, request=t.request_key(packed), option=o)))
    return dict(state=packed, question=QUESTION, options=options)


def check_original(row, raw, witness, parent, user, systems, event, tokenizer):
    if witness != parent['witness'] or digest(raw) != witness['row_sha256']:
        raise ValueError('Source identity changed')
    item = render(raw, user, systems, event, original=True)
    ids = encode(tokenizer, item, 4096)
    expected = dict(input=item, target={'option_id':event['target']}, message_index=event['message_index'],
                    preceding_tool_calls=len(event['history']), prefix_sha256=digest(event['history']),
                    response_sha256=event['response_sha256'], id=digest(messages(item)), input_ids=ids,
                    option_ids=[o['id'] for o in item['options']], token_sha256=digest(ids), role='train_candidate',
                    group_id=parent['group_id'], request_key=parent['request_key'], servers=parent['servers'],
                    first_question_id=parent['id'], lineage=[witness],
                    target_kind='observed_next_tool_imitation', admitted=False)
    expected['target_indices'] = [expected['option_ids'].index(event['target'])]
    if row != expected:
        raise ValueError('Saved candidate differs from independent source reconstruction')


def audit_and_build(inventory, source, output):
    from transformers import AutoTokenizer
    started = time.monotonic()
    old_plan = json.loads((inventory/'plan.json').read_text())
    old_summary = json.loads((inventory/'summary.json').read_text())
    for key, name in [('code','release_lab/trajectory_inventory.py'),('protocol','docs/trajectory-inventory-v1-protocol.md')]:
        if file_hash(t.ROOT/name) != old_plan['hashes'][key]:
            raise ValueError('Frozen inventory source changed')
    manifest = json.loads((source/'manifest-private.json').read_text())
    admission = json.loads((source/'admission.json').read_text())
    if admission['status'] != 'qualified_supervised_behavior_only' or admission['manifest_sha256'] != file_hash(source/'manifest-private.json'):
        raise ValueError('Original source admission changed')
    original_freeze=json.loads(Path(manifest['freeze_path']).read_text())
    if file_hash(manifest['freeze_path'])!=manifest['freeze_sha256']:
        raise ValueError('Original source freeze changed')
    for name in ('release_lab/toucan.py','scale_lab/common.py'):
        if file_hash(t.ROOT/name)!=original_freeze['files'][name]:
            raise ValueError('Inherited parser/formatter source changed')
    for name,expected in original_freeze['tokenizer_files'].items():
        if file_hash(name)!=expected: raise ValueError('Pinned tokenizer changed')
    if [spec for spec,_ in t.source_paths()] != old_plan['sources']:
        raise ValueError('Source shards changed')
    for name, expected in manifest['outputs'].items():
        if file_hash(source/name) != expected:
            raise ValueError('Source pack changed: '+name)
    if file_hash(inventory/'candidates-private.jsonl') != old_summary['candidate_sha256']:
        raise ValueError('Inventory changed')
    owners = json.loads((source/'ownership-private.json').read_text())
    parents, evaluation_keys, prior_tokens = {}, defaultdict(set), set()
    for role in t.ROLES:
        for line in (source/f'{role}-private.jsonl').open():
            row = json.loads(line)
            prior_tokens.add(row['token_sha256'])
            if role == 'train':
                for w in row['lineage']:
                    parents[(w['file'],w['row'])] = dict(id=row['id'],group_id=row['group_id'],
                        request_key=row['request_key'],servers=row['servers'],witness=w)
            else:
                for key in ('servers','schema_keys'):
                    evaluation_keys[key].update(row[key])
                evaluation_keys['requests'].add(row['request_key'])
    original_index = defaultdict(dict)
    with (inventory/'candidates-private.jsonl').open('rb') as stream:
        while True:
            offset = stream.tell(); line = stream.readline()
            if not line: break
            row = json.loads(line); w = row['lineage'][0]
            key = (w['file'],w['row']); position = row['message_index']
            if position in original_index[key]: raise ValueError('Duplicate inventory source position')
            original_index[key][position] = (offset,len(line))
    if sum(map(len,original_index.values())) != old_summary['candidate_unique_token_inputs']:
        raise ValueError('Inventory coverage changed')
    old_requests, old_schemas, old_tokens, _ = t.legacy_index()
    prior_tokens.update(old_tokens)
    output.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(manifest['model']['id'],revision=manifest['model']['revision'],local_files_only=True,trust_remote_code=False)
    plan = dict(version='trajectory-admission-v1',original_inventory_sha256=file_hash(inventory/'plan.json'),
                original_candidate_sha256=old_summary['candidate_sha256'],source_admission_sha256=file_hash(source/'admission.json'),
                source_manifest_sha256=file_hash(source/'manifest-private.json'),sources=old_plan['sources'],model=manifest['model'],
                code_sha256=file_hash(Path(__file__)),protocol_sha256=file_hash(t.ROOT/'docs/trajectory-admission-v1-protocol.md'),
                tests_sha256=file_hash(t.ROOT/'tests/test_trajectory_admission.py'),
                max_tokens=4096,role='train',label_contract='observed_next_tool_imitation')
    write_json(output/'plan.json',plan)
    counts=Counter(); exclusions=Counter(); wrappers=Counter(); selected={}; targets=defaultdict(set)
    source_requests=set(); source_servers=set(); origins={}; inspected=set(); audited=set()
    spool=output/'spool-private.jsonl'
    with (inventory/'candidates-private.jsonl').open('rb') as original, spool.open('wb') as sink, (output/'exclusions-private.jsonl').open('w') as rejected:
        for raw,witness in t.raw_rows():
            key=(witness['file'],witness['row'])
            if key not in parents: continue
            parent=parents[key]
            if witness != parent['witness'] or key in inspected: raise ValueError('Changed/duplicate source witness')
            inspected.add(key)
            record=t.inventory(raw,witness)
            _,remote=t.remote_schemas(raw)
            schema_keys={x['key'] for x in record['schemas']}
            if (t.row_role(record,owners)!='train' or record['request_key']!=parent['request_key'] or
                set(record['servers'])&evaluation_keys['servers'] or schema_keys&evaluation_keys['schema_keys'] or
                record['request_key'] in evaluation_keys['requests'] or record['request_key'] in old_requests or schema_keys&old_schemas):
                raise ValueError('Training ownership/legacy exclusion breached')
            try:
                user,systems,events=source_events(raw)
            except (t.Exclude,ValueError,TypeError,KeyError,AttributeError) as exc:
                if key in original_index: raise ValueError('Auditor rejected an inventory source') from exc
                exclusions['invalid_source_trajectory']+=1
                continue
            for event in events:
                if event['message_index'] in original_index.get(key,{}):
                    offset,size=original_index[key][event['message_index']]
                    original.seek(offset);saved=json.loads(original.read(size))
                    check_original(saved,raw,witness,parent,user,systems,event,tokenizer)
                    audited.add((key,event['message_index'])); counts['original_candidates_verified']+=1
            try: kinds=system_review(systems,raw)
            except (t.Exclude,ValueError,TypeError,KeyError,AttributeError) as exc:
                exclusions['nonredundant_or_unrecognized_system']+=1
                rejected.write(json.dumps(dict(witness=witness,reason=str(exc)))+'\n');continue
            wrappers.update(kinds)
            for event in events:
                try:
                    item=render(raw,user,systems,event)
                    ids=encode(tokenizer,item,4096)
                except ValueError:
                    exclusions['complete_input_over_limit']+=1;continue
                token_key=digest(ids)
                if token_key in prior_tokens: raise ValueError('Prior token collision')
                names=[o['id'] for o in item['options']]; target=names.index(event['target'])
                targets[token_key].add(target)
                row=dict(id=digest(messages(item)),input=item,input_ids=ids,token_sha256=token_key,
                         option_ids=names,target_indices=[target],target_contract='acceptable_set',role='train',
                         target_kind='observed_next_tool_imitation',group_id=parent['group_id'],request_key=parent['request_key'],
                         servers=parent['servers'],schema_keys=sorted(schema_keys),family='toucan_history',task='toucan_next_tool',
                         lineage=[dict(source=witness,message_index=event['message_index'],response_sha256=event['response_sha256'])],
                         first_question_id=parent['id'],preceding_tool_calls=len(event['history']),
                         rendering='history_without_proven_redundant_tool_declarations_v1')
                counts['candidate_presentations']+=1
                if token_key in selected:
                    rejected.write(json.dumps(dict(lineage=row['lineage'],reason='duplicate_input_pending_conflict_filter',token_sha256=token_key))+'\n')
                    continue
                offset=sink.tell(); content=(json.dumps(row,sort_keys=True,ensure_ascii=False)+'\n').encode();sink.write(content)
                selected[token_key]=(offset,len(content));origins[token_key]=(key,event['message_index'])
            if len(inspected)%500==0:
                write_json(output/'progress.json',dict(inspected=len(inspected),**counts))
    expected={(key,position) for key,positions in original_index.items() for position in positions}
    if inspected!=set(parents) or audited!=expected: raise ValueError('Incomplete source/candidate coverage')
    rows_manifest={}; conflicts={k for k,v in targets.items() if len(v)>1}; lengths=[]; trajectories=set(); recovered=0
    with spool.open('rb') as stream,(output/'train-private.jsonl').open('wb') as final:
        for token_key,(offset,size) in sorted(selected.items()):
            if token_key in conflicts: continue
            stream.seek(offset);content=stream.read(size);row=json.loads(content);final.write(content)
            rows_manifest[row['id']]=dict(sha256=digest(row),role='train')
            source_requests.add(row['request_key']);source_servers.update(row['servers']);lengths.append(len(row['input_ids']))
            key,position=origins[token_key];trajectories.add(key)
            recovered+=(key,position) not in expected
    summary=dict(status='qualified_supervised_history_imitation',questions=len(lengths),input_tokens=sum(lengths),max_tokens=max(lengths,default=0),
        original_candidates_verified=counts['original_candidates_verified'],inspected_training_trajectories=len(inspected),
        source_trajectories=len(trajectories),distinct_normalized_requests=len(source_requests),source_server_groups=len(source_servers),
        recovered_beyond_original_inventory=recovered,candidate_presentations=counts['candidate_presentations'],
        duplicate_presentations=counts['candidate_presentations']-len(selected),conflicting_inputs_removed=len(conflicts),
        exclusions=dict(exclusions),system_wrappers=dict(wrappers),training_presentations=0,optimizer_steps=0,
        new_executed_branches=0,new_independent_tasks=0,elapsed_seconds=time.monotonic()-started,
        scope='Teacher next-tool imitation only; no optimal-action or outcome-reward guarantee; existing training ownership, no new task families.')
    write_json(output/'summary.json',summary)
    output_manifest=dict(version='trajectory-admission-v1',model=manifest['model'],plan_sha256=file_hash(output/'plan.json'),
        files={name:file_hash(output/name) for name in ('train-private.jsonl','summary.json','exclusions-private.jsonl')},rows=rows_manifest)
    write_json(output/'manifest-private.json',output_manifest)
    write_json(output/'admission.json',dict(status=summary['status'],manifest_sha256=file_hash(output/'manifest-private.json'),
        questions=len(lengths),training_presentations=0,plan_sha256=output_manifest['plan_sha256'],original_candidate_coverage_complete=True,
        role='train_only',target_kind='observed_next_tool_imitation'))
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory',type=Path,default=Path('output/trajectory-inventory-v1-retry'))
    parser.add_argument('--source',type=Path,default=Path('output/release-tool-data-v1'))
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(audit_and_build(args.inventory,args.source,args.output),indent=2))
