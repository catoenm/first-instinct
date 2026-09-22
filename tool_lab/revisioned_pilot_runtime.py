"""Combine existing isolated workers and the qualified replanning database actor."""
from copy import deepcopy
import json

from scale_lab.common import encode, digest
from tool_lab.report_contract import CONTRACT, SEPARATOR, STATE_KEYS
from tool_lab.live_pilot_train import LivePool
from tool_lab.decision_learning_v2 import collect_revisioned


class ContractTokenizer:
    """A declared input transformation, retained and checked in actor receipts."""
    def __init__(self, tokenizer): self.tokenizer = tokenizer
    def __getattr__(self, name): return getattr(self.tokenizer, name)
    def apply_chat_template(self, messages, **kwargs):
        value = deepcopy(messages)
        if len(value) != 2 or value[1]['role'] != 'user': raise ValueError('Unexpected public message shape')
        item = json.loads(value[1]['content'])
        try: state = json.loads(item['state'])
        except json.JSONDecodeError: state = None
        if isinstance(state, dict) and state.get('family') == 'report':
            if set(state) != STATE_KEYS: raise ValueError('Report public contract schema differs')
            item['state'] += SEPARATOR+CONTRACT
            value[1]['content'] = json.dumps(item, ensure_ascii=False, sort_keys=True)
        return self.tokenizer.apply_chat_template(value, **kwargs)


class RevisionedPool(LivePool):
    def __init__(self, args):
        super().__init__(args); self.database_resets = 0; self.database_turns = 0
    def collect(self, policy, tokenizer, cases, resets, check, sample, revisioned=None):
        wrapped = ContractTokenizer(tokenizer)
        records, traces = super().collect(policy, wrapped, cases, resets, check, sample)
        receipt_updates = {}
        for trace in traces:
            prior_hash = digest(trace)
            for actor in trace['actor_events']:
                row = actor['row'] if 'receipt' in trace else actor['encoded_input']
                if encode(wrapped, actor['input'], self.args.max_tokens) != row['input_ids']:
                    raise ValueError('Recorded actor tokens differ from declared public input contract')
            trace['model_input_contract'] = 'full-public-report-contract-v1; unchanged other mechanisms'
            trace['report_contract_sha256'] = digest(CONTRACT)
            if 'receipt' not in trace: receipt_updates[prior_hash] = digest(trace)
        for record in records:
            if record['receipt_sha256'] in receipt_updates:
                record['receipt_sha256'] = receipt_updates[record['receipt_sha256']]
        if revisioned:
            if not sample: raise ValueError('Database collection is for sampled training, not a greedy evaluation claim')
            if self.database_resets+len(revisioned) > 32: raise ValueError('Database reset cap exceeded')
            found, executed = collect_revisioned(policy, tokenizer, revisioned, self.args.max_tokens, check)
            records += found; traces += executed
            self.database_resets += len(executed); self.database_turns += len(found)
        return records, traces
    def counts(self):
        return dict(**super().counts(), database_resets=self.database_resets, database_turns=self.database_turns)
