"""Reversible shared framing for an action menu with long rules/descriptions.

This moves text out of models' narrow option/instruction fields. It does not
summarize observations, shorten commands, or promise that the total context fits.
Use the identical returned input for every compared model, and audit token loss.
"""
import json

from scale_lab.common import LABELS, validate_input

VERSION = 'shared-action-frame-v1'
QUESTION = 'Follow decision_question using observations. Choose the corresponding action.'


def visible_input(item):
    validate_input(item)
    return dict(state=item['state'], question=item['question'],
                options=[dict(id=o['id'], description=o['description']) for o in item['options']])


def pack(item):
    """Keep every visible string verbatim; tool IDs remain outside model text."""
    original = visible_input(item)
    content = dict(observations=original['state'], decision_question=original['question'],
                   actions={LABELS[i]: o['description'] for i, o in enumerate(original['options'])})
    result = dict(state=json.dumps(content, ensure_ascii=False, separators=(',', ':')),
                  question=QUESTION, options=[dict(id=o['id'], description='Action '+LABELS[i])
                    for i, o in enumerate(original['options'])])
    if unpack(result) != original:
        raise ValueError('Action framing did not preserve the public decision')
    return result


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate key in action frame')
        result[key] = value
    return result


def unpack(item):
    """Recover the complete public boundary, including original dispatch IDs."""
    validate_input(item)
    if item['question'] != QUESTION:
        raise ValueError('Unknown action framing instruction')
    content = json.loads(item['state'], object_pairs_hook=_unique_object)
    if not isinstance(content, dict) or set(content) != {'observations', 'decision_question', 'actions'}:
        raise ValueError('Unknown action framing fields')
    labels = list(LABELS[:len(item['options'])])
    if not isinstance(content['actions'], dict) or list(content['actions']) != labels:
        raise ValueError('Action labels or order differ')
    if [o['description'] for o in item['options']] != ['Action '+label for label in labels]:
        raise ValueError('Action reference does not match its label')
    result = dict(state=content['observations'], question=content['decision_question'],
                  options=[dict(id=o['id'], description=content['actions'][label])
                    for label, o in zip(labels, item['options'], strict=True)])
    validate_input(result)
    return result
