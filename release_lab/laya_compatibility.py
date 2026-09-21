"""Inspect Laya's native choice input limits without loading model weights.

The token budgets and framing follow the public Laya interface at
NandhaKishorM/laya@573e5b62696ba441230cd6be71d593331b5d23af.
This independent accounting implementation must match the pinned upstream
formatter before a compatibility report is accepted. It is not an inference
runtime and makes no claim about a model's accuracy or calibration.
"""
from scale_lab.common import LABELS, digest, validate_input


def request(item):
    """Keep private source IDs and ground truth outside the model request."""
    validate_input(item)
    return dict(
        state=item['state'],
        questions={'decision': dict(type='choice', instructions=item['question'],
            criteria={LABELS[i]: option['description'] for i, option in enumerate(item['options'])})})


def inspect(item, tokenizer, config):
    payload = request(item)
    question = payload['questions']['decision']
    max_length = config.get('max_len', 512)
    head_length = config.get('head_max_len', 192)
    if any(type(n) is not int or n < 1 for n in (max_length, head_length)):
        raise ValueError('Invalid native token budgets')
    replacements = 0

    def clean(text):
        nonlocal replacements
        replacements += text.count(tokenizer.mask_token)
        return text.replace(tokenizer.mask_token, ' ')

    def tokens(text):
        return tokenizer(text, add_special_tokens=False)['input_ids']

    instruction = tokens('choice question: ' + clean(question['instructions']))
    descriptions = [tokens(' ' + clean(label + ': ' + text))
                    for label, text in question['criteria'].items()]
    state = tokens(clean(payload['state']))
    # Native formatting first caps each description, then divides a shared
    # head budget. A longer total context alone does not remove these limits.
    option_sizes = [min(len(ids), 48) + 1 for ids in descriptions]
    available = head_length - sum(option_sizes)
    if available < 16:
        per_option = max(4, (head_length - 16) // len(descriptions))
        option_sizes = [min(n, per_option) for n in option_sizes]
        available = head_length - sum(option_sizes)
    kept_instruction = instruction[:max(8, available)]
    sequence = [tokenizer.cls_token_id, *kept_instruction, tokenizer.sep_token_id]
    markers = []
    for description, size in zip(descriptions, option_sizes, strict=True):
        markers.append(len(sequence))
        sequence.extend([tokenizer.mask_token_id, *description[:size - 1]])
    sequence.append(tokenizer.sep_token_id)
    room = max(0, max_length - len(sequence) - 1)
    kept_state = state[:room]
    sequence.extend([*kept_state, tokenizer.sep_token_id])
    dropped = dict(
        instruction=len(instruction) - len(kept_instruction),
        option_descriptions=sum(len(ids) - (size - 1)
                                for ids, size in zip(descriptions, option_sizes, strict=True)),
        state=len(state) - len(kept_state), final_sequence=max(0, len(sequence) - max_length))
    present_markers = [m for m in markers if m < max_length]
    return dict(
        request=payload, request_sha256=digest(payload),
        option_ids=[o['id'] for o in item['options']],
        input_ids=sequence[:max_length], marker_positions=present_markers,
        max_len=max_length, head_max_len=head_length,
        original_token_counts=dict(instruction=len(instruction),
            option_descriptions=sum(map(len, descriptions)), state=len(state)),
        dropped_tokens=dropped, mask_literal_replacements=replacements,
        missing_option_markers=len(markers) - len(present_markers),
        full_information=(not any(dropped.values()) and replacements == 0 and
                          len(markers) == len(present_markers)))


def require_native_parity(observed, native_sequence, native_markers):
    if observed['input_ids'] != native_sequence or observed['marker_positions'] != native_markers:
        raise ValueError('Pinned native formatter differs from compatibility accounting')


def require_full_information(observed):
    if not observed['full_information']:
        raise ValueError('Native formatting loses supplied decision information')


def probabilities(answer, item):
    """Align public choice probabilities, preserving upstream rounding exactly.

    Raw probability metrics should use separately qualified unrounded logits.
    This helper never silently renormalizes the public interface's rounded values.
    """
    import math
    validate_input(item)
    labels = list(LABELS[:len(item['options'])])
    values = answer['probabilities']
    if answer.get('type') != 'choice' or set(values) != set(labels) or answer.get('choice') not in labels:
        raise ValueError('Laya answer does not match the declared choice menu')
    p = [values[label] for label in labels]
    if any(type(x) not in (float, int) or not math.isfinite(x) or not 0 <= x <= 1 for x in p):
        raise ValueError('Invalid choice probability')
    if abs(sum(p) - 1) > len(p) * .00005 + 1e-7:
        raise ValueError('Probability mass exceeds four-decimal rounding tolerance')
    if values[answer['choice']] < max(p):
        raise ValueError('Selected answer disagrees with its probability distribution')
    return dict(choice=item['options'][labels.index(answer['choice'])]['id'],
                probabilities={option['id']: value for option, value in zip(item['options'], p, strict=True)},
                probability_mass=sum(p), precision='native_public_four_decimals')
