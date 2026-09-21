"""Readable, lossless sharing of repeated public JSON values.

Catalog entries are exact values, not summaries. This codec never receives
outcomes, row identifiers, task partitions or model predictions.
"""
from collections import Counter
import json

REF = '$ref'
CATALOG = '$catalog'
VALUE = '$value'
TAGS = {REF, CATALOG, VALUE}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)


def validate_source(value):
    if isinstance(value, dict):
        if any(type(k) is not str for k in value) or TAGS.intersection(value):
            raise ValueError('Reserved sharing tag or non-string key in source')
        for child in value.values(): validate_source(child)
    elif isinstance(value, list):
        for child in value: validate_source(child)
    elif type(value) not in (str, int, float, bool, type(None)):
        raise ValueError('Source is not JSON')
    canonical(value)  # Reject non-finite numbers; preserves int/float distinctions.


def pack(value):
    validate_source(value)
    counts, values = Counter(), {}

    def count(node):
        key = canonical(node)
        if isinstance(node, (dict, list, str)) and len(key) >= 96:
            counts[key] += 1
            values[key] = node
        if isinstance(node, dict):
            for child in node.values(): count(child)
        elif isinstance(node, list):
            for child in node: count(child)
    count(value)
    selected = {key for key, n in counts.items() if n >= 2 and (n-1)*len(key)-16*n > 32}

    def encode(keys):
        ordered = sorted(keys, key=lambda key: (len(key), key))
        indices = {key: i for i, key in enumerate(ordered)}
        def visit(node, root=False):
            key = canonical(node)
            if not root and key in indices: return {REF: indices[key]}
            if isinstance(node, dict): return {k: visit(v) for k, v in sorted(node.items())}
            if isinstance(node, list): return [visit(v) for v in node]
            return node
        return ordered, [visit(values[key], True) for key in ordered], visit(value)

    # Sharing a parent can make its children no longer worth sharing. Remove
    # unused/single-use entries and recompute until every retained entry saves
    # space in the actual graph, rather than only in the original tree.
    while True:
        ordered, catalog, root = encode(selected)
        uses = Counter()
        def references(node):
            if isinstance(node, dict):
                if set(node) == {REF}: uses[node[REF]] += 1
                else:
                    for child in node.values(): references(child)
            elif isinstance(node, list):
                for child in node: references(child)
        references(root)
        for entry in catalog: references(entry)
        remove = {key for i, key in enumerate(ordered)
                  if uses[i] < 2 or (uses[i]-1)*len(canonical(catalog[i]))
                  - uses[i]*len(canonical({REF: i})) - 1 <= 16}
        if not remove: break
        selected -= remove
    result = {CATALOG: catalog, VALUE: root}
    if canonical(unpack(result)) != canonical(value):
        raise ValueError('Sharing changed a JSON value or type')
    return result


def unpack(encoded):
    if not isinstance(encoded, dict) or set(encoded) != {CATALOG, VALUE} or not isinstance(encoded[CATALOG], list):
        raise ValueError('Malformed catalog envelope')
    catalog, active, cache = encoded[CATALOG], set(), {}
    def visit(node):
        if isinstance(node, dict):
            if REF in node:
                index = node[REF]
                if set(node) != {REF} or type(index) is not int or not 0 <= index < len(catalog):
                    raise ValueError('Malformed catalog reference')
                if index in active: raise ValueError('Cyclic catalog reference')
                if index not in cache:
                    active.add(index)
                    cache[index] = visit(catalog[index])
                    active.remove(index)
                return cache[index]
            if TAGS.intersection(node): raise ValueError('Unexpected nested catalog tag')
            return {k: visit(v) for k, v in node.items()}
        if isinstance(node, list): return [visit(v) for v in node]
        if type(node) not in (str, int, float, bool, type(None)):
            raise ValueError('Catalog contains non-JSON data')
        return node
    result = visit(encoded[VALUE])
    if len(cache) != len(catalog): raise ValueError('Unreachable catalog entry')
    validate_source(result)
    return result


def shared_pack(values):
    """Encode a bundle; all its parts use one outcome-independent catalog."""
    encoded = pack(values)
    return encoded[CATALOG], encoded[VALUE]


def expand_part(catalog, value):
    """Resolve a public question's subset of a shared bundle's catalog."""
    # The full bundle is checked by unpack; individual questions may use a
    # subset. Include entries as explicit roots solely for strict validation.
    result = unpack({CATALOG: catalog, VALUE: [value, [{REF: i} for i in range(len(catalog))]]})
    return result[0]
