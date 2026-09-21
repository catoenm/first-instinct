"""Lossless column encoding for repeated public API records in model context."""
import json

TAG = '$record_table'


def pack(value):
    if isinstance(value, dict):
        if TAG in value:
            raise ValueError('Reserved record codec tag in source')
        return {k: pack(v) for k, v in value.items()}
    if not isinstance(value, list):
        return value
    plain = [pack(v) for v in value]
    if len(value) < 2 or not all(isinstance(v, dict) for v in value):
        return plain
    columns = sorted(value[0])
    if not all(sorted(v) == columns for v in value):
        return plain
    table = {TAG: {'columns': columns, 'rows': [[v[k] for k in columns] for v in plain]}}
    if len(json.dumps(table, separators=(',', ':'))) < len(json.dumps(plain, separators=(',', ':'))):
        return table
    return plain


def unpack(value):
    if isinstance(value, dict):
        if TAG in value:
            if set(value) != {TAG} or set(value[TAG]) != {'columns', 'rows'}:
                raise ValueError('Invalid table encoding')
            columns, rows = value[TAG]['columns'], value[TAG]['rows']
            if len(set(columns)) != len(columns) or any(len(r) != len(columns) for r in rows):
                raise ValueError('Invalid record dimensions')
            return [dict(zip(columns, (unpack(v) for v in row))) for row in rows]
        return {k: unpack(v) for k, v in value.items()}
    if isinstance(value, list):
        return [unpack(v) for v in value]
    return value


def checked_pack(value):
    encoded = pack(value)
    if unpack(encoded) != value:
        raise ValueError('Public history compression changed a value')
    return encoded
