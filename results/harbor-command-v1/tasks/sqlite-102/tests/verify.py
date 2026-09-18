import hashlib, json, os, sqlite3, stat
from pathlib import Path

def unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON field')
        result[key] = value
    return result

def check(root, spec):
    root = Path(root)
    observed = {}
    if root.is_symlink() or not root.is_dir():
        return False
    for directory, dirs, names in os.walk(root, followlinks=False):
        for name in dirs:
            if (Path(directory)/name).is_symlink():
                return False
        for name in names:
            p = Path(directory)/name
            s = p.lstat()
            if not stat.S_ISREG(s.st_mode) or s.st_size > 2_000_000:
                return False
            observed[str(p.relative_to(root))] = p
    if set(observed) != set(spec['files']):
        return False
    for name, expected in spec['files'].items():
        p = observed[name]
        if expected['kind'] == 'bytes':
            if hashlib.sha256(p.read_bytes()).hexdigest() != expected['sha256']:
                return False
        elif expected['kind'] == 'json':
            actual = json.loads(p.read_text(), object_pairs_hook=unique)
            if json.dumps(actual, sort_keys=True, allow_nan=False) != json.dumps(expected['value'], sort_keys=True, allow_nan=False):
                return False
        elif expected['kind'] == 'sqlite':
            connection = sqlite3.connect(p.resolve().as_uri()+'?mode=ro', uri=True)
            try:
                if connection.execute('PRAGMA integrity_check').fetchone() != ('ok',):
                    return False
                schema = [list(row) for row in connection.execute("SELECT type,name,tbl_name,sql FROM sqlite_master ORDER BY type,name")]
                if schema != expected['schema']:
                    return False
                for table, rows in expected['tables'].items():
                    # Names originate in the trusted generated verifier specification.
                    got = [list(row) for row in connection.execute('SELECT * FROM "'+table+'" ORDER BY 1')]
                    if got != rows:
                        return False
            finally:
                connection.close()
        else:
            raise ValueError('Unknown trusted verifier kind')
    return True

if __name__ == '__main__':
    spec = json.loads(Path('/tests/expected.json').read_text())
    try:
        success = check('/app/data', spec)
        error = None
    except (ValueError, OSError, sqlite3.Error) as exc:
        success, error = False, type(exc).__name__
    Path('/logs/verifier').mkdir(parents=True, exist_ok=True)
    Path('/logs/verifier/reward.txt').write_text('1' if success else '0')
    Path('/logs/verifier/check.json').write_text(json.dumps({'success':success,'error':error}))
