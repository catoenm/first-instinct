"""Independent, real SQLite execution for auditing the native simulator."""

import json
from pathlib import Path
import sqlite3
import time

from .contract import ACTIONS, CODES, WORLDS, public


class AttemptBudget:
    """Append an attempt before constructing its database; failures still count."""
    def __init__(self, path, maximum=1424):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.maximum = maximum
        self.count = len(self.path.read_text().splitlines()) if self.path.exists() else 0

    def claim(self, identity):
        if self.count >= self.maximum: raise RuntimeError('Database attempt budget exhausted')
        with self.path.open('a') as handle:
            handle.write(json.dumps({'attempt':self.count,'identity':identity,'time_ns':time.time_ns()})+'\n')
            handle.flush()
        self.count += 1


class SqlEpisode:
    def __init__(self, world, profile, budget, identity):
        budget.claim(identity)
        self.profile = profile
        self.initial_a,self.initial_b,self.initial_account = WORLDS[world]
        self.added_a = self.added_b = 0
        self.known_a = self.known_b = self.known_account = -1
        self.steps = self.cost = self.code = self.done = self.outcome = 0
        self.statements = []
        self.db = sqlite3.connect(':memory:', isolation_level=None)
        self.db.set_trace_callback(self.statements.append)
        self.db.execute('PRAGMA foreign_keys=ON')
        self.db.executescript('''
CREATE TABLE accounts(id INTEGER PRIMARY KEY, name TEXT NOT NULL, balance INTEGER NOT NULL);
CREATE TABLE inventory(sku TEXT PRIMARY KEY, qty INTEGER NOT NULL CHECK(qty>=0));
CREATE TABLE requests(id INTEGER PRIMARY KEY, account INTEGER NOT NULL REFERENCES accounts(id));
CREATE TABLE allocations(request INTEGER NOT NULL REFERENCES requests(id), sku TEXT NOT NULL REFERENCES inventory(sku), qty INTEGER NOT NULL CHECK(qty=1), PRIMARY KEY(request,sku));
CREATE TABLE audit(id INTEGER PRIMARY KEY, message TEXT NOT NULL);
INSERT INTO accounts VALUES(99,'protected',700);
INSERT INTO inventory VALUES('C',7);
INSERT INTO requests VALUES(999,99);
INSERT INTO allocations VALUES(999,'C',1);
INSERT INTO audit VALUES(1,'preserve');
''')
        self.db.executemany('INSERT INTO inventory VALUES(?,?)', [('A',self.initial_a),('B',self.initial_b)])
        if self.initial_account: self.db.execute("INSERT INTO accounts VALUES(7,'target',23)")
        self.schema = self._schema()
        self.history = []

    def _schema(self):
        return self.db.execute('SELECT type,name,tbl_name,sql FROM sqlite_master ORDER BY type,name').fetchall()

    def state(self):
        stock = dict(self.db.execute('SELECT sku,qty FROM inventory'))
        account = int(self.db.execute('SELECT COUNT(*) FROM accounts WHERE id=7').fetchone()[0] > 0)
        header = int(self.db.execute('SELECT COUNT(*) FROM requests WHERE id=17').fetchone()[0] > 0)
        allocations = dict(self.db.execute('SELECT sku,qty FROM allocations WHERE request=17'))
        return dict(a=stock['A'],b=stock['B'],account=account,header=header,
                    alloc_a=allocations.get('A',0),alloc_b=allocations.get('B',0),
                    added_a=self.added_a,added_b=self.added_b,initial_a=self.initial_a,
                    initial_b=self.initial_b,initial_account=self.initial_account,
                    step=self.steps,horizon=self.profile['horizon'],done=self.done,
                    outcome=self.outcome,cost=self.cost,code=self.code,
                    known_a=self.known_a,known_b=self.known_b,known_account=self.known_account)

    def public(self): return public(self.state(),self.profile)

    def verify(self):
        """Recompute the goal and frame from database rows, including protected data."""
        s = self.state()
        expected_accounts = [(99,'protected',700)]
        if s['account']:
            expected_accounts.insert(0,(7,'target',23 if self.initial_account else 0))
        expected_requests = [(999,99)]
        if s['header']: expected_requests.insert(0,(17,7))
        expected_allocations = [(999,'C',1)]
        if s['alloc_a']: expected_allocations.append((17,'A',1))
        if s['alloc_b']: expected_allocations.append((17,'B',1))
        forbidden = (
            self._schema() != self.schema or
            self.db.execute('PRAGMA integrity_check').fetchall() != [('ok',)] or
            self.db.execute('PRAGMA foreign_key_check').fetchall() != [] or
            self.db.execute('SELECT * FROM accounts ORDER BY id').fetchall() != expected_accounts or
            self.db.execute('SELECT * FROM requests ORDER BY id').fetchall() != expected_requests or
            self.db.execute('SELECT * FROM allocations ORDER BY request,sku').fetchall() != sorted(expected_allocations) or
            self.db.execute('SELECT * FROM audit ORDER BY id').fetchall() != [(1,'preserve')] or
            self.db.execute('SELECT * FROM inventory ORDER BY sku').fetchall() != [('A',s['a']),('B',s['b']),('C',7)] or
            s['a']+s['alloc_a'] != self.initial_a+self.added_a or
            s['b']+s['alloc_b'] != self.initial_b+self.added_b or
            (self.initial_account and not s['account']) or
            (s['header'] and not s['account']) or
            ((s['alloc_a'] or s['alloc_b']) and not s['header'])
        )
        if forbidden: return 4
        if s['header'] and s['alloc_a'] == 1 and s['alloc_b'] == 1: return 1
        if s['header'] or s['alloc_a'] or s['alloc_b']: return 3
        return 2

    def _reserve(self, atomic):
        saved = (self.known_a,self.known_b,self.known_account)
        stage = 'header'
        if atomic: self.db.execute('BEGIN')
        try:
            self.db.execute('INSERT INTO requests VALUES(17,7)')
            self.known_account = 1
            stage = 'a'
            self.db.execute("UPDATE inventory SET qty=qty-1 WHERE sku='A'")
            if self.known_a >= 0: self.known_a -= 1
            self.db.execute("INSERT INTO allocations VALUES(17,'A',1)")
            stage = 'b'
            self.db.execute("UPDATE inventory SET qty=qty-1 WHERE sku='B'")
            if self.known_b >= 0: self.known_b -= 1
            self.db.execute("INSERT INTO allocations VALUES(17,'B',1)")
            if atomic: self.db.execute('COMMIT')
            self.code = CODES.index('reserved')
        except sqlite3.IntegrityError as exc:
            if atomic:
                self.db.execute('ROLLBACK')
                self.known_a,self.known_b,self.known_account = saved
            if stage == 'header':
                if exc.sqlite_errorname == 'SQLITE_CONSTRAINT_FOREIGNKEY':
                    self.known_account = 0; self.code = CODES.index('missing_account')
                elif exc.sqlite_errorname == 'SQLITE_CONSTRAINT_PRIMARYKEY':
                    self.code = CODES.index('duplicate')
                else: raise
            elif stage == 'a' and exc.sqlite_errorname == 'SQLITE_CONSTRAINT_CHECK':
                self.known_account = 1; self.known_a = 0; self.code = CODES.index('short_a')
            elif stage == 'b' and exc.sqlite_errorname == 'SQLITE_CONSTRAINT_CHECK':
                self.known_account = 1; self.known_b = 0; self.code = CODES.index('short_b')
            else: raise

    def step(self, action):
        if self.done: raise ValueError('Episode already ended')
        action_index = ACTIONS.index(action)
        before_statements = len(self.statements)
        self.steps += 1
        fee = self.profile['costs'][action_index]
        self.cost += fee
        if action == 'inspect_stock':
            rows = dict(self.db.execute("SELECT sku,qty FROM inventory WHERE sku IN ('A','B')"))
            self.known_a,self.known_b = rows['A'],rows['B'];self.code = 1
        elif action == 'inspect_account':
            self.known_account = int(self.db.execute('SELECT EXISTS(SELECT 1 FROM accounts WHERE id=7)').fetchone()[0]);self.code = 2
        elif action in ('atomic','sequential'):
            self._reserve(action == 'atomic')
        elif action in ('replenish_a','replenish_b'):
            sku = 'A' if action == 'replenish_a' else 'B'
            changed = self.db.execute('UPDATE inventory SET qty=qty+1 WHERE sku=? AND qty=0',(sku,)).rowcount
            self.code = 8 if changed else 12
            if changed:
                if sku == 'A': self.added_a += 1;self.known_a = 1
                else: self.added_b += 1;self.known_b = 1
        elif action == 'create_account':
            changed = self.db.execute("INSERT OR IGNORE INTO accounts VALUES(7,'target',0)").rowcount
            self.known_account = 1;self.code = 9 if changed else 12
        elif action == 'undo':
            self.db.execute('BEGIN')
            try:
                allocations = self.db.execute('SELECT sku,qty FROM allocations WHERE request=17').fetchall()
                for sku,qty in allocations:
                    self.db.execute('UPDATE inventory SET qty=qty+? WHERE sku=?',(qty,sku))
                    if sku == 'A' and self.known_a >= 0: self.known_a += qty
                    if sku == 'B' and self.known_b >= 0: self.known_b += qty
                self.db.execute('DELETE FROM allocations WHERE request=17')
                self.db.execute('DELETE FROM requests WHERE id=17')
                self.db.execute('COMMIT')
            except BaseException:
                self.db.execute('ROLLBACK');raise
            self.code = 10
        else:
            self.code = 11
        terminal = 0
        if action == 'finish' or self.steps >= self.profile['horizon']:
            self.done = 1;self.outcome = self.verify()
            terminal = 400 if self.outcome == 1 else -400 if self.outcome >= 3 else 0
        reward = (terminal-fee)/400.
        self.history.append({'action':action,'reward':reward,'public':self.public(),
                             'state':self.state(),'statements':self.statements[before_statements:]})
        return reward

    def close(self): self.db.close()
    def __enter__(self): return self
    def __exit__(self,*args): self.close()
