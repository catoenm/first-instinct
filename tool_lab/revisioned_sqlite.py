"""Small competing-writer executor. No model, network or training dependency."""
from copy import deepcopy
import json
from pathlib import Path
import sqlite3
import tempfile

VERSION='revisioned-sqlite-v1'
GOALS={
    'increment_latest':'Add exactly one to the counter as it stood after the colleague opportunity. Preserve the colleague note and all unrelated rows.',
    'approved_revision':'The approved edit sets counter 11 and is authorized only for revision 0. If the colleague changed the revision, leave their entire row unchanged. Preserve unrelated rows.'}
DESCRIPTIONS={
    'read':'Read row 1 and replace the local cached counter, note and revision.',
    'cached_write':'Set row 1 counter to cached counter plus one and note to cached note. Increment its current revision; do not check the cached revision.',
    'checked_write':'Perform cached_write only if the current revision equals the cached revision; otherwise refuse without changing anything.',
    'increment':'Atomically add one to the current counter and revision of row 1, preserving its current note.',
    'missing_read':'Read nonexistent row 404; a missing row returns an error and leaves the cache unchanged.',
    'finish':'Stop without further commands.'}
CONTRACT=('Every write targets only row 1. A successful write ends the episode immediately. '
          'Reads and a refused conditional write leave it open; stop or four decisions also end it. '
          'Read failures and refused writes still cost the stated amount. An initial read cached '
          'counter 10, note original, revision 0. Afterwards, independently of the goal and costs, '
          'a colleague has equal probability of doing nothing or atomically adding five to the '
          'counter, setting note colleague and incrementing revision. No more colleague actions '
          'occur. Current reads are exact; repeating them adds no independent evidence. '
          'Goal completion earns 100, damage earns -100 and a valid unfinished state earns 0, '
          'minus all future command costs. All figures are abstract reward units.')


def costs(profile):
    if profile not in ('cheap','expensive_read','expensive_write'):raise ValueError('Unknown fees')
    return dict(read=120 if profile=='expensive_read' else 2,
                cached_write=120 if profile=='expensive_write' else 1,
                checked_write=120 if profile=='expensive_write' else 3,
                increment=120 if profile=='expensive_write' else 40,missing_read=1,finish=0)


def snapshot(path):
    """Read actual rows through a separate, read-only connection."""
    db=sqlite3.connect('file:'+str(path)+'?mode=ro',uri=True)
    try:
        return dict(schema=[list(r) for r in db.execute('SELECT name,sql FROM sqlite_master WHERE type="table" ORDER BY name')],
                    rows=[list(r) for r in db.execute('SELECT id,counter,note,revision FROM documents ORDER BY id')],
                    integrity=[r[0] for r in db.execute('PRAGMA integrity_check')])
    finally:db.close()


class World:
    def __init__(self,goal,intervened,profile):
        if goal not in GOALS or type(intervened) is not bool:raise ValueError('Unknown world/goal')
        self.goal,self.intervened,self.profile=goal,intervened,profile
        self.fees=costs(profile);self.folder=tempfile.TemporaryDirectory(prefix='first-instinct-revision-')
        self.path=Path(self.folder.name)/'world.sqlite';self.actor=sqlite3.connect(self.path)
        self.actor.executescript('CREATE TABLE documents(id INTEGER PRIMARY KEY,counter INTEGER NOT NULL,note TEXT NOT NULL,revision INTEGER NOT NULL);'
                                "INSERT INTO documents VALUES(1,10,'original',0),(99,80,'protected',4);")
        self.initial=snapshot(self.path)
        self.cache=list(self.actor.execute('SELECT counter,note,revision FROM documents WHERE id=1').fetchone())
        self.prefix=[dict(action='read',returncode=0,row=list(self.cache),historical=True)]
        self.colleague=sqlite3.connect(self.path)
        if intervened:
            with self.colleague:
                self.colleague.execute("UPDATE documents SET counter=counter+5,note='colleague',revision=revision+1 WHERE id=1")
        self.start=snapshot(self.path);self.history=[];self.events=[];self.spent=0;self.done=False

    def visible(self):
        return dict(goal=GOALS[self.goal],contract=CONTRACT,cache=list(self.cache),
                    history=deepcopy(self.prefix+self.history),costs=dict(self.fees),remaining=4-len(self.events),
                    options=[dict(id=k,description=v) for k,v in DESCRIPTIONS.items()])

    def step(self,action):
        if self.done or action not in DESCRIPTIONS:raise ValueError('Illegal or post-terminal action')
        before=snapshot(self.path);visible=self.visible();old_cache=list(self.cache)
        observed=dict(returncode=0);write=False
        if action in ('read','missing_read'):
            row=self.actor.execute('SELECT counter,note,revision FROM documents WHERE id=?',(1 if action=='read' else 404,)).fetchone()
            if row is None:observed=dict(returncode=66,error='missing row')
            else:self.cache=list(row);observed['row']=list(row)
        elif action in ('cached_write','checked_write','increment'):
            with self.actor:
                if action=='increment':
                    c=self.actor.execute('UPDATE documents SET counter=counter+1,revision=revision+1 WHERE id=1')
                else:
                    sql='UPDATE documents SET counter=?,note=?,revision=revision+1 WHERE id=1'
                    args=(self.cache[0]+1,self.cache[1])
                    if action=='checked_write':sql+=' AND revision=?';args+=self.cache[2],
                    c=self.actor.execute(sql,args)
            write=c.rowcount==1
            observed=dict(returncode=0 if write else 75,changed_rows=c.rowcount)
        self.history.append(dict(action=action,**observed))
        self.spent+=self.fees[action]
        self.done=write or action=='finish' or len(self.events)+1==4
        event=dict(input=visible,action=action,observation=observed,before=before,after=snapshot(self.path),
                   cache_before=old_cache,cache_after=list(self.cache),cost=self.fees[action],terminal=self.done)
        self.events.append(event);return event

    def close(self):
        self.actor.close();self.colleague.close();self.folder.cleanup()


def execute(goal,intervened,profile,actions):
    world=World(goal,intervened,profile)
    try:
        for action in actions:
            if world.done:break
            world.step(action)
        if not world.done:raise ValueError('Witness plan did not terminate')
        return dict(version=VERSION,goal=goal,intervened=intervened,profile=profile,
                    initial=world.initial,start=world.start,prefix=world.prefix,events=world.events,
                    final=snapshot(world.path),cost=world.spent)
    finally:world.close()
