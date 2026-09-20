"""Fixed calendar command programs. Reads public request, never verifier targets."""
import datetime as dt
import json
from pathlib import Path
import sqlite3
import sys
from zoneinfo import ZoneInfo


def span(request,fold):
    with Path('zone.tzif').open('rb') as source:
        zone=ZoneInfo.from_file(source,key='America/New_York')
    local=dt.datetime.fromisoformat(request['local_start']).replace(tzinfo=zone,fold=fold)
    start=int(local.timestamp());end=start+60*request['elapsed_minutes']
    return start,end


def instant(value):return dt.datetime.fromtimestamp(value,dt.timezone.utc).isoformat()


def observation(row):
    return dict(id=row[0],calendar=row[1],start_utc=instant(row[2]),end_utc=instant(row[3]),title=row[4],note=row[5])


def run(action):
    request=json.loads(Path('request.json').read_text())
    if action=='cached':return dict(generation=0,events=[]),0
    conn=sqlite3.connect('calendar.sqlite',isolation_level=None)
    conn.execute('PRAGMA foreign_keys=ON')
    attempted=None;atomic=False
    try:
        if action=='inspect':
            return dict(generation=1,events=[observation(r) for r in conn.execute('SELECT * FROM events ORDER BY id')],
                conversions={name:dict(start_utc=instant(span(request,fold)[0]),end_utc=instant(span(request,fold)[1]))
                             for name,fold in [('first',0),('second',1)]}),0
        if action not in {'book_first','book_second','atomic_first','atomic_second','sequential_first','sequential_second','wrong_calendar','bad_calendar'}:
            raise ValueError('Outside fixed command catalog')
        fold=int(action.endswith('second'));start,end=span(request,fold)
        calendar=request['calendar']
        if action=='wrong_calendar':calendar='personal'
        if action=='bad_calendar':calendar='absent'
        identity=-999 if action=='bad_calendar' else request['id']
        attempted=dict(id=identity,calendar=calendar,start_utc=instant(start),end_utc=instant(end))
        title,note=request['title'],request['note']
        if action.startswith(('atomic_','sequential_')):
            old=conn.execute('SELECT * FROM events WHERE id=?',(identity,)).fetchone()
            if old is None:return dict(error='missing_event',attempted=attempted),66
            title,note=old[4:6]
            atomic=action.startswith('atomic_')
            if atomic:conn.execute('BEGIN IMMEDIATE')
            conn.execute('DELETE FROM events WHERE id=?',(identity,))
        conn.execute('INSERT INTO events VALUES(?,?,?,?,?,?)',(identity,calendar,start,end,title,note))
        if atomic:conn.execute('COMMIT')
        return dict(changed=True,attempted=attempted),0
    except sqlite3.IntegrityError as error:
        detail=str(error)
        allowed={'calendar_conflict','FOREIGN KEY constraint failed','UNIQUE constraint failed: events.id'}
        if detail not in allowed:raise
        if conn.in_transaction:conn.execute('ROLLBACK')
        return dict(error='constraint',detail=detail,attempted=attempted),65
    finally:conn.close()


if __name__=='__main__':
    result,code=run(sys.argv[1]);print(json.dumps(result,sort_keys=True));sys.exit(code)
