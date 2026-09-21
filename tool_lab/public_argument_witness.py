"""Typed witnesses derived only from a public goal, clock and earlier responses."""
import calendar
from datetime import datetime, timedelta
from decimal import Decimal
import math
import re

SECRET_KEYS = {'password', 'access_token', 'refresh_token', 'token'}
SECRET_APIS = {'login', 'show_account_passwords'}
CURRENCY = re.compile(r'(?<![\w$])\$(?:0|[1-9]\d{0,2}(?:,\d{3})+|[1-9]\d*)(?:\.\d{1,2})?(?![\w.,])')
WINDOW = re.compile(r'\blast ([1-9]\d{0,2}) days?\s*\(including today\)', re.I)


def strings(value, path=()):
    if isinstance(value, str):
        yield path, value
    elif isinstance(value, dict):
        for key, child in sorted(value.items()):
            if key not in SECRET_KEYS:
                yield from strings(child, path+(key,))
    elif isinstance(value, list):
        for i, child in enumerate(value):
            yield from strings(child, path+(i,))


def number(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError('Expected a finite non-Boolean number')
    return Decimal(str(value))


def currency_witness(history, value):
    try: target=number(value)
    except ValueError: return None
    for i,event in enumerate(history):
        if event['api'] in SECRET_APIS:continue
        for path,text in strings(event['response']):
            for match in CURRENCY.finditer(text):
                decimal=Decimal(match.group()[1:].replace(',',''))
                if decimal==target:
                    return dict(kind='currency_literal',source_call=i,response_path=list(path),
                                span=list(match.span()),lexeme=match.group(),decimal=str(decimal))
    return None


def calendar_shift(date, offset):
    """Independent calendar walk used to verify datetime day arithmetic."""
    if type(offset) is not int or not -999<=offset<=0:raise ValueError('Unsupported day offset')
    year,month,day=date.year,date.month,date.day
    for _ in range(-offset):
        if day>1:day-=1
        else:
            month-=1
            if month==0:year-=1;month=12
            if year<1:raise ValueError('Date outside supported calendar')
            day=calendar.monthrange(year,month)[1]
    return f'{year:04d}-{month:02d}-{day:02d}'


def clock_date(clock):
    value=clock['date']
    date=datetime.strptime(value,'%A, %B %d, %Y').date()
    if date.strftime('%A, %B %d, %Y')!=value:raise ValueError('Malformed public clock date')
    return date


def goal_offsets(goal):
    for match in WINDOW.finditer(goal):
        yield dict(rule='last_n_days_including_today',span=list(match.span()),offset=1-int(match.group(1)))
    for term,offset in (('yesterday',-1),('today',0)):
        for match in re.finditer(r'\b'+term+r'\b',goal,re.I):
            yield dict(rule=term,span=list(match.span()),offset=offset)


def date_witness(goal,clock,value):
    if not clock or not isinstance(value,str):return None
    date=clock_date(clock)
    for rule in goal_offsets(goal):
        computed=(date+timedelta(days=rule['offset'])).isoformat()
        if calendar_shift(date,rule['offset'])!=computed:raise ValueError('Date arithmetic disagrees')
        if computed==value:
            return dict(kind='calendar_date',clock_date=clock['date'],goal_span=rule['span'],
                        rule=rule['rule'],day_offset=rule['offset'],value=value)
    return None


def discover(goal,history,clock,argument,value):
    if argument=='amount':return currency_witness(history,value)
    if argument in ('min_created_at','max_created_at'):return date_witness(goal,clock,value)
    return None


def verify(witness,goal,history,clock,argument,value):
    """Reject forged witnesses rather than trusting a generated value."""
    if witness is None:raise ValueError('Missing public argument witness')
    if witness.get('kind')=='currency_literal':
        if argument!='amount':raise ValueError('Wrong witness type for argument')
        i=witness['source_call']
        if type(i) is not int or not 0<=i<len(history):raise ValueError('Future or invalid source call')
        event=history[i]
        if event['api'] in SECRET_APIS:raise ValueError('Secret response cannot ground an amount')
        path=witness['response_path']
        if not isinstance(path,list) or any(p in SECRET_KEYS for p in path if isinstance(p,str)):
            raise ValueError('Secret or malformed response path')
        text=event['response']
        for component in path:
            if type(component) not in (int,str):raise ValueError('Invalid response path')
            if isinstance(text,list) and (type(component) is not int or not 0<=component<len(text)):
                raise ValueError('Invalid list offset')
            text=text[component]
        if not isinstance(text,str):raise ValueError('Currency source is not text')
        matches=[m for m in CURRENCY.finditer(text) if list(m.span())==witness['span']]
        if len(matches)!=1 or matches[0].group()!=witness['lexeme']:raise ValueError('Currency span differs')
        literal=Decimal(matches[0].group()[1:].replace(',',''))
        if str(literal)!=witness['decimal'] or literal!=number(value):raise ValueError('Currency value differs')
    elif witness.get('kind')=='calendar_date':
        if argument not in ('min_created_at','max_created_at') or not clock or witness['clock_date']!=clock['date']:
            raise ValueError('Clock/argument witness differs')
        matching=[r for r in goal_offsets(goal) if r['span']==witness['goal_span']
                  and r['rule']==witness['rule'] and r['offset']==witness['day_offset']]
        if len(matching)!=1:raise ValueError('Date offset not supported by visible goal')
        date=clock_date(clock)
        if calendar_shift(date,witness['day_offset'])!=value or witness['value']!=value:
            raise ValueError('Derived date differs')
    else:raise ValueError('Unknown witness operation')
    return True
