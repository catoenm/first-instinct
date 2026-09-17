"""Twenty-four authored contracts, separate reference functions, and input streams.

These are synthetic tasks written for this pilot, not mined repository changes.
Reference functions and candidate implementations are different code paths, but
were not independently human-reviewed. Fixtures check their intended semantics.
"""
from dataclasses import dataclass
import random


@dataclass(frozen=True)
class Task:
    name: str
    family: str
    split: str
    description: str
    implementation: str
    alternative: str
    reference: object
    fixtures: tuple


def task(name,family,index,description,body,alternative,reference,fixtures,args='xs'):
    split=('train','train','validation','test')[index] if family!='predicates' else 'new_family'
    source=lambda b:f'def solve({args}):\n'+''.join('    '+line+'\n' for line in b.splitlines())
    return Task(name,family,split,description,source(body),source(alternative),reference,tuple(fixtures))


TASKS=(
 task('count_positive','aggregate',0,'Return the number of integers strictly greater than zero.',
      'return sum(1 for x in xs if x > 0)',
      'n = 0\nfor x in xs:\n    if x > 0:\n        n += 1\nreturn n',
      lambda xs:len(list(filter(lambda x:x>0,xs))),[([[-1,0,2]],1),([[]],0)]),
 task('sum_even','aggregate',1,'Return the sum of the even integers, including negative even integers.',
      'return sum(x for x in xs if x % 2 == 0)',
      'n = 0\nfor x in xs:\n    if x % 2 == 0:\n        n += x\nreturn n',
      lambda xs:sum(filter(lambda x:not x%2,xs)),[([[-2,1,4]],2),([[]],0)]),
 task('count_above','aggregate',2,'Return how many integers are strictly greater than limit.',
      'return sum(1 for x in xs if x > limit)',
      'return len([x for x in xs if x > limit])',
      lambda xs,limit:len(list(filter(lambda x:x>limit,xs))),[([[1,2,3],2],1),([[],0],0)],args='xs, limit'),
 task('sum_squares','aggregate',3,'Return the sum of the squares of all integers. Empty input returns zero.',
      'return sum(x * x for x in xs)',
      'n = 0\nfor x in xs:\n    n += x ** 2\nreturn n',
      lambda xs:sum(map(lambda x:pow(x,2),xs)),[([[-2,0,3]],13),([[]],0)]),
 task('max_default','ordering',0,'Return the greatest integer, or default if the list is empty.',
      'return max(xs) if len(xs) > 0 else default',
      'best = default\nif xs:\n    best = xs[0]\n    for x in xs:\n        if x > best:\n            best = x\nreturn best',
      lambda xs,default:sorted(xs)[-1] if xs else default,[([[-3,-1],7],-1),([[],7],7)],args='xs, default'),
 task('second_distinct','ordering',1,'Return the second-smallest distinct integer, or None if fewer than two distinct values exist.',
      'values = sorted(set(xs))\nreturn values[1] if len(values) >= 2 else None',
      'values = []\nfor x in sorted(xs):\n    if x not in values:\n        values.append(x)\nreturn values[1] if len(values) > 1 else None',
      lambda xs: min(set(xs)-{min(xs)}) if len(set(xs))>1 else None,[([[3,1,1,2]],2),([[4,4]],None)]),
 task('clamp','ordering',2,'Return value clamped to the inclusive interval [low, high]. low is at most high.',
      'return max(low, min(high, value))',
      'if value < low:\n    return low\nif value > high:\n    return high\nreturn value',
      lambda value,low,high: sorted([value,low,high])[1],[([8,1,5],5),([1,1,5],1)],args='value, low, high'),
 task('median_odd','ordering',3,'Input has positive odd length. Return its middle value after sorting.',
      'values = sorted(xs)\nreturn values[len(values) // 2]',
      'values = sorted(xs, reverse=True)\nreturn values[(len(values) - 1) // 2]',
      lambda xs:next(x for x in sorted(xs) if sum(v<x for v in xs)<=len(xs)//2 and sum(v>x for v in xs)<=len(xs)//2),
      [([[9,1,4]],4),([[2]],2)]),
 task('unique_order','sequence',0,'Remove repeated integers, preserving each integer\'s first occurrence.',
      'out = []\nfor x in xs:\n    if x not in out:\n        out.append(x)\nreturn out',
      'return list(dict.fromkeys(xs))',
      lambda xs:[x for i,x in enumerate(xs) if x not in xs[:i]],[([[2,1,2,3]], [2,1,3]),([[]],[])]),
 task('rotate_left','sequence',1,'Rotate the list left by k positions, allowing negative k. Return [] for empty input.',
      'if not xs:\n    return []\nk = k % len(xs)\nreturn xs[k:] + xs[:k]',
      'return [xs[(i + k) % len(xs)] for i in range(len(xs))]',
      lambda xs,k:list(__import__('collections').deque(xs)) if not xs else [xs[j] for j in [(i+k)%len(xs) for i in range(len(xs))]],
      [([[1,2,3],1],[2,3,1]),([[1,2,3],-1],[3,1,2]),([[],3],[])],args='xs, k'),
 task('chunks','sequence',2,'Split the list into consecutive chunks of positive size k. Keep a shorter final chunk.',
      'return [xs[i:i + k] for i in range(0, len(xs), k)]',
      'out = []\nfor x in xs:\n    if not out or len(out[-1]) == k:\n        out.append([])\n    out[-1].append(x)\nreturn out',
      lambda xs,k:[list(g) for g in [xs[i*k:(i+1)*k] for i in range((len(xs)+k-1)//k)]],
      [([[1,2,3,4,5],2],[[1,2],[3,4],[5]]),([[],2],[])],args='xs, k'),
 task('adjacent_diff','sequence',3,'Return each integer minus its immediate predecessor. Fewer than two elements returns [].',
      'return [xs[i] - xs[i - 1] for i in range(1, len(xs))]',
      'return [b - a for a, b in zip(xs, xs[1:])]',
      lambda xs:list(map(lambda p:p[1]-p[0],zip(xs[:-1],xs[1:]))),[([[1,4,2]],[3,-2]),([[1]],[])]),
 task('clean_spaces','text',0,'Collapse whitespace runs into one ASCII space and remove leading and trailing whitespace.',
      "return ' '.join(s.split())",
      "words = s.split()\nout = ''\nfor word in words:\n    if out:\n        out += ' '\n    out += word\nreturn out",
      lambda s:__import__('re').sub(r'\s+',' ',s).strip(),[(['  a\t b\n'],'a b'),(['   '],'')],args='s'),
 task('count_vowels','text',1,'Count ASCII vowels a, e, i, o, u, ignoring letter case.',
      "return sum(1 for c in s.lower() if c in 'aeiou')",
      "n = 0\nfor c in s:\n    if c.lower() in 'aeiou':\n        n += 1\nreturn n",
      lambda s:sum(s.lower().count(c) for c in 'aeiou'),[(['AEi! x'],3),(['xyz'],0)],args='s'),
 task('longest_word','text',2,'Split on whitespace. Return the first longest word, or an empty string if there are no words.',
      "words = s.split()\nreturn max(words, key=len) if words else ''",
      "best = ''\nfor word in s.split():\n    if len(word) > len(best):\n        best = word\nreturn best",
      lambda s:sorted(s.split(),key=lambda x:-len(x))[0] if s.split() else '',[(['one two a'],'one'),(['   '],'')],args='s'),
 task('first_index','text',3,'Return the index of the first case-insensitive occurrence of needle in s, or -1 if absent. ASCII strings only.',
      'return s.lower().find(needle.lower())',
      's = s.lower()\nneedle = needle.lower()\nfor i in range(len(s) - len(needle) + 1):\n    if s[i:i + len(needle)] == needle:\n        return i\nreturn -1',
      lambda s,needle: next((i for i in range(len(s)+1) if s[i:].lower().startswith(needle.lower())),-1),
      [(['abCabc','ca'],2),(['hi',''],0),(['abc','z'],-1)],args='s, needle'),
 task('count_values','mapping',0,'Return a dictionary of integer value frequencies. Keys in JSON output are strings.',
      'out = {}\nfor x in xs:\n    key = str(x)\n    out[key] = out.get(key, 0) + 1\nreturn out',
      'return {str(x): xs.count(x) for x in set(xs)}',
      lambda xs:{str(k):v for k,v in __import__('collections').Counter(xs).items()},[([[2,2,-1]],{'2':2,'-1':1}),([[]],{})]),
 task('invert_unique','mapping',1,'The dictionary has unique integer values. Return a dictionary mapping their string forms to original keys.',
      'return {str(v): k for k, v in d.items()}',
      'out = {}\nfor key in d:\n    out[str(d[key])] = key\nreturn out',
      lambda d:dict(zip(map(str,d.values()),d.keys())),[([{'a':2,'b':5}],{'2':'a','5':'b'}),([{}],{})],args='d'),
 task('total_values','mapping',2,'Return the sum of all integer dictionary values. Empty input returns zero.',
      'return sum(d.values())',
      'total = 0\nfor key in d:\n    total += d[key]\nreturn total',
      lambda d:sum(v for _,v in d.items()),[([{'a':2,'b':-3}],-1),([{}],0)],args='d'),
 task('merge_max','mapping',3,'Merge two dictionaries of integer values, using the greater value for keys present in both.',
      'out = dict(a)\nfor k, v in b.items():\n    if k not in out or v > out[k]:\n        out[k] = v\nreturn out',
      'return {k: max([d[k] for d in [a, b] if k in d]) for k in set(a) | set(b)}',
      lambda a,b:{k:max(a.get(k,float('-inf')),b.get(k,float('-inf'))) for k in a.keys()|b.keys()},
      [([{'x':-2},{'x':-3,'y':1}],{'x':-2,'y':1}),([{},{}],{})],args='a, b'),
 task('strict_sorted','predicates',0,'Return True exactly when every adjacent pair is strictly increasing. Empty and singleton lists satisfy this.',
      'return all(a < b for a, b in zip(xs, xs[1:]))',
      'for i in range(1, len(xs)):\n    if xs[i] <= xs[i - 1]:\n        return False\nreturn True',
      lambda xs:xs==sorted(set(xs)),[([[1,1]],False),([[]],True),([[1,2]],True)]),
 task('palindrome','predicates',1,'Return True exactly when s equals its reverse, case sensitive and including punctuation.',
      'return s == s[::-1]',
      'return all(s[i] == s[len(s) - i - 1] for i in range(len(s) // 2))',
      lambda s:all(a==b for a,b in zip(s,reversed(s))),[(['aba'],True),(['Aa'],False)],args='s'),
 task('contains_all','predicates',2,'Return True exactly when every distinct required integer occurs in xs. Multiplicity does not matter.',
      'return all(x in xs for x in required)',
      'return set(required).issubset(set(xs))',
      lambda xs,required:len(set(required)-set(xs))==0,[([[1],[1,1]],True),([[1],[2]],False)],args='xs, required'),
 task('all_in_range','predicates',3,'Return True exactly when every integer is in the inclusive interval [low, high]. Empty input satisfies this.',
      'return all(low <= x <= high for x in xs)',
      'for x in xs:\n    if x < low or x > high:\n        return False\nreturn True',
      lambda xs,low,high: not any(x not in range(low,high+1) for x in xs),[([[0,2],0,2],True),([[3],0,2],False)],args='xs, low, high'),
)


def input_case(t,rng):
    xs=[rng.randrange(-5,6) for _ in range(rng.randrange(0,9))]
    name=t.name
    if name in {'count_positive','sum_even','sum_squares','second_distinct','unique_order','adjacent_diff','count_values','strict_sorted'}:return [xs]
    if name=='count_above':return [xs,rng.randrange(-4,5)]
    if name=='max_default':return [xs,rng.randrange(-5,6)]
    if name=='clamp':
        a,b=sorted([rng.randrange(-5,6),rng.randrange(-5,6)])
        return [rng.randrange(-8,9),a,b]
    if name=='median_odd':return [[rng.randrange(-5,6) for _ in range(rng.choice([1,3,5,7]))]]
    if name=='rotate_left':return [xs,rng.randrange(-10,11)]
    if name=='chunks':return [xs,rng.randrange(1,6)]
    if name in {'clean_spaces','count_vowels','longest_word','first_index','palindrome'}:
        s=''.join(rng.choice('abCeIoUx \t\n') for _ in range(rng.randrange(0,22)))
        if name=='palindrome' and rng.random()<.4:s=s+s[::-1]
        if name=='first_index':
            needle=rng.choice(['','a','Z',s[:rng.randrange(1,4)]])
            return [s,needle]
        return [s]
    if name in {'invert_unique','total_values','merge_max'}:
        keys=rng.sample(list('abcde'),rng.randrange(0,6))
        d=dict(zip(keys,rng.sample(range(-8,9),len(keys))))
        if name=='merge_max':return [d,{k:rng.randrange(-8,9) for k in rng.sample(list('abcde'),rng.randrange(0,6))}]
        return [d]
    if name=='contains_all':return [xs,[rng.randrange(-5,6) for _ in range(rng.randrange(0,5))]]
    if name=='all_in_range':
        a,b=sorted([rng.randrange(-5,6),rng.randrange(-5,6)])
        return [xs,a,b]
    raise ValueError(name)


def cases(t,seed,count,exclude=()):
    import json
    rng=random.Random(seed)
    seen={json.dumps(x,sort_keys=True) for x in exclude}
    rows=[]
    while len(rows)<count:
        value=input_case(t,rng);key=json.dumps(value,sort_keys=True)
        if key not in seen:rows.append(value);seen.add(key)
    return rows
