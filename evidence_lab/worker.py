"""Execute only the bounded, authored function subset used by this pilot.

This is not a security sandbox for arbitrary third-party code. The controller
also uses a fresh process, wall-time limit and a maximum number of input cases.
"""
import ast
import builtins
import contextlib
import io
import json
import sys

BUILTINS={'all','any','dict','list','set','sorted','sum','max','min','len','str','range','zip','enumerate','reversed','abs'}
METHODS={'append','get','items','keys','values','fromkeys','lower','upper','find','rfind','split','splitlines','join','count','issubset'}
NODES={ast.Module,ast.FunctionDef,ast.arguments,ast.arg,ast.Return,ast.Assign,ast.AugAssign,ast.Expr,
       ast.For,ast.If,ast.IfExp,ast.ListComp,ast.DictComp,ast.GeneratorExp,ast.comprehension,
       ast.Call,ast.Name,ast.Load,ast.Store,ast.Constant,ast.List,ast.Tuple,ast.Dict,ast.Set,
       ast.Subscript,ast.Slice,ast.Attribute,ast.keyword,ast.BinOp,ast.UnaryOp,ast.BoolOp,ast.Compare,
       ast.Add,ast.Sub,ast.Mult,ast.Div,ast.FloorDiv,ast.Mod,ast.Pow,ast.BitOr,ast.BitAnd,
       ast.USub,ast.UAdd,ast.Not,ast.And,ast.Or,ast.Eq,ast.NotEq,ast.Lt,ast.LtE,ast.Gt,ast.GtE,
       ast.In,ast.NotIn,ast.Is,ast.IsNot}


def validate(source):
    tree=ast.parse(source)
    if len(tree.body)!=1 or not isinstance(tree.body[0],ast.FunctionDef) or tree.body[0].name!='solve':
        raise ValueError('Expected exactly one solve function')
    if tree.body[0].decorator_list or tree.body[0].args.defaults:raise ValueError('No decorators or defaults')
    for n in ast.walk(tree):
        if type(n) not in NODES:raise ValueError(f'Unsupported syntax: {type(n).__name__}')
        if isinstance(n,ast.Name) and n.id.startswith('__'):raise ValueError('No private names')
        if isinstance(n,ast.Attribute) and n.attr not in METHODS:raise ValueError('Unsupported method')
        if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id not in BUILTINS:
            raise ValueError('Unsupported call')
        if isinstance(n,ast.Call) and not isinstance(n.func,(ast.Name,ast.Attribute)):raise ValueError('Unsupported callable')
    return tree


def execute(source,inputs):
    if len(inputs)>128:raise ValueError('Too many input cases')
    tree=validate(source)
    namespace={'__builtins__':{n:getattr(builtins,n) for n in BUILTINS}}
    exec(compile(tree,'<candidate>','exec'),namespace)
    result=[]
    for args in inputs:
        try:
            # Fresh JSON objects for each call prevent state sharing through arguments.
            value=namespace['solve'](*json.loads(json.dumps(args)))
            encoded=json.dumps(value,sort_keys=True,allow_nan=False)
            if len(encoded)>20000:raise ValueError('Output exceeds limit')
            result.append({'value':json.loads(encoded),'error':None})
        except Exception as exc:
            result.append({'value':None,'error':type(exc).__name__})
    return result


if __name__=='__main__':
    request=json.load(sys.stdin)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            results=execute(request['code'],request['inputs'])
        response={'results':results,'error':None}
    except Exception as exc:response={'results':None,'error':type(exc).__name__}
    print(json.dumps(response,allow_nan=False))
