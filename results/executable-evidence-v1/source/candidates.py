"""Generate proposals without asking the verifier whether they pass."""
import ast
import copy
import hashlib
import random

HELD_OUT={'boundary','input_slice'}


def digest_text(s):return hashlib.sha256(s.encode()).hexdigest()


def edits(tree,held_out=False):
    for i,n in enumerate(ast.walk(tree)):
        if isinstance(n,ast.Constant):
            if type(n.value) is int:
                for value in sorted({n.value-1,n.value+1,0,1} - {n.value}):
                    yield i,'literal',ast.Constant(value=value)
            elif isinstance(n.value,str):
                for value in sorted({'',' ',n.value.upper(),n.value[::-1]} - {n.value}):
                    yield i,'literal',ast.Constant(value=value)
        if isinstance(n,ast.BinOp):
            options={ast.Add:ast.Sub,ast.Sub:ast.Add,ast.Mult:ast.Add,ast.Mod:ast.FloorDiv,
                     ast.FloorDiv:ast.Mod,ast.BitOr:ast.BitAnd}
            if type(n.op) in options:
                v=copy.deepcopy(n);v.op=options[type(n.op)]();yield i,'arithmetic',v
        if isinstance(n,ast.Compare):
            for j,op in enumerate(n.ops):
                options={ast.Eq:ast.NotEq,ast.NotEq:ast.Eq,ast.In:ast.NotIn,ast.NotIn:ast.In}
                if type(op) in options:
                    v=copy.deepcopy(n);v.ops[j]=options[type(op)]();yield i,'predicate',v
                bounds={ast.Lt:ast.LtE,ast.LtE:ast.Lt,ast.Gt:ast.GtE,ast.GtE:ast.Gt}
                if held_out and type(op) in bounds:
                    v=copy.deepcopy(n);v.ops[j]=bounds[type(op)]();yield i,'boundary',v
        if isinstance(n,ast.Call):
            if isinstance(n.func,ast.Name):
                options={'sum':'len','len':'sum','max':'min','min':'max','all':'any','any':'all','sorted':'list','set':'list'}
                if n.func.id in options:
                    v=copy.deepcopy(n);v.func.id=options[n.func.id];yield i,'call',v
            if isinstance(n.func,ast.Attribute):
                options={'lower':'upper','find':'rfind','split':'splitlines'}
                if n.func.attr in options:
                    v=copy.deepcopy(n);v.func.attr=options[n.func.attr];yield i,'call',v
        if isinstance(n,ast.If):
            v=copy.deepcopy(n);v.test=ast.UnaryOp(op=ast.Not(),operand=v.test);yield i,'predicate',v
        if held_out and isinstance(n,ast.Name) and n.id in {'xs','s'} and isinstance(n.ctx,ast.Load):
            v=ast.Subscript(value=copy.deepcopy(n),slice=ast.Slice(lower=ast.Constant(value=1)),ctx=ast.Load())
            yield i,'input_slice',v


def replace(tree,index,replacement):
    t=copy.deepcopy(tree);target=list(ast.walk(t))[index]
    class Change(ast.NodeTransformer):
        def visit(self,node):
            return copy.deepcopy(replacement) if node is target else super().visit(node)
    return ast.fix_missing_locations(Change().visit(t))


def proposals(task,seed,limit=24,held_out=False):
    rng=random.Random(seed);pool={}
    def add(tree,mechanism,parent,edit):
        source=ast.unparse(tree)+'\n';sha=digest_text(source)
        if sha not in pool:pool[sha]={'code':source,'code_sha256':sha,'mechanism':mechanism,'parent_sha256':parent,'edit':edit}
    bases=[ast.parse(s) for s in (task.implementation,task.alternative)]
    if not held_out:
        for b in bases:add(b,'authored',None,None)
    for base in bases:
        parent=digest_text(ast.unparse(base)+'\n')
        for i,family,value in edits(base,held_out):
            if (family in HELD_OUT)!=held_out:continue
            add(replace(base,i,value),family,parent,{'node':i,'replacement':ast.dump(value)})
    # Compose ordinary transformations when simple sources yield few proposals.
    if not held_out and len(pool)<limit:
        first=list(pool.values());rng.shuffle(first)
        for p in first:
            if p['mechanism']=='authored':continue
            for i,family,value in edits(ast.parse(p['code'])):
                add(replace(ast.parse(p['code']),i,value),'composition',p['code_sha256'],
                    {'family':family,'node':i,'replacement':ast.dump(value)})
                if len(pool)>=limit*4:break
            if len(pool)>=limit*4:break
    originals=[p for p in pool.values() if p['mechanism']=='authored']
    mutants=[p for p in pool.values() if p['mechanism']!='authored'];rng.shuffle(mutants)
    selected=originals+mutants[:max(0,limit-len(originals))]
    rng.shuffle(selected)
    return selected
