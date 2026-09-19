"""Compare frozen Docker branch receipts with the unprivileged Linux catalog."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

from scale_lab.common import digest,file_hash,read_rows,write_json,write_rows
from tool_lab.decision_curriculum import CatalogExecutor,execute_branch

# The qualified reference Docker image uses SQLite 3.46.1. SQLite records the
# writer library version in bytes 96..99 of the database header. Compare every
# other byte, as well as every independently decoded schema/table value.
REFERENCE_SQLITE_WRITER_VERSION = 3046001


def compare(case,reference,actual,database_bytes=None):
    if actual==reference:return dict(kind='exact')
    if case['family']!='sqlite' or database_bytes is None:
        raise ValueError('Non-SQLite execution receipt differs')
    path=case['base']['expected']['path']
    if not any(e['action'].startswith('repair_') and e['observation'] and e['observation']['returncode']==0
               for e in actual['events']):
        raise ValueError('Cannot normalize an unmodified database')
    if len(database_bytes)<100 or database_bytes[:16]!=b'SQLite format 3\0':
        raise ValueError('Not a SQLite database header')
    actual_hash=hashlib.sha256(database_bytes).hexdigest()
    if actual_hash!=actual['after'][path]['sha256']:raise ValueError('Private read-back differs from receipt')
    normalized=database_bytes[:96]+REFERENCE_SQLITE_WRITER_VERSION.to_bytes(4,'big')+database_bytes[100:]
    reference_hash=reference['after'][path]['sha256']
    if hashlib.sha256(normalized).hexdigest()!=reference_hash:
        raise ValueError('Database differs beyond the four writer-version bytes')
    adjusted=copy.deepcopy(actual);adjusted['after'][path]['sha256']=reference_hash
    if adjusted!=reference:raise ValueError('Public observations, database contents or protected files differ')
    return dict(kind='sqlite_writer_version_only',reference_sha256=reference_hash,actual_sha256=actual_hash,
        reference_writer_version=REFERENCE_SQLITE_WRITER_VERSION,
        actual_writer_version=int.from_bytes(database_bytes[96:100],'big'),normalized_offsets=[96,97,98,99],
        every_other_database_byte_identical=True,database_never_modified_by_audit=True)


def database_readback(engine,case):
    if case['family']!='sqlite':return None
    name=case['base']['expected']['path']
    if Path(name).name!=name:raise ValueError('Database must be a flat fixture file')
    matches=list(Path(engine.folder.name).glob('*/'+name))
    if len(matches)!=1 or matches[0].is_symlink() or matches[0].stat().st_size>2*1024**2:
        raise ValueError('Unexpected private database read-back path')
    return matches[0].read_bytes()


def run(cases_path,references_path,output):
    cases={c['id']:c for c in read_rows(cases_path)};references=read_rows(references_path)
    if not 1<=len(references)<=252:raise ValueError('Shell parity branch cap')
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/'plan.json',dict(cases_sha256=file_hash(cases_path),references_sha256=file_hash(references_path),
                                     maximum_branches=252,maximum_commands=2400))
    engine=CatalogExecutor('catalog');checks=[];failure=None;reference=actual=None
    try:
        for reference in references:
            if engine.count+12>2400:raise ValueError('Shell parity command cap')
            case=cases[reference['case_id']];actual=None
            actual=execute_branch(case,engine,reference['action'],reference['plan'])
            equivalence=compare(case,reference,actual,database_readback(engine,case) if actual!=reference else None)
            checks.append(dict(id=actual['id'],receipt_sha256=digest(actual),equivalence=equivalence))
            write_json(output/'progress.json',dict(status='running',matched_branches=len(checks),actual_commands=engine.count))
        write_rows(output/'checks.jsonl',checks)
    except BaseException as error:
        failure=dict(type=type(error).__name__,detail=str(error))
        write_json(output/'mismatch.json',dict(error=failure,reference=reference,actual=actual,
                    actual_commands=engine.count,matched_branches=len(checks)))
        write_rows(output/'checks.jsonl',checks)
        raise
    finally:
        try:engine.close()
        except BaseException as error:
            failure=failure or dict(type=type(error).__name__,detail=str(error),phase='cleanup')
            raise
        finally:
            result=dict(status='failed' if failure else 'passed',branches=len(checks),actual_commands=engine.count,
                error=failure,sqlite_writer_version_differences=sum(c['equivalence']['kind']!='exact' for c in checks),
                contract='Exact observations/outcomes and protected bytes. After a SQLite mutation only, '
                    'a copy with writer-version bytes96..99 normalized must reproduce the reference file checksum; '
                    'schema, tables, integrity and every other database byte remain exact. The real database is never altered.')
            write_json(output/'parity.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('cases','references','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();print(json.dumps(run(a.cases,a.references,a.output),indent=2))
