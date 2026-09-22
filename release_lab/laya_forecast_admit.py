"""Explicit pre-scoring correction for the L40S's usable-memory admission check.

Original frozen sources remain intact. Generate and retain a source overlay that
changes exactly one byte in the minimum-memory constant, then execute that file.
No model, formatting, calibration, data, numeric gate or deadline is altered.
"""
import json
from pathlib import Path
import runpy
import sys

from scale_lab.common import file_hash, write_json

BEFORE = 'torch.cuda.get_device_properties(0).total_memory < 45*1024**3'
AFTER = 'torch.cuda.get_device_properties(0).total_memory < 44*1024**3'


def corrected(source):
    if source.count(BEFORE) != 1 or AFTER in source:
        raise ValueError('Unexpected original admission condition')
    result=source.replace(BEFORE,AFTER)
    if len(result)!=len(source) or sum(a!=b for a,b in zip(source,result))!=1:
        raise ValueError('Admission correction must change exactly one byte')
    compile(result,'admitted-laya-forecast-run','exec')
    return result


def main():
    if sys.platform=='darwin':raise ValueError('No foundation inference on the Mac')
    import torch
    if '--root' not in sys.argv:raise ValueError('Explicit root required')
    root=Path(sys.argv[sys.argv.index('--root')+1]).resolve()
    props=torch.cuda.get_device_properties(0)
    if props.name!='NVIDIA L40S' or not 44*1024**3<=props.total_memory<48*1024**3:
        raise ValueError('This correction is specific to the observed nominal 48-GB L40S')
    if (root/'run').exists():raise ValueError('Correction is only eligible before task scoring starts')
    frozen=json.loads((root/'forecast-freeze.json').read_text())
    original=root/'release_lab/laya_forecast_run.py'
    if file_hash(original)!=frozen['files']['release_lab/laya_forecast_run.py']:
        raise ValueError('Original frozen runner changed')
    target=root/'hardware-admission';target.mkdir(exist_ok=False)
    overlay=target/'laya_forecast_run.py';overlay.write_text(corrected(original.read_text()))
    write_json(target/'correction.json',dict(status='explicit_pre_scoring_hardware_admission',
        original_sha256=file_hash(original),executed_overlay_sha256=file_hash(overlay),
        correction_entrypoint_sha256=file_hash(Path(__file__)),gpu=props.name,
        observed_cuda_memory_bytes=props.total_memory,minimum_before_bytes=45*1024**3,
        minimum_after_bytes=44*1024**3,only_changed_source_byte='5 -> 4 in minimum memory constant',
        primary_predictions_before_correction=0,original_frozen_source_unchanged=True))
    runpy.run_path(str(overlay),run_name='__main__')


if __name__=='__main__':main()
