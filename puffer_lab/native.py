"""Small ctypes driver for the same C core used by the PufferLib adapter."""

import ctypes as c
from pathlib import Path
import subprocess
import sys

from .contract import ACTIONS, OBS_SIZE, STATE_NAMES, public

ROOT = Path(__file__).resolve().parent


def compile_core(output):
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    flags = ['-dynamiclib'] if sys.platform == 'darwin' else ['-shared', '-fPIC']
    subprocess.run(['cc', '-std=c11', '-O2', '-Wall', '-Wextra', '-Werror', *flags,
                    str(ROOT/'native_bridge.c'), '-o', str(output)], check=True)
    return output


def library(path):
    lib = c.CDLL(str(Path(path).resolve()))
    lib.reservation_create.argtypes = [c.c_int,c.c_int,c.POINTER(c.c_int),c.POINTER(c.c_int)]
    lib.reservation_create.restype = c.c_void_p
    lib.reservation_step.argtypes = [c.c_void_p,c.c_int]
    lib.reservation_step.restype = c.c_float
    lib.reservation_observe.argtypes = [c.c_void_p,c.POINTER(c.c_float)]
    lib.reservation_dump.argtypes = [c.c_void_p,c.POINTER(c.c_int)]
    lib.reservation_destroy.argtypes = [c.c_void_p]
    return lib


class NativeEpisode:
    def __init__(self, lib, world, profile):
        self.lib,self.profile = lib,profile
        self.ptr = lib.reservation_create(world,profile['horizon'],
            (c.c_int*9)(*profile['costs']),(c.c_int*6)(*profile['prior']))
        if not self.ptr: raise ValueError('Invalid native environment parameters')

    def state(self):
        values = (c.c_int*len(STATE_NAMES))()
        self.lib.reservation_dump(self.ptr,values)
        return dict(zip(STATE_NAMES,values))

    def observe(self):
        values = (c.c_float*OBS_SIZE)()
        self.lib.reservation_observe(self.ptr,values)
        return list(values)

    def public(self):
        return public(self.state(),self.profile)

    def step(self, action):
        if self.state()['done']: raise ValueError('Episode already ended')
        return float(self.lib.reservation_step(self.ptr,ACTIONS.index(action)))

    def close(self):
        if self.ptr:
            self.lib.reservation_destroy(self.ptr);self.ptr = None

    def __enter__(self): return self
    def __exit__(self,*args): self.close()
