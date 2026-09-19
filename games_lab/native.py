"""Python boundary around unmodified PufferLib Lights Out and 2048 mechanics."""
import ctypes as c
import json
from pathlib import Path
import subprocess
import sys

from scale_lab.common import file_hash

ROOT = Path(__file__).resolve().parent
GAMES = ('lightsout', 'g2048')


def verify_vendor():
    root = ROOT / 'vendor/puffer'
    manifest = json.loads((root / 'provenance.json').read_text())
    for name, record in manifest['files'].items():
        if file_hash(root / name) != record['sha256']:
            raise ValueError('Changed upstream file: ' + name)
    return manifest


def compile_game(game, output):
    if game not in GAMES:
        raise ValueError('Unknown pinned game')
    verify_vendor()
    output = Path(output).resolve(); output.parent.mkdir(parents=True, exist_ok=True)
    flags = ['-dynamiclib'] if sys.platform == 'darwin' else ['-shared', '-fPIC']
    if game == 'g2048':
        flags += ['-DGAME_2048']
    subprocess.run(['cc', '-std=c11', '-O2', *flags, '-I' + str(ROOT / 'vendor/puffer'),
                    str(ROOT / 'bridge.c'), str(ROOT / 'headless.c'), '-lm', '-o', str(output)], check=True)
    return output


def library(path):
    lib = c.CDLL(str(Path(path).resolve()))
    lib.game_create.argtypes = [c.c_uint, c.POINTER(c.c_int), c.c_int]; lib.game_create.restype = c.c_void_p
    lib.game_observe.argtypes = [c.c_void_p, c.POINTER(c.c_int)]
    lib.game_step.argtypes = [c.c_void_p, c.c_int]; lib.game_step.restype = c.c_float
    lib.game_done.argtypes = [c.c_void_p]; lib.game_done.restype = c.c_int
    lib.game_score.argtypes = [c.c_void_p]; lib.game_score.restype = c.c_float
    lib.game_close.argtypes = [c.c_void_p]
    if hasattr(lib, 'game_merge'):
        lib.game_merge.argtypes = [c.POINTER(c.c_int), c.c_int, c.POINTER(c.c_int), c.POINTER(c.c_float), c.POINTER(c.c_float)]
        lib.game_merge.restype = c.c_int
    return lib


class Game:
    def __init__(self, lib, game, seed, board=None, horizon=24):
        if game not in GAMES or not 1 <= horizon <= 1000:
            raise ValueError('Invalid game/horizon')
        self.lib, self.game, self.horizon = lib, game, horizon
        self.size = 25 if game == 'lightsout' else 16
        if board is not None and (len(board) != self.size or
             any(type(x) is not int or not 0 <= x <= (1 if game == 'lightsout' else 16) for x in board)):
            raise ValueError('Invalid board')
        self.pointer = lib.game_create(seed, (c.c_int * self.size)(*board) if board is not None else None, horizon)
        if not self.pointer:
            raise MemoryError('Game allocation failed')
        self.steps = 0; self.done = False; self.terminated = False

    def observe(self):
        # Upstream automatically resets on termination. Do not call the reset
        # board the terminal state of the game that just ended.
        if self.terminated:
            return None
        values = (c.c_int * self.size)(); self.lib.game_observe(self.pointer, values)
        return list(values)

    def step(self, action):
        maximum = 25 if self.game == 'lightsout' else 4
        if self.done or type(action) is not int or not 0 <= action < maximum:
            raise ValueError('Invalid action or completed episode')
        reward = float(self.lib.game_step(self.pointer, action))
        self.steps += 1; self.terminated = bool(self.lib.game_done(self.pointer))
        self.done = self.terminated or self.steps >= self.horizon
        return dict(reward=reward, done=self.done, terminated=self.terminated,
                    truncated=self.done and not self.terminated,
                    score=float(self.lib.game_score(self.pointer)), observation=self.observe())

    def close(self):
        if self.pointer:
            self.lib.game_close(self.pointer); self.pointer = None

    def __enter__(self): return self
    def __exit__(self, *args): self.close()


def merge(lib, board, action):
    if len(board) != 16 or action not in range(4):
        raise ValueError('Invalid merge input')
    after = (c.c_int * 16)(); reward, points = c.c_float(), c.c_float()
    changed = lib.game_merge((c.c_int * 16)(*board), action, after, c.byref(reward), c.byref(points))
    return dict(board=list(after), moved=bool(changed), reward=float(reward.value), points=int(points.value))
