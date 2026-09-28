"""Batched, partially observed foraging world; fitness never uses internal reward."""
from dataclasses import asdict, dataclass
import hashlib

import numpy as np


ACTIONS = ('up', 'right', 'down', 'left', 'wait')
MOVES = np.array([[-1, 0], [0, 1], [1, 0], [0, -1], [0, 0]])
FEATURES = ('alive', 'food', 'energy_change', 'novelty', 'avoid_hazard', 'avoid_death', 'avoid_collision')
RADIUS = 2
OBS_SIZE = 3 * (2 * RADIUS + 1) ** 2 + 1


@dataclass(frozen=True)
class WorldConfig:
    size: int = 12
    horizon: int = 128
    initial_energy: int = 24
    max_energy: int = 40
    food_energy: int = 14
    regrow_steps: int = 20
    plants: int = 8
    walls: int = 12
    hazards: int = 8
    hazard_cost: int = 3

    def __post_init__(self):
        if (not all(type(x) is int for x in asdict(self).values()) or
                self.size < 5 or self.horizon < 1 or not 1 <= self.initial_energy <= self.max_energy or
                min(self.food_energy, self.regrow_steps) < 1 or min(self.plants, self.walls, self.hazards, self.hazard_cost) < 0 or
                self.plants + self.walls + self.hazards + 1 >= self.size ** 2):
            raise ValueError('Invalid bounded world configuration')


def seed_for(namespace, role, index):
    if type(index) is not int or index < 0:
        raise ValueError('Invalid world index')
    return int.from_bytes(hashlib.sha256(f'{namespace}/{role}/{index}'.encode()).digest()[:8], 'little')


def connected(walls):
    free = set(map(tuple, np.argwhere(~walls)))
    reached = {next(iter(free))}
    pending = list(reached)
    while pending:
        y, x = pending.pop()
        for dy, dx in MOVES[:4]:
            p = (y + int(dy), x + int(dx))
            if p in free and p not in reached:
                reached.add(p); pending.append(p)
    return reached == free


class ForagingBatch:
    def __init__(self, seeds, config=WorldConfig()):
        self.config = config
        self.n = len(seeds)
        if not self.n:
            raise ValueError('Empty batch')
        shape = (self.n, config.size, config.size)
        self.walls = np.zeros(shape, dtype=bool)
        self.hazards = np.zeros(shape, dtype=bool)
        self.plants = np.zeros(shape, dtype=bool)
        self.cooldown = np.zeros(shape, dtype=np.int16)
        self.visits = np.zeros(shape, dtype=np.int16)
        self.position = np.zeros((self.n, 2), dtype=np.int64)
        self.energy = np.zeros(self.n, dtype=np.int16)
        self.steps = np.zeros(self.n, dtype=np.int32)
        self.energy_sum = np.zeros(self.n, dtype=np.int32)
        self.food_eaten = np.zeros(self.n, dtype=np.int32)
        self.hazard_visits = np.zeros(self.n, dtype=np.int32)
        self.done = np.zeros(self.n, dtype=bool)
        self.seeds = [0] * self.n
        self.reset(np.arange(self.n), seeds)

    def reset(self, indices, seeds):
        if len(indices) != len(seeds):
            raise ValueError('Reset identities differ')
        c = self.config
        for i, seed in zip(indices, seeds, strict=True):
            self.seeds[i] = int(seed)
            rng = np.random.default_rng(seed)
            for _ in range(100):
                cells = rng.permutation(c.size ** 2)
                self.walls[i] = False
                self.walls[i].flat[cells[:c.walls]] = True
                if connected(self.walls[i]):
                    break
            else:
                raise ValueError('Failed to sample connected world')
            self.hazards[i] = False; self.plants[i] = False
            self.hazards[i].flat[cells[c.walls:c.walls+c.hazards]] = True
            self.plants[i].flat[cells[c.walls+c.hazards:c.walls+c.hazards+c.plants]] = True
            start = int(cells[c.walls+c.hazards+c.plants])
            self.position[i] = divmod(start, c.size)
            self.energy[i] = c.initial_energy
            self.cooldown[i] = 0; self.visits[i] = 0
            self.visits[i, *self.position[i]] = 1
            self.steps[i] = self.energy_sum[i] = self.food_eaten[i] = self.hazard_visits[i] = 0
            self.done[i] = False

    def observe(self):
        c = self.config
        dy, dx = np.meshgrid(np.arange(-RADIUS, RADIUS+1), np.arange(-RADIUS, RADIUS+1), indexing='ij')
        y = self.position[:, 0, None] + dy.ravel()
        x = self.position[:, 1, None] + dx.ravel()
        outside = (y < 0) | (y >= c.size) | (x < 0) | (x >= c.size)
        y = np.clip(y, 0, c.size-1); x = np.clip(x, 0, c.size-1)
        b = np.arange(self.n)[:, None]
        food = self.plants[b, y, x] & (self.cooldown[b, y, x] == 0) & ~outside
        walls = self.walls[b, y, x] | outside
        hazards = self.hazards[b, y, x] & ~outside
        return np.concatenate((food, walls, hazards, self.energy[:, None]/c.max_energy), axis=1).astype(np.float32)

    def step(self, actions):
        actions = np.asarray(actions)
        if actions.shape != (self.n,) or actions.dtype.kind not in 'iu' or np.any((actions < 0) | (actions >= len(ACTIONS))):
            raise ValueError('Invalid action batch')
        c = self.config
        active = ~self.done
        before = self.energy.copy()
        np.subtract(self.cooldown, 1, out=self.cooldown, where=(self.cooldown > 0) & active[:, None, None])
        proposed = self.position + MOVES[actions]
        outside = np.any((proposed < 0) | (proposed >= c.size), axis=1)
        safe = np.clip(proposed, 0, c.size-1)
        b = np.arange(self.n)
        blocked = outside | self.walls[b, safe[:, 0], safe[:, 1]]
        moving = active & ~blocked
        self.position[moving] = proposed[moving]
        y, x = self.position.T
        hazard = self.hazards[b, y, x] & active
        food = self.plants[b, y, x] & (self.cooldown[b, y, x] == 0) & active
        self.cooldown[b[food], y[food], x[food]] = c.regrow_steps
        self.energy[active] = np.clip(before[active]-1-c.hazard_cost*hazard[active]+c.food_energy*food[active], 0, c.max_energy)
        # Eating takes place on arrival, before the starvation check.
        self.steps += active
        self.energy_sum += self.energy * active
        self.food_eaten += food
        self.hazard_visits += hazard
        self.visits[b[active], y[active], x[active]] += 1
        novelty = np.where(active & (actions != 4) & ~blocked, 1/np.sqrt(np.maximum(1, self.visits[b, y, x])), 0)
        starved = active & (self.energy == 0)
        self.done |= starved | (self.steps >= c.horizon)
        features = np.stack((active & ~starved, food, (self.energy-before)/c.food_energy,
                             novelty, -hazard.astype(float), -starved.astype(float),
                             -(blocked & active).astype(float)), axis=1).astype(np.float32)
        return np.clip(features, -1, 1), active & self.done

    def statistics(self):
        c = self.config
        mean_energy = self.energy_sum / np.maximum(self.steps, 1)
        return [dict(seed=self.seeds[i], steps=int(self.steps[i]), food_eaten=int(self.food_eaten[i]),
                     hazard_visits=int(self.hazard_visits[i]), mean_energy=float(mean_energy[i]),
                     survived_horizon=bool(self.steps[i] == c.horizon and self.energy[i] > 0),
                     fitness=float(self.steps[i] + mean_energy[i]/c.max_energy)) for i in range(self.n)]

    def frame(self, i=0):
        return dict(size=self.config.size, walls=self.walls[i].astype(int).tolist(),
                    hazards=self.hazards[i].astype(int).tolist(),
                    food=(self.plants[i] & (self.cooldown[i] == 0)).astype(int).tolist(),
                    position=self.position[i].tolist(), energy=int(self.energy[i]), steps=int(self.steps[i]))
