"""One public-observation contract for reservation, Lights Out, and 2048."""
from puffer_lab.native import NativeEpisode
from puffer_lab.contract import ACTIONS
from puffer_lab.environment_rl_data import public_item
from .native import Game
from .data import action_input


class Episode:
    def __init__(self, case, libraries):
        self.case, self.family, self.history, self.steps = case, case['family'], [], 0
        self.done, self.outcome, self.score = False, 0, 0.
        if self.family == 'reservation':
            self.native = NativeEpisode(libraries[self.family], case['world'], case['profile'])
            self.horizon = case['profile']['horizon']
        else:
            self.horizon = 12 if self.family == 'lightsout' else 24
            self.native = Game(libraries[self.family], self.family, case['seed'], case['board'], self.horizon)

    def input(self):
        if self.family == 'reservation':
            return public_item(self.native.public(), self.history, self.case['variant'])
        item = action_input(self.family, self.native.observe(), self.horizon - self.steps, self.history)
        if self.case['variant'] == 'reversed': item['options'].reverse()
        if self.case['variant'] == 'reworded':
            item['question'] = ('Which press should come next to finish with all lights off in as few presses as possible?'
                                if self.family == 'lightsout' else
                                'Which direction should we slide now to earn the most total game reward before the move budget ends?')
        return item

    def step(self, identity):
        action = int(identity[1:]); self.steps += 1
        if self.family == 'reservation':
            reward = self.native.step(ACTIONS[action]); after = self.native.public(); state = self.native.state()
            self.history.append(dict(action=ACTIONS[action], result=after['last_result']))
            self.done, self.outcome = bool(state['done']), state['outcome']
            return dict(reward=reward, after=after, verifier_state=state, done=self.done)
        result = self.native.step(action); self.history.append(action)
        self.done, self.score = result['done'], result['score']
        self.outcome = (1 if result['reward'] == 2. else 2) if self.family == 'lightsout' else (2 if result['terminated'] else 0)
        return dict(reward=result['reward'], after=result['observation'], verifier_state=result, done=self.done)

    def close(self): self.native.close()
