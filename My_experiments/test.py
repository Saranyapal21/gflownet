# from gflownet.envs.scrabble import Scrabble
# env = Scrabble()

# print(env.action_space)
# print(env.state)


"""
Deterministic 4×4 FrozenLake wrapped as a GFlowNetEnv.
State:   single int in {0,…,15}
Actions: 0=←, 1=↓, 2=→, 3=↑, 4=EOS (end‑of‑sequence)
Reward:  1 if goal reached, 0 otherwise
"""
from typing import List, Optional, Tuple, Union

import gymnasium as gym
import torch
from torchtyping import TensorType

from gflownet.envs.base import GFlowNetEnv
from gflownet.utils.common import tlong

class FrozenLake(GFlowNetEnv):
    def __init__(self,
                 map_name: str = "4x4",
                 is_slippery: bool = False,
                 **kwargs):
        # underlying Gym env
        self.env = gym.make("FrozenLake-v1",
                            map_name=map_name,
                            is_slippery=is_slippery)
        # EOS pseudo‑action
        self.eos = 4
        super().__init__(**kwargs)

    # ─────── required API ─────── #
    def get_action_space(self):
        # 4 navigation moves + EOS
        return [0, 1, 2, 3, self.eos]

    def reset(self):
        state, _ = self.env.reset()
        self.state = int(state)
        self.done = False
        self.n_actions = 0
        return self.state

    def step(self, action: Tuple[int],
             skip_mask_check: bool = False) -> Tuple[int, int, bool]:
        do_step, self.state, action = self._pre_step(
            action, skip_mask_check or self.skip_mask_check)
        if not do_step:
            return self.state, action, False

        # EOS: end immediately
        if action == self.eos:
            self.done = True
            self.n_actions += 1
            return self.state, action, True

        # take a step in Gym
        nxt, reward, term, trunc, _ = self.env.step(action)
        self.state = int(nxt)
        self.n_actions += 1
        self.done = term or trunc

        # auto‑inject EOS once done
        if self.done:
            action = self.eos
        return self.state, action, True

    # ───── masks ───── #
    def get_mask_invalid_actions_forward(self,
                                         state: Optional[int] = None,
                                         done: Optional[bool] = None):
        done = self._get_done(done)
        if done:
            # only EOS is valid
            return [a != self.eos for a in self.action_space]
        else:
            # navigation moves valid, EOS invalid
            return [a == self.eos for a in self.action_space]

    def get_parents(self, state=None, done=None, action=None):
        # dummy backward mask: only EOS brings you here
        return [state], [self.eos]

    # ───── encodings ───── #
    def states2policy(self,
                      states: Union[List[int],
                                    TensorType["batch"]]):
        """One‑hot encode 16 discrete tiles."""
        states = tlong(states, device=self.device)
        n = states.shape[0]
        out = torch.zeros(n, 16, dtype=self.float, device=self.device)
        out[torch.arange(n, device=self.device), states] = 1.
        return out

    def states2proxy(self, states):
        # just reuse one‑hot
        return self.states2policy(states)

    def state2readable(self, state=None, alphabet=None):
        s = self._get_state(state)
        r, c = divmod(s, 4)
        return f"({r},{c})"
