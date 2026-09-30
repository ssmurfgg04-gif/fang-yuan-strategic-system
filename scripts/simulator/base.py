"""Simulator base: hidden-state environment protocol + metrics + agent protocol.

Design rules (from the project notes):
- The agent never sees the full state; it sees observations.
- The environment, not the model, decides whether actions work.
- Resource accounting is exact (Python does arithmetic; the model only selects).
- Irreversible failure states exist (ruin).
- Metrics: survival, final resources, max drawdown, ruin, information gained,
  future options remaining, dependency on single actors, recovery after betrayal.
"""
from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class StepResult:
    observation: dict
    outcome: str
    succeeded: bool
    reward_events: list[str] = field(default_factory=list)   # for Stage-C RL hooks
    info: dict = field(default_factory=dict)


@dataclass
class SimMetrics:
    survival: bool = True
    final_resources: float = 0.0
    max_drawdown: float = 0.0
    ruined: bool = False
    information_gained: float = 0.0
    future_options_remaining: int = 0
    dependency_max: float = 0.0
    recovered_after_betrayal: bool = False
    turns_survived: int = 0

    def as_dict(self) -> dict:
        return self.__dict__.copy()


class Environment(ABC):
    """Turn-based partially-observed environment."""
    name: str = "base"
    horizon: int = 10

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)
        self.turn = 0
        self.history: list[dict] = []

    @abstractmethod
    def observation(self) -> dict:
        """Compact observation for the agent (target 300-800 token equivalent)."""

    @abstractmethod
    def actions(self) -> list[str]:
        """Legal action keys for the current turn."""

    @abstractmethod
    def step(self, action: str) -> StepResult:
        """Apply action; update hidden state; return next observation."""

    @abstractmethod
    def metrics(self) -> SimMetrics:
        """Final evaluation of the trajectory."""

    def done(self) -> bool:
        return self.turn >= self.horizon

    def run(self, agent) -> list[dict]:
        """Full trajectory loop."""
        traj = []
        while not self.done():
            obs = self.observation()
            obs["_env"] = self.name          # routing hint for agents (stripped in logs)
            legal = self.actions()
            action = agent.choose(obs, legal)
            result = self.step(action)
            self.turn += 1
            traj.append({
                "turn": self.turn, "action": action,
                "outcome": result.outcome, "succeeded": result.succeeded,
                "reward_events": result.reward_events,
                "observation_after": {k: v for k, v in result.observation.items()
                                      if not k.startswith("_")},
            })
            self.history.append(traj[-1])
        return traj


class Agent(ABC):
    """Agent protocol: choose an action from an observation + legal actions."""
    name: str = "agent"

    @abstractmethod
    def choose(self, obs: dict, legal_actions: list[str]) -> str: ...


def max_drawdown(values: list[float]) -> float:
    peak, mdd = float("-inf"), 0.0
    for v in values:
        peak = max(peak, v)
        if peak > 0:
            mdd = max(mdd, (peak - v) / peak)
    return mdd
