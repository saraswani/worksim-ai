"""
WorkSim AI – OpenEnv-compliant real-world office task simulation environment.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel

from tasks.email_task import EmailTriageTask
from tasks.data_clean_task import DataCleaningTask
from tasks.code_review_task import CodeReviewTask


# ──────────────────────────────────────────────
# Pydantic Models
# ──────────────────────────────────────────────

class Observation(BaseModel):
    task_id: str
    task_name: str
    input_data: str
    history: List[Dict[str, Any]] = []
    step_count: int = 0
    done: bool = False


class Action(BaseModel):
    action_type: str          # e.g. "classify_email", "clean_data", "review_code"
    output: str               # agent's answer / fix / classification


class Reward(BaseModel):
    value: float              # 0.0 – 1.0
    breakdown: Dict[str, float] = {}
    feedback: str = ""


class Info(BaseModel):
    task_name: str
    step: int
    cumulative_reward: float
    done: bool
    grader_details: Dict[str, Any] = {}


# ──────────────────────────────────────────────
# Environment
# ──────────────────────────────────────────────

TASK_REGISTRY = {
    "email_triage":   EmailTriageTask,
    "data_cleaning":  DataCleaningTask,
    "code_review":    CodeReviewTask,
}


class WorkSimEnv:
    """
    OpenEnv-compliant environment for real-world office task simulation.
    Supports three tasks: email_triage, data_cleaning, code_review.
    """

    METADATA = {
        "name": "WorkSim-AI",
        "version": "1.0.0",
        "tasks": list(TASK_REGISTRY.keys()),
        "max_steps": 10,
    }

    def __init__(self, task_name: str = "email_triage"):
        if task_name not in TASK_REGISTRY:
            raise ValueError(f"Unknown task '{task_name}'. Choose from {list(TASK_REGISTRY.keys())}")
        self.task_name = task_name
        self._task = TASK_REGISTRY[task_name]()
        self._step_count = 0
        self._cumulative_reward = 0.0
        self._history: List[Dict[str, Any]] = []
        self._done = False

    # ── OpenEnv Interface ──────────────────────

    def reset(self) -> Observation:
        """Reset the environment and return the initial observation."""
        self._step_count = 0
        self._cumulative_reward = 0.0
        self._history = []
        self._done = False
        self._task.reset()
        return Observation(
            task_id=self.task_name,
            task_name=self.task_name,
            input_data=self._task.get_input(),
            history=[],
            step_count=0,
            done=False,
        )

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Info]:
        """Execute one action and return (observation, reward, done, info)."""
        if self._done:
            raise RuntimeError("Episode is done. Call reset() first.")

        self._step_count += 1

        # Penalise excessive looping
        loop_penalty = 0.0
        if self._step_count > self.METADATA["max_steps"]:
            loop_penalty = -0.2
            self._done = True

        # Score the action
        reward_value, breakdown, feedback, done = self._task.evaluate(action)
        reward_value = max(0.0, min(1.0, reward_value + loop_penalty))

        self._cumulative_reward += reward_value
        self._done = self._done or done

        self._history.append({
            "step": self._step_count,
            "action": action.model_dump(),
            "reward": reward_value,
            "feedback": feedback,
        })

        obs = Observation(
            task_id=self.task_name,
            task_name=self.task_name,
            input_data=self._task.get_input(),
            history=list(self._history),
            step_count=self._step_count,
            done=self._done,
        )
        reward = Reward(value=reward_value, breakdown=breakdown, feedback=feedback)
        info = Info(
            task_name=self.task_name,
            step=self._step_count,
            cumulative_reward=self._cumulative_reward,
            done=self._done,
            grader_details=breakdown,
        )
        return obs, reward, self._done, info

    def state(self) -> Dict[str, Any]:
        """Return the full current state of the environment."""
        return {
            "task_name": self.task_name,
            "step_count": self._step_count,
            "cumulative_reward": self._cumulative_reward,
            "done": self._done,
            "history": self._history,
            "current_input": self._task.get_input(),
        }
