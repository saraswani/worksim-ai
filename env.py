from __future__ import annotations
import os
import sys

# Tasks are now in the root directory to ensure compatibility 
# with the environment build context.
try:
    from email_task import EmailTriageTask
    from data_clean_task import DataCleaningTask
    from code_review_task import CodeReviewTask
except ImportError as e:
    print(f"\nCRITICAL IMPORT ERROR: {e}")
    print(f"Current Directory: {os.getcwd()}")
    print(f"Listing: {os.listdir('.')}\n")
    raise
from typing import List, Dict, Any, Tuple, Optional
from pydantic import BaseModel, Field

# ──────────────────────────────────────────────
# OpenEnv Models
# ──────────────────────────────────────────────

class Observation(BaseModel):
    task_id: str = Field(..., description="ID of the current task")
    task_name: str = Field(..., description="Human-readable name of the task")
    input_data: str = Field(..., description="The main input/prompt for the agent")
    history: List[Dict[str, Any]] = Field(default_factory=list, description="Step history in the current episode")
    step_count: int = Field(0, description="Number of steps taken in current episode")
    done: bool = Field(False, description="Whether the episode is finished")

class Action(BaseModel):
    action_type: str = Field(..., description="Type of action (e.g., classify_email, clean_data, review_code)")
    output: str = Field(..., description="The agent's text output or answer")

class Reward(BaseModel):
    value: float = Field(..., description="Scalar reward value (usually 0.0 - 1.0)")
    breakdown: Dict[str, float] = Field(default_factory=dict, description="Detailed components of the reward")
    feedback: str = Field("", description="Natural language feedback for the agent")

# ──────────────────────────────────────────────
# Environment
# ──────────────────────────────────────────────

class WorkSimEnv:
    """
    OpenEnv-compliant implementation for office productivity tasks.
    """
    
    TASK_MAP = {
        "email_triage": EmailTriageTask,
        "data_cleaning": DataCleaningTask,
        "code_review": CodeReviewTask
    }

    def __init__(self, task_name: str = "email_triage"):
        if task_name not in self.TASK_MAP:
            raise ValueError(f"Unknown task: {task_name}. Available: {list(self.TASK_MAP.keys())}")
        
        self.task_id = task_name
        self._task_impl = self.TASK_MAP[task_name]()
        self._history: List[Dict[str, Any]] = []
        self._step_count = 0
        self._done = False
        self._current_reward = 0.0

    def reset(self) -> Observation:
        """Reset the environment to a fresh episode for the current task."""
        self._task_impl.reset()
        self._history = []
        self._step_count = 0
        self._done = False
        self._current_reward = 0.0
        
        return self._get_obs()

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        """Perform one step in the environment."""
        if self._done:
            raise RuntimeError("Cannot step in a finished episode. Call reset() first.")

        self._step_count += 1
        
        # Get reward and feedback from the task implementation
        val, breakdown, feedback, task_done = self._task_impl.evaluate(action)
        
        # Penalize excessive steps (as per README rule)
        if self._step_count > 10:
            val -= 0.2
            feedback += " [Step penalty: too many turns]"

        reward = Reward(value=val, breakdown=breakdown, feedback=feedback)
        
        # Update state
        self._history.append({
            "step": self._step_count,
            "action": action.dict(),
            "reward": reward.dict()
        })
        self._done = task_done or self._step_count >= 15 # safety limit
        
        return self._get_obs(), reward, self._done, {}

    def state(self) -> Dict[str, Any]:
        """Return the current internal state of the environment."""
        return {
            "task_id": self.task_id,
            "step_count": self._step_count,
            "done": self._done,
            "history": self._history,
            "task_state": getattr(self._task_impl, "__dict__", {})
        }

    def _get_obs(self) -> Observation:
        return Observation(
            task_id=self.task_id,
            task_name=self.task_id.replace("_", " ").title(),
            input_data=self._task_impl.get_input(),
            history=self._history,
            step_count=self._step_count,
            done=self._done
        )