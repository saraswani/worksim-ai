from pydantic import BaseModel
from typing import Dict, Any, List
from tasks import get_task


class Observation(BaseModel):
    task_id: str
    task_name: str
    input_data: str
    history: List[Dict[str, Any]] = []
    step_count: int = 0
    done: bool = False


class Action(BaseModel):
    action_type: str
    output: str


class Reward(BaseModel):
    value: float
    breakdown: Dict[str, float] = {}
    feedback: str = ""


class Info(BaseModel):
    task_name: str
    step: int
    cumulative_reward: float
    done: bool
    grader_details: Dict[str, Any] = {}


class WorkSimEnv:
    def __init__(self):
        self._task = None
        self._step_count = 0
        self._cumulative_reward = 0.0
        self._done = False

    def reset(self):
        self._task = get_task()
        self._step_count = 0
        self._cumulative_reward = 0.0
        self._done = False

        return Observation(
            task_id=self._task["type"],
            task_name=self._task["type"],
            input_data=self._task["input"],
            history=[],
            step_count=0,
            done=False,
        )

    def step(self, action):
        self._step_count += 1

        expected = self._task["expected"]
        reward_value = 1.0 if action.output.strip() == expected else 0.0

        self._cumulative_reward += reward_value
        self._done = True

        obs = Observation(
            task_id=self._task["type"],
            task_name=self._task["type"],
            input_data=self._task["input"],
            history=[],
            step_count=self._step_count,
            done=True,
        )

        reward = Reward(
            value=reward_value,
            breakdown={},
            feedback="Correct" if reward_value else "Incorrect",
        )

        info = Info(
            task_name=self._task["type"],
            step=self._step_count,
            cumulative_reward=self._cumulative_reward,
            done=True,
            grader_details={},
        )

        return obs, reward, True, info

    def state(self):
        return self._task