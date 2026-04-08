"""
run_env.py – Quick interactive demo of WorkSim AI.
Runs one episode of each task with a naive rule-based agent (no API key needed).
"""

from env import WorkSimEnv, Action

DEMO_AGENTS = {
    "email_triage": lambda obs: Action(
        action_type="classify_email",
        output="urgent" if "urgent" in obs.input_data.lower() or "down" in obs.input_data.lower()
               else "spam" if "free" in obs.input_data.lower() or "win" in obs.input_data.lower()
               else "normal",
    ),
    "data_cleaning": lambda obs: Action(
        action_type="clean_data",
        output="\n".join(
            line for line in obs.input_data.split("\n")
            if "," in line and line.split(",")[0].strip()
        ),
    ),
    "code_review": lambda obs: Action(
        action_type="review_code",
        output="# Fixed code\ndef placeholder():\n    pass\n",
    ),
}

def run_demo(task_name: str):
    print(f"\n{'='*55}")
    print(f"  DEMO – {task_name.upper()}")
    print(f"{'='*55}")
    env = WorkSimEnv(task_name=task_name)
    obs = env.reset()
    agent = DEMO_AGENTS[task_name]
    step = 0

    while not obs.done:
        step += 1
        action = agent(obs)
        obs, reward, done, info = env.step(action)
        print(f"  step {step} | reward={reward.value:.3f} | {reward.feedback}")

    print(f"  Cumulative reward: {info.cumulative_reward:.3f}")

if __name__ == "__main__":
    for task in ["email_triage", "data_cleaning", "code_review"]:
        run_demo(task)
