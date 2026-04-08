from env import WorkSimEnv, Action

DEMO_AGENTS = {
    "email_triage": lambda obs: Action(
        action_type="classify_email",
        output="urgent"
    ),
    "data_cleaning": lambda obs: Action(
        action_type="clean_data",
        output="John,25\nAlice,20"
    ),
    "code_review": lambda obs: Action(
        action_type="review_code",
        output="def add(a,b):\n    return a+b"
    ),
}


def run_demo(task_name):
    print(f"\n{'='*55}")
    print(f"  DEMO – {task_name.upper()}")
    print(f"{'='*55}")

    # ✅ FIXED
    env = WorkSimEnv()

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