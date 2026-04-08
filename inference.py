from env import WorkSimEnv

env = WorkSimEnv()

obs = env.reset()
done = False

while not done:
    action = {"action_type": "demo", "output": "urgent"}
    obs, reward, done, info = env.step(action)

print("Final reward:", reward)