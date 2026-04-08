import time
import os
import sys

def run_demo():
    from env import WorkSimEnv, Action
    print("=" * 60)
    print("  WorkSim AI – OpenEnv Demo")
    print("=" * 60)

    tasks_to_demo = ["email_triage", "data_cleaning", "code_review"]

    for task_name in tasks_to_demo:
        print(f"\n[DEMO] Starting Task: {task_name.upper()}")
        print("-" * 40)
        
        env = WorkSimEnv(task_name=task_name)
        obs = env.reset()
        
        step = 1
        while not obs.done:
            print(f"\nStep {step}")
            print(f"Input Data Snippet: {obs.input_data[:100]}...")
            
            # Simple rule-based/mock responses for demo purposes
            if task_name == "email_triage":
                model_output = "urgent" if "urgent" in obs.input_data.lower() or "server" in obs.input_data.lower() else "normal"
            elif task_name == "data_cleaning":
                # Returns dummy CSV for cleaning demo
                model_output = "Name,Age,Department\nAlice,30,Engineering\nBob,30,Marketing\nDave,28,Engineering"
            else:
                # Returns dummy fix for code review demo
                model_output = "def find_max(lst):\n    max_val = lst[0]\n    for i in range(len(lst)):\n        if lst[i] > max_val:\n            max_val = lst[i]\n    return max_val"

            action = Action(action_type=f"demo_{task_name}", output=model_output)
            obs, reward, done, _ = env.step(action)
            
            print(f"Action Type: {action.action_type}")
            print(f"Reward Value: {reward.value}")
            print(f"Feedback: {reward.feedback}")
            
            step += 1
            if step > 5: break # Don't loop forever in demo
            
        print(f"\n[DEMO] Task {task_name} Finished.")
        print("=" * 60)

if __name__ == "__main__":
    run_demo()