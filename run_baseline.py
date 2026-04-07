"""
Baseline inference script for WorkSim AI.

Usage:
    HF_TOKEN=<your_token> python run_baseline.py

Reads HF_TOKEN from env and uses it with the OpenAI-compatible
Hugging Face Inference API to run a baseline model against all three tasks.
"""

from __future__ import annotations

import os
import json
import statistics
from typing import List, Dict, Any

from openai import OpenAI

from env import WorkSimEnv, Action

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
HF_TOKEN = os.environ.get("HF_TOKEN", "")
BASE_URL  = "https://api-inference.huggingface.co/v1/"
MODEL     = "mistralai/Mistral-7B-Instruct-v0.3"   # free HF model

TASK_NAMES = ["email_triage", "data_cleaning", "code_review"]
EPISODES_PER_TASK = 3

# ──────────────────────────────────────────────
# Client
# ──────────────────────────────────────────────
client = OpenAI(api_key=HF_TOKEN, base_url=BASE_URL)


def ask_model(system_prompt: str, user_content: str) -> str:
    """Call the LLM and return the raw text response."""
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_content},
            ],
            max_tokens=512,
            temperature=0.0,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"  [API error] {e}")
        return ""


# ──────────────────────────────────────────────
# Per-task prompts
# ──────────────────────────────────────────────
SYSTEM_PROMPTS = {
    "email_triage": (
        "You are an email assistant. "
        "When given an email, reply with EXACTLY one word: urgent, normal, or spam. "
        "No other words."
    ),
    "data_cleaning": (
        "You are a data engineer. "
        "When given a messy CSV and cleaning rules, return ONLY the cleaned CSV. "
        "No markdown, no explanation."
    ),
    "code_review": (
        "You are a Python expert. "
        "When given buggy Python code and a list of issues, return ONLY the corrected Python code. "
        "Use inline comments to mark each fix. No markdown fences."
    ),
}

ACTION_TYPES = {
    "email_triage":  "classify_email",
    "data_cleaning": "clean_data",
    "code_review":   "review_code",
}


# ──────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────
def run_task(task_name: str, episodes: int = 3) -> Dict[str, Any]:
    scores: List[float] = []
    system_prompt = SYSTEM_PROMPTS[task_name]
    action_type   = ACTION_TYPES[task_name]

    for ep in range(1, episodes + 1):
        env = WorkSimEnv(task_name=task_name)
        obs = env.reset()
        ep_reward = 0.0
        steps = 0

        print(f"\n  Episode {ep}/{episodes} – {task_name}")

        while not obs.done:
            model_output = ask_model(system_prompt, obs.input_data)
            action = Action(action_type=action_type, output=model_output)
            obs, reward, done, info = env.step(action)
            ep_reward += reward.value
            steps += 1

            print(f"    step {steps}: reward={reward.value:.3f}  | {reward.feedback[:80]}")

        # Normalise cumulative reward by number of steps
        normalised = ep_reward / max(steps, 1)
        scores.append(normalised)
        print(f"  ✓ Episode {ep} done. Normalised score: {normalised:.3f}")

    return {
        "task": task_name,
        "scores": scores,
        "mean": round(statistics.mean(scores), 4),
        "stdev": round(statistics.stdev(scores), 4) if len(scores) > 1 else 0.0,
    }


def main():
    if not HF_TOKEN:
        print("⚠  HF_TOKEN not set. Export it before running: export HF_TOKEN=hf_...")
        return

    print("=" * 60)
    print("  WorkSim AI – Baseline Evaluation")
    print("=" * 60)

    results = []
    for task in TASK_NAMES:
        print(f"\n{'─'*60}\n  Task: {task.upper()}\n{'─'*60}")
        result = run_task(task, episodes=EPISODES_PER_TASK)
        results.append(result)

    print("\n" + "=" * 60)
    print("  FINAL BASELINE SCORES")
    print("=" * 60)
    overall = []
    for r in results:
        print(f"  {r['task']:<20}  mean={r['mean']:.4f}  stdev={r['stdev']:.4f}")
        overall.append(r["mean"])
    print(f"\n  Overall mean score : {statistics.mean(overall):.4f}")
    print("=" * 60)

    # Save to file
    with open("baseline_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n  Results saved to baseline_results.json")


if __name__ == "__main__":
    main()
