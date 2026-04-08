---
title: WorkSim AI
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: docker
app_file: run_env.py
pinned: false
---
# WorkSim AI 🏢🤖

> **OpenEnv-compliant simulation environment for real-world office task evaluation of AI agents.**

WorkSim AI benchmarks AI agents on the kind of work humans actually do every day — not toy puzzles or games. Agents triage emails, clean messy data, and fix buggy code, receiving incremental feedback and a final score for each task.

---

## Motivation

Most existing AI benchmarks test agents on trivia, maths, or synthetic puzzles. Real-world deployments require agents that can handle ambiguous, messy, context-dependent tasks. WorkSim AI fills this gap by providing a structured, reproducible evaluation harness for **office productivity tasks**.

---

## Environment Overview

| Property | Value |
|---|---|
| Interface | OpenEnv (Pydantic models, `reset / step / state`) |
| Tasks | 3 (easy → medium → hard) |
| Reward | Incremental, scalar, range `[-0.5, 1.0]` |
| Containerised | ✅ Docker + Hugging Face Spaces |

---

## Action & Observation Spaces

### Observation
```python
class Observation(BaseModel):
    task_id:    str
    task_name:  str
    input_data: str          # what the agent sees
    history:    List[dict]   # previous steps
    step_count: int
    done:       bool
```

### Action
```python
class Action(BaseModel):
    action_type: str   # "classify_email" | "clean_data" | "review_code"
    output:      str   # agent's answer
```

### Reward
```python
class Reward(BaseModel):
    value:     float          # 0.0 – 1.0 (can be negative for bad actions)
    breakdown: Dict[str, float]
    feedback:  str
```

---

## Task Descriptions

### ✅ Task 1 — Email Triage (Easy)
- **Goal:** Classify each email as `urgent`, `normal`, or `spam`.
- **Reward:** `+1.0` correct, `-0.5` wrong.
- **Episode length:** 5 emails.
- **Example:**
  - `"Server down!!"` → `urgent`
  - `"Team lunch?"` → `normal`
  - `"Win a free iPhone!"` → `spam`

### ✅ Task 2 — Data Cleaning (Medium)
- **Goal:** Apply stated cleaning rules to a messy CSV and return the corrected version.
- **Reward:** F1-based scoring against expected output (0.0 – 1.0).
- **Episode length:** Single step.
- **Example:** Remove rows with missing keys, convert word-form numbers to integers.

### ✅ Task 3 — Code Review (Hard)
- **Goal:** Fix a Python snippet containing syntax errors, logic bugs, or design issues.
- **Reward:** 40% syntax validity + 60% keyword coverage of expected fixes (0.0 – 1.0).
- **Episode length:** Single step.
- **Example:** Fix off-by-one indexing, missing `self`, wrong operator.

---

## Reward Function Design

| Signal | When | Effect |
|---|---|---|
| Correct classification | Each step (email triage) | +1.0 |
| Wrong classification | Each step (email triage) | -0.5 |
| F1 score | End of episode (data cleaning) | 0.0 – 1.0 |
| Syntax + keyword score | End of episode (code review) | 0.0 – 1.0 |
| Excessive steps (> 10) | Any task | -0.2 penalty |

The reward is always incremental — agents receive feedback after **every** step, not just at the end.

---

## Baseline Performance

Run with `Mistral-7B-Instruct-v0.3` via Hugging Face Inference API:

| Task | Mean Score | Stdev |
|---|---|---|
| email_triage | 0.72 | 0.08 |
| data_cleaning | 0.55 | 0.12 |
| code_review | 0.41 | 0.15 |
| **Overall** | **0.56** | — |

*(Scores are approximate; re-run `run_baseline.py` for reproducible results.)*

---

## Setup & Usage

### Local (no Docker)

```bash
# Install dependencies
pip install -r requirements.txt

# Quick demo (no API key needed)
python run_env.py

# Baseline evaluation (requires HF_TOKEN)
export HF_TOKEN=hf_your_token_here
python run_baseline.py
```

### Docker

```bash
# Build
docker build -t worksim-ai .

# Run demo
docker run worksim-ai

# Run baseline
docker run -e HF_TOKEN=hf_your_token_here worksim-ai python run_baseline.py
```

### Hugging Face Spaces

1. Fork this repo into a new HF Space (Docker SDK).
2. Set `HF_TOKEN` as a Space secret.
3. The Space will run the demo on startup.

---

## Project Structure

```
worksim-ai/
├── env.py              # Main WorkSimEnv class (OpenEnv interface)
├── tasks/
│   ├── email_task.py   # Task 1 – Email Triage
│   ├── data_clean_task.py  # Task 2 – Data Cleaning
│   └── code_review_task.py # Task 3 – Code Review
├── run_env.py          # Demo script (rule-based agent, no API key)
├── run_baseline.py     # Baseline evaluation (LLM via HF Inference API)
├── openenv.yaml        # OpenEnv metadata
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## License

MIT
