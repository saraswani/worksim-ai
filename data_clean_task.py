"""Task 2 – Data Cleaning (Medium)"""

from __future__ import annotations
from typing import Dict, List, Tuple, Any
import random
import csv
import io


RAW_DATASETS = [
    {
        "description": "Employee records with missing names and non-numeric ages.",
        "raw": "Name,Age,Department\nAlice,30,Engineering\n,25,HR\nBob,thirty,Marketing\nCarol,,Finance\nDave,28,Engineering",
        "rules": [
            "Remove rows where Name is empty.",
            "Convert word-form ages (e.g. 'thirty') to integers.",
            "Remove rows where Age is missing or cannot be converted.",
        ],
        "expected": "Name,Age,Department\nAlice,30,Engineering\nBob,30,Marketing\nDave,28,Engineering",
    },
    {
        "description": "Sales data with inconsistent date formats and null values.",
        "raw": "Product,Sales,Date\nWidget A,1500,2024-01-15\nWidget B,,Jan 20 2024\nWidget C,3200,2024-01-25\n,2100,2024-01-30\nWidget D,0,2024-02-01",
        "rules": [
            "Remove rows where Product is empty.",
            "Remove rows where Sales is empty or zero.",
            "Standardise Date to YYYY-MM-DD.",
        ],
        "expected": "Product,Sales,Date\nWidget A,1500,2024-01-15\nWidget C,3200,2024-01-25",
    },
    {
        "description": "Customer contact list with duplicate emails and missing phone numbers.",
        "raw": "Name,Email,Phone\nJohn,john@example.com,555-1234\nJane,jane@example.com,\nJohn,john@example.com,555-1234\nBob,bob@example.com,555-5678\nAnna,,555-9999",
        "rules": [
            "Remove duplicate rows (same Name+Email).",
            "Remove rows where Email is empty.",
            "Keep rows even if Phone is missing.",
        ],
        "expected": "Name,Email,Phone\nJohn,john@example.com,555-1234\nJane,jane@example.com,\nBob,bob@example.com,555-5678",
    },
]

WORD_TO_NUM = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
}


def _parse_csv(text: str) -> List[List[str]]:
    reader = csv.reader(io.StringIO(text.strip()))
    return [row for row in reader]


def _score_cleaning(expected: str, actual: str) -> Tuple[float, Dict[str, float]]:
    exp_rows = _parse_csv(expected)
    try:
        act_rows = _parse_csv(actual)
    except Exception:
        return 0.0, {"parse_error": 1.0}

    if not exp_rows or not act_rows:
        return 0.0, {"empty_output": 1.0}

    # Header match
    header_score = 1.0 if exp_rows[0] == act_rows[0] else 0.0

    # Row-level F1
    exp_data = set(tuple(r) for r in exp_rows[1:])
    act_data = set(tuple(r) for r in act_rows[1:])

    tp = len(exp_data & act_data)
    fp = len(act_data - exp_data)
    fn = len(exp_data - act_data)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    final = 0.2 * header_score + 0.8 * f1
    return round(final, 4), {"header": header_score, "precision": precision, "recall": recall, "f1": f1}


class DataCleaningTask:
    """
    Medium task: clean a messy CSV dataset according to stated rules.
    Single-step evaluation – agent gets one dataset per episode.
    """

    def __init__(self):
        self._dataset: dict = {}
        self._attempted: bool = False

    def reset(self):
        self._dataset = random.choice(RAW_DATASETS)
        self._attempted = False

    def get_input(self) -> str:
        return (
            f"Data Cleaning Task\n"
            f"Description: {self._dataset['description']}\n\n"
            f"Rules:\n" + "\n".join(f"  {i+1}. {r}" for i, r in enumerate(self._dataset["rules"])) +
            f"\n\nRaw CSV:\n{self._dataset['raw']}\n\n"
            f"Return ONLY the cleaned CSV (header + rows, no markdown)."
        )

    def evaluate(self, action) -> Tuple[float, Dict[str, float], str, bool]:
        self._attempted = True
        score, breakdown = _score_cleaning(self._dataset["expected"], action.output.strip())
        feedback = (
            f"Data cleaning score: {score:.2f}. "
            f"F1={breakdown.get('f1', 0):.2f}, "
            f"Precision={breakdown.get('precision', 0):.2f}, "
            f"Recall={breakdown.get('recall', 0):.2f}."
        )
        return score, breakdown, feedback, True   # single-step task
