"""Task 3 – Code Review (Hard)"""

from __future__ import annotations
from typing import Dict, List, Tuple, Any
import ast
import random


BUGGY_SNIPPETS = [
    {
        "description": "Fix indentation, off-by-one error, and missing return.",
        "code": (
            "def find_max(lst):\n"
            "for i in range(len(lst)):\n"
            "    if lst[i] > lst[i+1]:\n"
            "        max_val = lst[i]\n"
        ),
        "issues": [
            "Missing indentation for the for-loop body",
            "IndexError: lst[i+1] goes out of range on the last element",
            "Missing return statement",
        ],
        "fixed": (
            "def find_max(lst):\n"
            "    max_val = lst[0]\n"
            "    for i in range(len(lst)):\n"
            "        if lst[i] > max_val:\n"
            "            max_val = lst[i]\n"
            "    return max_val\n"
        ),
        "keywords": ["indentation", "range", "return", "max_val", "index"],
    },
    {
        "description": "Fix recursion base case, variable shadowing, and wrong operator.",
        "code": (
            "def factorial(n):\n"
            "    if n = 0:\n"
            "        return 1\n"
            "    return n + factorial(n-1)\n"
        ),
        "issues": [
            "SyntaxError: assignment (=) used instead of comparison (==) in if condition",
            "Wrong operator: should be n * factorial(n-1), not n + factorial(n-1)",
        ],
        "fixed": (
            "def factorial(n):\n"
            "    if n == 0:\n"
            "        return 1\n"
            "    return n * factorial(n - 1)\n"
        ),
        "keywords": ["==", "multiply", "*", "factorial", "base case"],
    },
    {
        "description": "Fix list mutation bug and incorrect string concatenation in loop.",
        "code": (
            "def double_and_join(items):\n"
            "    for item in items:\n"
            "        items.append(item * 2)\n"
            "    result = ''\n"
            "    for item in items:\n"
            "        result = result + item\n"
            "    return result\n"
        ),
        "issues": [
            "Mutating a list while iterating over it causes an infinite loop",
            "Type error: appending integers (item*2) then concatenating to a string without str()",
        ],
        "fixed": (
            "def double_and_join(items):\n"
            "    doubled = [item * 2 for item in items]\n"
            "    result = ''\n"
            "    for item in doubled:\n"
            "        result = result + str(item)\n"
            "    return result\n"
        ),
        "keywords": ["doubled", "copy", "str(", "mutation", "infinite"],
    },
    {
        "description": "Fix missing self parameter, wrong method name, and division by zero.",
        "code": (
            "class Stats:\n"
            "    def __init__(data):\n"
            "        self.data = data\n\n"
            "    def average():\n"
            "        total = sum(self.data)\n"
            "        return total / len(self.data)\n"
        ),
        "issues": [
            "Missing 'self' parameter in __init__",
            "Missing 'self' parameter in average()",
            "No guard against empty list (ZeroDivisionError)",
        ],
        "fixed": (
            "class Stats:\n"
            "    def __init__(self, data):\n"
            "        self.data = data\n\n"
            "    def average(self):\n"
            "        if not self.data:\n"
            "            return 0\n"
            "        return sum(self.data) / len(self.data)\n"
        ),
        "keywords": ["self", "average(self)", "__init__(self", "empty", "zero"],
    },
]


def _syntax_valid(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def _keyword_score(fixed_code: str, keywords: List[str]) -> Tuple[float, int]:
    lower = fixed_code.lower()
    matched = sum(1 for kw in keywords if kw.lower() in lower)
    return matched / len(keywords) if keywords else 0.0, matched


class CodeReviewTask:
    """
    Hard task: the agent receives buggy Python code + a description of the problem
    and must return corrected code with comments explaining the fixes.
    """

    def __init__(self):
        self._snippet: dict = {}
        self._attempted: bool = False

    def reset(self):
        self._snippet = random.choice(BUGGY_SNIPPETS)
        self._attempted = False

    def get_input(self) -> str:
        issues_text = "\n".join(f"  - {iss}" for iss in self._snippet["issues"])
        return (
            f"Code Review Task\n"
            f"Description: {self._snippet['description']}\n\n"
            f"Known issues to fix:\n{issues_text}\n\n"
            f"Buggy code:\n```python\n{self._snippet['code']}\n```\n\n"
            f"Return ONLY the corrected Python code (no markdown fences, no explanation outside comments)."
        )

    def evaluate(self, action) -> Tuple[float, Dict[str, float], str, bool]:
        self._attempted = True
        submitted = action.output.strip()

        # Strip markdown fences if present
        if submitted.startswith("```"):
            lines = submitted.split("\n")
            submitted = "\n".join(
                l for l in lines if not l.strip().startswith("```")
            )

        syntax_ok = _syntax_valid(submitted)
        syntax_score = 1.0 if syntax_ok else 0.0

        kw_score, matched = _keyword_score(submitted, self._snippet["keywords"])

        # Partial credit for each tracked keyword / fix
        final = round(0.4 * syntax_score + 0.6 * kw_score, 4)

        breakdown = {
            "syntax_valid": syntax_score,
            "keyword_coverage": kw_score,
            "keywords_matched": matched,
            "total_keywords": len(self._snippet["keywords"]),
        }
        feedback = (
            f"Syntax {'[OK] valid' if syntax_ok else '[X] invalid'}. "
            f"Fix coverage: {matched}/{len(self._snippet['keywords'])} keywords matched. "
            f"Score: {final:.2f}."
        )
        return final, breakdown, feedback, True
