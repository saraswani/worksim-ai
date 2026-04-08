"""Task 1 – Email Triage (Easy)"""

from __future__ import annotations
from typing import Dict, List, Tuple, Any
import random


EMAILS = [
    {
        "subject": "URGENT: Production server is down!",
        "body": "Our main database server has crashed. Customers cannot access the app. Need immediate action!",
        "label": "urgent",
    },
    {
        "subject": "Team lunch this Friday?",
        "body": "Hey, just checking if everyone is free for lunch on Friday at 1 PM. Please reply!",
        "label": "normal",
    },
    {
        "subject": "You've won a $1000 gift card! Click now!",
        "body": "Congratulations! You've been selected. Click the link to claim your prize immediately.",
        "label": "spam",
    },
    {
        "subject": "Security breach detected in staging environment",
        "body": "Automated scanner found a potential SQL injection vulnerability. Please review and patch ASAP.",
        "label": "urgent",
    },
    {
        "subject": "Monthly newsletter – April edition",
        "body": "Check out what's happening at the company this month: new hires, product updates, and more.",
        "label": "normal",
    },
    {
        "subject": "FREE iPhone 15 – limited offer!",
        "body": "Act now to receive your complimentary smartphone. No purchase necessary. Offer expires soon!",
        "label": "spam",
    },
    {
        "subject": "Critical bug in payment module",
        "body": "Users are reporting failed transactions. Revenue impact estimated at $50k/hour. Fix required NOW.",
        "label": "urgent",
    },
    {
        "subject": "Reminder: Submit your timesheet by EOD",
        "body": "Please submit your weekly timesheet before end of day Friday. HR needs this for payroll.",
        "label": "normal",
    },
    {
        "subject": "Increase your followers 1000x overnight!",
        "body": "Buy real followers from our trusted service. 100% safe, money-back guarantee.",
        "label": "spam",
    },
    {
        "subject": "CEO approval required – Q2 budget revision",
        "body": "Board meeting is tomorrow morning. We need the Q2 budget finalised and signed off tonight.",
        "label": "urgent",
    },
]

VALID_LABELS = {"urgent", "normal", "spam"}


class EmailTriageTask:
    """
    Easy task: classify emails as urgent / normal / spam.
    Agent gets one email per step; episode ends when all emails are processed.
    """

    def __init__(self):
        self._emails: List[dict] = []
        self._index: int = 0

    def reset(self):
        self._emails = random.sample(EMAILS, k=5)   # 5 emails per episode
        self._index = 0

    def get_input(self) -> str:
        if self._index >= len(self._emails):
            return "All emails processed."
        e = self._emails[self._index]
        return (
            f"Email {self._index + 1}/{len(self._emails)}\n"
            f"Subject: {e['subject']}\n"
            f"Body: {e['body']}\n\n"
            f"Classify this email as: urgent | normal | spam"
        )

    def evaluate(self, action) -> Tuple[float, Dict[str, float], str, bool]:
        if self._index >= len(self._emails):
            return 0.0, {}, "No more emails.", True

        expected = self._emails[self._index]["label"]
        given = action.output.strip().lower()

        correct = given == expected
        reward = 1.0 if correct else -0.5
        feedback = f"Correct! '{given}' ✓" if correct else f"Wrong. Expected '{expected}', got '{given}'."

        self._index += 1
        done = self._index >= len(self._emails)

        return reward, {"classification_score": 1.0 if correct else 0.0}, feedback, done
