import random

def get_task():
    tasks = [
        {
            "type": "email_triage",
            "input": "Server is down! Fix immediately!",
            "expected": "urgent"
        },
        {
            "type": "data_cleaning",
            "input": "John,25\n,30\nAlice,twenty",
            "expected": "John,25\nAlice,20"
        },
        {
            "type": "code_review",
            "input": "def add(a,b):\nreturn a+b",
            "expected": "def add(a,b):\n    return a+b"
        }
    ]

    return random.choice(tasks)