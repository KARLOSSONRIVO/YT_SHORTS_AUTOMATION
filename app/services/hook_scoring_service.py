import re


class HookScoringService:
    HOOK_PATTERNS = [
        r"\bhere('?s| is) why\b",
        r"\bi tried\b",
        r"\bthe biggest mistake\b",
        r"\bno one tells you\b",
        r"\bthis changed everything\b",
        r"\btop \d+\b",
        r"\bsecret\b",
        r"\bwarning\b",
        r"\bmistake\b",
        r"\bhow to\b",
    ]

    def score(self, text: str) -> float:
        lowered = text.lower()
        score = 0.0
        for pattern in self.HOOK_PATTERNS:
            if re.search(pattern, lowered):
                score += 0.15
        if "?" in text:
            score += 0.05
        if any(char.isdigit() for char in text):
            score += 0.05
        return min(score, 1.0)
