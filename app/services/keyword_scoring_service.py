from app.utils.text import tokenize_text


class KeywordScoringService:
    def score(self, text: str, target_keywords: list[str]) -> float:
        if not target_keywords:
            return 0.0

        tokens = set(tokenize_text(text))
        normalized_keywords = {keyword.lower().strip() for keyword in target_keywords if keyword.strip()}
        if not normalized_keywords:
            return 0.0

        matches = len(tokens.intersection(normalized_keywords))
        return min(matches / max(len(normalized_keywords), 1), 1.0)
