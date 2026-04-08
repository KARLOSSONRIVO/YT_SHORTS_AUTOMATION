from app.utils.text import tokenize_text


class SpeechIntensityService:
    def score(self, text: str, duration: float) -> float:
        words_per_second = len(tokenize_text(text)) / max(duration, 0.1)
        if words_per_second >= 3.0:
            return 1.0
        if words_per_second >= 2.2:
            return 0.8
        if words_per_second >= 1.6:
            return 0.6
        if words_per_second >= 1.0:
            return 0.4
        return 0.2
