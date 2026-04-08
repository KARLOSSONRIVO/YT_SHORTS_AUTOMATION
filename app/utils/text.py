import re


def compact_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def tokenize_text(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9']+", text.lower())


def summarize_excerpt(text: str, max_words: int = 8) -> str:
    words = tokenize_text(text)
    if not words:
        return "Candidate clip"
    return " ".join(words[:max_words]).title()


def split_caption_lines(text: str, max_chars_per_line: int, max_lines: int) -> str:
    words = compact_whitespace(text).split(" ")
    lines: list[str] = []
    current = ""

    for word in words:
        tentative = word if not current else f"{current} {word}"
        if len(tentative) <= max_chars_per_line:
            current = tentative
            continue

        if current:
            lines.append(current)
        current = word
        if len(lines) == max_lines - 1:
            break

    if current and len(lines) < max_lines:
        lines.append(current)

    return "\n".join(lines[:max_lines])
