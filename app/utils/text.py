import re


def compact_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def tokenize_text(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9']+", text.lower())


def summarize_excerpt(text: str, max_words: int = 8) -> str:
    clean = compact_whitespace(text)
    if not clean:
        return "Candidate Clip"

    sentence = re.split(r"(?<=[.!?])\s+", clean, maxsplit=1)[0].rstrip(".!?").strip()
    words = tokenize_text(sentence or clean)
    if not words:
        return "Candidate Clip"
    return " ".join(word[:1].upper() + word[1:].lower() for word in words[:max_words])


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
