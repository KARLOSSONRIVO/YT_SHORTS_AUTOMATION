DEFAULT_RENDER_FONT_FAMILY = "DejaVu Sans"

# These families are used by older projects or the browser form, but are not
# installed in the Python worker image. Falling back here keeps the ASS output
# bold and predictable instead of letting libass choose a thin substitute.
UNAVAILABLE_RENDER_FONT_FAMILIES = frozenset(
    {
        "arial",
        "arial black",
        "bebas neue",
        "impact",
        "inter",
        "montserrat",
        "montserrat extrabold",
    }
)


def resolve_render_font_family(font_family: str | None) -> str:
    requested = (font_family or "").strip()
    if not requested or requested.casefold() in UNAVAILABLE_RENDER_FONT_FAMILIES:
        return DEFAULT_RENDER_FONT_FAMILY
    return requested
