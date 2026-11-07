"""GITAM brand colour palette (Brand Guidelines 2024-25, v1.0) used across the app.

Primary   : GITAM Green, Antique White
Secondary : Dark Green, Light Green, Antique Gold, Beige
Tertiary  : Kind Coral, Mellow Yellow, Wide Blue  (keep primary:tertiary <= 5:1)
"""

GITAM_GREEN = "#007367"     # Pantone 328 C
ANTIQUE_WHITE = "#F4E4C9"   # Pantone 7506 C 70%
DARK_GREEN = "#004740"      # Pantone 330 C
LIGHT_GREEN = "#8BCBB7"     # Pantone 570 C
ANTIQUE_GOLD = "#A58255"    # Pantone 2318 C
BEIGE = "#CCBA8D"           # Pantone 7502 C
KIND_CORAL = "#DD736E"      # Pantone 2345 C
MELLOW_YELLOW = "#E0B541"   # Pantone 7408 C
WIDE_BLUE = "#5E95CD"       # Pantone 2171 C

# Light tint of Antique White used for page surfaces
CREAM = "#FBF6EC"
TEXT = "#16332F"

PALETTE = {
    "GITAM Green": GITAM_GREEN, "Antique White": ANTIQUE_WHITE, "Dark Green": DARK_GREEN,
    "Light Green": LIGHT_GREEN, "Antique Gold": ANTIQUE_GOLD, "Beige": BEIGE,
    "Kind Coral": KIND_CORAL, "Mellow Yellow": MELLOW_YELLOW, "Wide Blue": WIDE_BLUE,
}

MODEL_COLORS = {
    "ID3": GITAM_GREEN,
    "C4.5": ANTIQUE_GOLD,
    "sklearn CART (entropy)": WIDE_BLUE,
}

# Order used to colour classes in trees and charts (primary/secondary first)
CLASS_COLORS = [GITAM_GREEN, ANTIQUE_GOLD, WIDE_BLUE, KIND_CORAL, MELLOW_YELLOW,
                LIGHT_GREEN, BEIGE, DARK_GREEN]


def model_color(name: str) -> str:
    return MODEL_COLORS.get(name, DARK_GREEN)


def hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def blend_with_white(h: str, alpha: float) -> str:
    """Colour ``h`` at opacity ``alpha`` over white, as an opaque hex string."""
    r, g, b = hex_to_rgb(h)
    mix = [round(255 - (255 - c) * alpha) for c in (r, g, b)]
    return "#{:02X}{:02X}{:02X}".format(*mix)


def readable_text(bg_hex: str) -> str:
    r, g, b = (c / 255 for c in hex_to_rgb(bg_hex))
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "#FFFFFF" if lum < 0.5 else TEXT
