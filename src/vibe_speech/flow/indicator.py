from __future__ import annotations

import sys

_COLOR_GREEN = "\033[92m"
_COLOR_DIM = "\033[90m"
_COLOR_RESET = "\033[0m"
_LABEL_WIDTH = max(len("LISTENING: ON"), len("LISTENING: OFF"))


def show_listening_banner(listening: bool) -> None:
    """Print a clear banner in the terminal indicating listening state."""
    color = _COLOR_GREEN if listening else _COLOR_DIM
    label = "LISTENING: ON" if listening else "LISTENING: OFF"
    inner_width = _LABEL_WIDTH
    border = "+" + "-" * (inner_width + 2) + "+"
    line = f"| {label:<{inner_width}} |"
    sys.stdout.write("\n" + color + border + "\n" + line + "\n" + border + _COLOR_RESET + "\n")
    sys.stdout.flush()
