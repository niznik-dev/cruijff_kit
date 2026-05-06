"""Extract training loss from torchtune SLURM output.

Provides the canonical regex and helpers for parsing loss lines
from custom torchtune recipe stdout (epoch|step|Loss: value format).
"""

import re

LOSS_PATTERN = re.compile(r"(\d+)\|(\d+)\|Loss: ([0-9.eE+-]+)")


def extract_losses(text: str) -> list[tuple[int, int, float]]:
    """Extract all (epoch, step, loss) tuples from torchtune output."""
    return [
        (int(epoch), int(step), float(loss))
        for epoch, step, loss in LOSS_PATTERN.findall(text)
    ]


def final_loss(text: str) -> tuple[int, int, float] | None:
    """Return the last (epoch, step, loss) tuple, or None if no matches."""
    losses = extract_losses(text)
    return losses[-1] if losses else None
