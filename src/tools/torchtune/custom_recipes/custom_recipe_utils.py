# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Utility functions for custom recipes.

!--- cruijff_kit patch ---!
"""


def validate_epochs_to_save(epochs_to_save, total_epochs: int) -> list[int]:
    """Validate and normalize the cruijff_kit `epochs_to_save` config value.

    Without this guard, a misformatted value (out-of-range index, empty list,
    wrong type) silently produces a run with zero checkpoints — the recipe just
    logs "Skipping checkpoint save" every epoch.

    Accepts the string 'all', or any iterable of ints. Returns a Python list of
    valid epoch indices. Raises ValueError on bad input.
    """
    if isinstance(epochs_to_save, str):
        if epochs_to_save == "all":
            return list(range(total_epochs))
        raise ValueError(
            f"epochs_to_save string value must be 'all', got: {epochs_to_save!r}"
        )

    try:
        epochs_list = list(epochs_to_save)
    except TypeError:
        raise ValueError(
            f"epochs_to_save must be a list of ints or 'all', got "
            f"{type(epochs_to_save).__name__}: {epochs_to_save!r}"
        )

    if not epochs_list:
        raise ValueError(
            "epochs_to_save resolved to an empty list — no checkpoints would be saved. "
            "Set epochs_to_save: 'all' or provide at least one valid epoch index."
        )

    bad = [
        e
        for e in epochs_list
        if isinstance(e, bool) or not isinstance(e, int) or not (0 <= e < total_epochs)
    ]
    if bad:
        raise ValueError(
            f"epochs_to_save contains values outside [0, {total_epochs}) or of wrong type: "
            f"{bad}. total_epochs={total_epochs}."
        )

    return epochs_list
