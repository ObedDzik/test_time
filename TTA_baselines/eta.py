"""ETA baseline for test-time adaptation.

ETA is EATA without Fisher regularization.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn

from .common import (
    check_adaptation_ready,
    collect_adaptation_params,
    configure_model_for_adaptation,
)
from .eata import EATA


class ETA(EATA):
    """Efficient test-time adaptation (EATA without regularization)."""

    def __init__(
        self,
        model,
        optimizer,
        steps: int = 1,
        episodic: bool = False,
        e_margin: float = math.log(1000) / 2 - 1,
        d_margin: float = 0.05,
    ):
        super().__init__(
            model=model,
            optimizer=optimizer,
            fishers=None,
            fisher_alpha=0.0,
            steps=steps,
            episodic=episodic,
            e_margin=e_margin,
            d_margin=d_margin,
        )


def setup_eta(
    model: nn.Module,
    lr: float = 1e-3,
    steps: int = 1,
    episodic: bool = False,
    architecture: str = "auto",
    layer_selection: str = "auto",
    e_margin: float = math.log(1000) / 2 - 1,
    d_margin: float = 0.05,
    optimizer_cls=torch.optim.Adam,
    optimizer_kwargs: dict | None = None,
) -> Tuple[ETA, list[str], str]:
    """Configure a model and wrap it with ETA (EATA without regularization)."""
    optimizer_kwargs = optimizer_kwargs or {}

    model, selected_names, arch = configure_model_for_adaptation(
        model=model,
        architecture=architecture,
        layer_selection=layer_selection,
    )
    check_adaptation_ready(model)

    params, _, _ = collect_adaptation_params(
        model=model,
        architecture=arch,
        layer_selection=layer_selection,
    )
    optimizer = optimizer_cls(params, lr=lr, **optimizer_kwargs)
    eta = ETA(
        model=model,
        optimizer=optimizer,
        steps=steps,
        episodic=episodic,
        e_margin=e_margin,
        d_margin=d_margin,
    )
    return eta, selected_names, arch