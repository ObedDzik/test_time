"""EATA baseline for test-time adaptation."""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import (
    check_adaptation_ready,
    collect_adaptation_params,
    configure_model_for_adaptation,
    copy_model_and_optimizer,
    extract_logits,
    softmax_entropy,
)


def _update_model_probs(current_probs, new_probs):
    if current_probs is None:
        if new_probs.numel() == 0:
            return None
        with torch.no_grad():
            return new_probs.mean(0)

    if new_probs.numel() == 0:
        return current_probs
    with torch.no_grad():
        return 0.9 * current_probs + 0.1 * new_probs.mean(0)


class EATA(nn.Module):
    """Efficient anti-forgetting test-time adaptation."""

    def __init__(
        self,
        model,
        optimizer,
        fishers: Optional[Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = None,
        fisher_alpha: float = 2000.0,
        steps: int = 1,
        episodic: bool = False,
        e_margin: float = math.log(1000) / 2 - 1,
        d_margin: float = 0.05,
    ):
        super().__init__()
        if steps <= 0:
            raise ValueError("EATA requires steps >= 1.")

        self.model = model
        self.optimizer = optimizer
        self.fishers = fishers
        self.fisher_alpha = fisher_alpha
        self.steps = steps
        self.episodic = episodic
        self.e_margin = e_margin
        self.d_margin = d_margin

        self.current_model_probs = None
        self.model_state, self.optimizer_state = copy_model_and_optimizer(model, optimizer)

    def forward(self, x):
        if self.episodic:
            self.reset()

        model_output = None
        for _ in range(self.steps):
            model_output, updated_probs = forward_and_adapt_eata(
                x=x,
                model=self.model,
                optimizer=self.optimizer,
                fishers=self.fishers,
                e_margin=self.e_margin,
                current_model_probs=self.current_model_probs,
                fisher_alpha=self.fisher_alpha,
                d_margin=self.d_margin,
            )
            self.current_model_probs = updated_probs
        return model_output

    def reset(self):
        if self.model_state is None:
            raise RuntimeError("Cannot reset EATA without saved model state.")
        self.model.load_state_dict(self.model_state, strict=True)
        self.current_model_probs = None

    def forward_no_adapt(self, x):
        return self.model(x)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


@torch.enable_grad()
def forward_and_adapt_eata(
    x,
    model,
    optimizer,
    fishers,
    e_margin,
    current_model_probs,
    fisher_alpha=2000.0,
    d_margin=0.05,
):
    model_output = model(x)
    logits = extract_logits(model_output)
    entropies = softmax_entropy(logits)

    # Filter unreliable samples (high entropy).
    filter_ids_1 = torch.where(entropies < e_margin)[0]
    selected_ent = entropies[filter_ids_1]
    selected_probs = logits.softmax(1)[filter_ids_1] if filter_ids_1.numel() > 0 else logits.new_zeros((0, logits.shape[1]))

    # Filter redundant samples by cosine similarity to moving average probs.
    if current_model_probs is not None and filter_ids_1.numel() > 0:
        cosine_sim = F.cosine_similarity(current_model_probs.unsqueeze(0), selected_probs, dim=1)
        filter_ids_2 = torch.where(torch.abs(cosine_sim) < d_margin)[0]
        selected_ent = selected_ent[filter_ids_2]
        selected_probs = selected_probs[filter_ids_2]

    updated_probs = _update_model_probs(current_model_probs, selected_probs)
    optimizer.zero_grad()

    if selected_ent.numel() > 0:
        coeff = 1.0 / torch.exp(selected_ent.detach() - e_margin)
        loss = (selected_ent * coeff).mean(0)

        if fishers is not None:
            ewc_loss = 0.0
            for name, param in model.named_parameters():
                if name in fishers:
                    fisher_matrix, param_ref = fishers[name]
                    ewc_loss = ewc_loss + fisher_alpha * (fisher_matrix * (param - param_ref) ** 2).sum()
            loss = loss + ewc_loss

        if not torch.isnan(loss):
            loss.backward()
            optimizer.step()

    optimizer.zero_grad()
    return model_output, updated_probs


def compute_fishers(
    model: nn.Module,
    fisher_loader,
    device: torch.device,
    num_samples: int | None = None,
):
    """Estimate Fisher information from a loader using predicted labels."""
    fishers = {}
    loss_fn = nn.CrossEntropyLoss().to(device)
    total_samples = 0
    num_iters = 0

    for images, _ in fisher_loader:
        images = images.to(device)
        batch_size = images.shape[0]
        if num_samples is not None and total_samples >= num_samples:
            break
        total_samples += batch_size
        num_iters += 1

        model_output = model(images)
        logits = extract_logits(model_output)
        pseudo_targets = logits.argmax(dim=1)
        loss = loss_fn(logits, pseudo_targets)
        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            fisher_now = param.grad.detach().clone() ** 2
            if name in fishers:
                fishers[name][0] = fishers[name][0] + fisher_now
            else:
                fishers[name] = [fisher_now, param.detach().clone()]

        model.zero_grad(set_to_none=True)

    if num_iters > 0:
        for name in fishers:
            fishers[name][0] = fishers[name][0] / float(num_iters)

    return fishers


def setup_eata(
    model: nn.Module,
    lr: float = 1e-3,
    steps: int = 1,
    episodic: bool = False,
    architecture: str = "auto",
    layer_selection: str = "auto",
    fishers=None,
    fisher_alpha: float = 2000.0,
    e_margin: float = math.log(1000) / 2 - 1,
    d_margin: float = 0.05,
    optimizer_cls=torch.optim.Adam,
    optimizer_kwargs: dict | None = None,
) -> Tuple[EATA, list[str], str]:
    """Configure a model and wrap it with EATA."""
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
    eata = EATA(
        model=model,
        optimizer=optimizer,
        fishers=fishers,
        fisher_alpha=fisher_alpha,
        steps=steps,
        episodic=episodic,
        e_margin=e_margin,
        d_margin=d_margin,
    )
    return eata, selected_names, arch
