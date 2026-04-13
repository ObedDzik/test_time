# TTA Baselines (Tent, ETA, EATA, SAR, MEMO, RoTTA, PeTTA, ROID, RMT)

This folder contains reusable test-time adaptation (TTA) baselines for image classification:

- **Tent**
- **ETA**
- **EATA**
- **SAR**
- **MEMO**
- **RoTTA**
- **PeTTA**
- **ROID**
- **RMT**

The implementations are designed to work with:

- Standard classifiers that return `logits`
- Models that return tuples/lists where `logits` are at index `0`

---

## 1) Quick Integration

Import setup helpers:

```python
from test_time.TTA_baselines import (
    setup_tent,
    setup_eta,
    setup_eata,
    setup_sar,
    setup_memo,
    setup_rotta,
    setup_petta,
    compute_source_prototypes,
    setup_roid,
    setup_rmt,
    compute_rmt_source_prototypes,
)
```

Wrap your model with a TTA method:

```python
tta_model, adapted_param_names, arch = setup_tent(
    model,
    lr=1e-3,
    steps=1,
    episodic=False,
    architecture="auto",      # auto | cnn | transformer | hybrid
    layer_selection="auto",   # auto | all | late
)
```

Use it during inference:

```python
# Online adaptation on each forward pass
output = tta_model(images)

# Optional: inference without adaptation
output_no_adapt = tta_model.forward_no_adapt(images)
```

---

## 2) Method-Specific Setup

### Tent

```python
from test_time.TTA_baselines import setup_tent

tent_model, names, arch = setup_tent(
    model,
    lr=1e-3,
    steps=1,
    episodic=False,
)
```

### ETA (EATA without regularization)

```python
from test_time.TTA_baselines import setup_eta

eta_model, names, arch = setup_eta(
    model,
    lr=1e-3,
    steps=1,
    episodic=False,
    e_margin=2.45,
    d_margin=0.05,
)
```

### EATA

```python
from test_time.TTA_baselines import setup_eata, compute_fishers

# Optional Fisher regularizer (recommended for anti-forgetting)
fishers = compute_fishers(model, fisher_loader, device)

eata_model, names, arch = setup_eata(
    model,
    lr=1e-3,
    steps=1,
    episodic=False,
    fishers=fishers,
    fisher_alpha=2000.0,
    e_margin=2.45,   # adjust to num classes / confidence
    d_margin=0.05,
)
```

ETA and EATA share the same selective adaptation mechanism. EATA adds Fisher regularization to reduce forgetting on clean/ID data.

### SAR

```python
from test_time.TTA_baselines import setup_sar

sar_model, names, arch = setup_sar(
    model,
    lr=1e-3,
    steps=1,
    episodic=False,
    margin_e0=0.4 * 6.9,  # default follows SAR-style setup for ~1000 classes
    reset_constant_em=0.2,
    rho=0.05,
    sam_adaptive=False,
)
```

### MEMO

```python
from test_time.TTA_baselines import setup_memo

memo_model, names, arch = setup_memo(
    model,
    lr=2.5e-4,
    steps=1,
    batch_size=32,           # number of augmented views per sample
    episodic=True,           # typical MEMO usage
    augmentation_type="augmix",   # augmix | standard
    image_size=224,
)
```

### RoTTA

```python
from test_time.TTA_baselines import setup_rotta

rotta_model, names, arch = setup_rotta(
    model,
    lr=1e-3,
    steps=1,
    episodic=False,
    memory_size=64,
    update_frequency=64,
    nu=1e-3,
    lambda_t=1.0,
    lambda_u=1.0,
    bn_momentum=0.05,
)
```

RoTTA uses an EMA teacher, robust BN replacement, and a class-sensitive memory bank. It requires BatchNorm layers in the model.

### PeTTA

```python
from test_time.TTA_baselines import setup_petta

petta_model, names, arch = setup_petta(
    model,
    lr=1e-3,
    steps=1,
    episodic=False,
    memory_size=64,
    alpha_0=1e-3,
    lambda_0=10.0,
    al_wgt=1.0,
    regularizer="cosine",   # cosine | l2 | none
    loss_func="sce",        # sce | ce
    adaptive_lambda=True,
    adaptive_alpha=True,
)
```

If you can provide source prototypes (closer to the original PeTTA setup), precompute them and pass to `setup_petta`:

```python
from test_time.TTA_baselines import compute_source_prototypes, setup_petta

src_mean, src_cov = compute_source_prototypes(
    model=model,
    source_loader=source_loader,   # yields (images, labels, ...)
    num_classes=num_classes,
    feature_extractor=feature_extractor,   # optional
    classifier_head=classifier_head,       # optional
    device=device,
)

petta_model, names, arch = setup_petta(
    model,
    source_prototypes=src_mean,
    source_covariances=src_cov,
    num_classes=num_classes,
)
```

For encoder/classifier-separated models, pass:
- `feature_extractor(model, x) -> features`
- `classifier_head(model, features) -> logits`

If these callbacks are not provided, PeTTA falls back to using logits as features.

### ROID

```python
from test_time.TTA_baselines import setup_roid

roid_model, names, arch = setup_roid(
    model,
    lr=2.5e-4,
    steps=1,
    episodic=False,
    use_weighting=True,
    use_prior_correction=True,
    use_consistency=True,
    momentum_src=0.99,
    momentum_probs=0.9,
    temperature=1.0 / 3.0,
)
```

ROID applies SLR loss with optional diversity/certainty weighting, consistency regularization, and source-weight ensembling.

### RMT

```python
from test_time.TTA_baselines import setup_rmt

rmt_model, names, arch = setup_rmt(
    model,
    source_loader=source_loader,  # yields (images, labels, ...)
    num_classes=num_classes,
    lr=1e-2,
    steps=1,
    lambda_ce_src=1.0,
    lambda_ce_trg=1.0,
    lambda_cont=1.0,
    teacher_momentum=0.999,
    temperature=0.1,
    projection_dim=128,
    warmup_samples=50000,
)
```

You can precompute source prototypes and pass them directly:

```python
from test_time.TTA_baselines import compute_rmt_source_prototypes, setup_rmt

src_proto = compute_rmt_source_prototypes(
    model=model,
    source_loader=source_loader,
    num_classes=num_classes,
    feature_extractor=feature_extractor,   # optional
    classifier_head=classifier_head,       # optional
    device=device,
)

rmt_model, names, arch = setup_rmt(
    model,
    source_loader=source_loader,
    source_prototypes=src_proto,
    num_classes=num_classes,
    feature_extractor=feature_extractor,
    classifier_head=classifier_head,
)
```

For encoder/classifier-separated models, pass:
- `feature_extractor(model, x) -> features`
- `classifier_head(model, features) -> logits`

If callbacks are not provided, RMT falls back to using logits as features.

---

## 3) Architecture-Aware Layer Adaptation

The baselines automatically adapt normalization parameters according to architecture:

- **CNN**: BN/GN/IN affine parameters
- **Transformer**: LayerNorm affine parameters
- **Hybrid**: both sets

Control this behavior with:

- `architecture`: `auto`, `cnn`, `transformer`, `hybrid`
- `layer_selection`: `all`, `late`, `auto`

`layer_selection="auto"` uses:

- `late` for transformer-like models (usually more stable)
- `all` for CNN-like models

---

## 4) Output Handling

These wrappers preserve model output format.

- If your model returns `logits`, wrapper returns `logits`
- If your model returns `(logits, extra1, extra2, ...)`, wrapper returns the full tuple

Internally, adaptation losses always use the extracted `logits`.

---

## 5) Minimal End-to-End Pattern

```python
model.eval()
model.to(device)

tta_model, _, _ = setup_tent(model, lr=1e-3, steps=1)
tta_model.to(device)

for images, _ in test_loader:
    images = images.to(device)
    output = tta_model(images)   # adapts online + predicts
```

---

## 6) Notes

- Use small learning rates (`1e-4` to `1e-3`) for stability.
- `steps > 1` can help adaptation but increases compute.
- `episodic=True` resets model each batch/episode; `False` enables continual adaptation.
- ETA is EATA without Fisher regularization (`setup_eta`), useful as the no-regularization baseline from the paper.
- MEMO typically adapts one sample at a time (`batch size = 1` input to wrapper).
- RoTTA (`setup_rotta`) is designed for dynamic streams and uses memory-bank updates every `update_frequency` instances.
- PeTTA (`setup_petta`) combines EMA teacher updates, memory replay, anchor loss, and adaptive regularization scheduling.
- ROID (`setup_roid`) relies on weighted soft-likelihood-ratio optimization with optional prior correction.
- RMT (`setup_rmt`) is source-aware and typically expects a labeled source loader for replay/prototype extraction.
- For fair baseline comparisons, keep augmentations and dataloaders identical across methods.
