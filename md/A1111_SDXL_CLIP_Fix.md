# Pony / Illustrious CLIP Issues in A1111 — Complete Explanation

> **Base commit:** `f175fb0984aaf4478d3233d402bf5111e790d120` (v1.15)
>
> All "before fix" code quoted in this document reflects the state at that commit; "after fix" code is the change relative to that commit.

## Overview

When using SDXL-derived models (Pony, Illustrious, etc.) in A1111 (Stable Diffusion WebUI), the following issues occurred:

1. **Images became noisy / LoRAs seemed ineffective** (suspected to be a CLIP-L issue)
2. **Model load itself failed with RuntimeError** (CLIP-G issue) ★ this is what actually required a fix

This document explains the technical background, investigation process, and the final fixes applied in full detail.

---

## Background: SDXL text encoder layout

SDXL uses two text encoders in parallel:

| Encoder | Model | Output dim | Role |
|---|---|---|---|
| **CLIP-L** | `openai/clip-vit-large-patch14` (HuggingFace Transformers) | 768 | Text embedding (part 1) |
| **CLIP-G** | `ViT-bigG-14` (open_clip) | 1280 | Text embedding (part 2) + pooled embedding |

Their outputs are concatenated along the channel dimension (768 + 1280 = 2048) and fed into the UNet cross-attention layers.

### Role of CLIP in SDXL models

```
Prompt text
    │
    ├──→ CLIP-L (HuggingFace Transformers) ──→ hidden_states[-2] (768-dim)
    │                                              ↓
    ├──→ CLIP-G (open_clip) ──────────────────→ hidden_states[-2] (1280-dim) + pooled
    │                                              ↓
    │                                    torch.cat([L, G], dim=-1) → 2048-dim
    │                                              ↓
    └──────────────────────────────────────→ UNet cross-attention
```

Pony and Illustrious were **trained assuming ComfyUI CLIP encoding behavior**. If A1111 behaves differently from ComfyUI, train/inference embedding distributions diverge, which can cause broken images, ineffective LoRAs, and similar issues.

---

## Investigation 1: CLIP-L — layer selection and layer_norm differences (no fix needed)

### File: `modules/sd_hijack_clip.py`

### Suspected issue

Pony / Illustrious were trained with ComfyUI behavior:

- Use `hidden_states[-2]` (output one layer before the final layer)
- Do **not** apply `final_layer_norm` (`layer_norm_hidden_state=False`)

A1111's `FrozenCLIPEmbedderForSDXLWithCustomWords.encode_with_transformers` does this:

```python
# Code at f175fb0 (unchanged today)

class FrozenCLIPEmbedderForSDXLWithCustomWords(FrozenCLIPEmbedderWithCustomWords):
    def __init__(self, wrapped, hijack):
        super().__init__(wrapped, hijack)

    def encode_with_transformers(self, tokens):
        outputs = self.wrapped.transformer(input_ids=tokens, output_hidden_states=self.wrapped.layer == "hidden")

        if opts.sdxl_clip_l_skip is True:
            z = outputs.hidden_states[-opts.CLIP_stop_at_last_layers]
        elif self.wrapped.layer == "last":
            z = outputs.last_hidden_state
        else:
            z = outputs.hidden_states[self.wrapped.layer_idx]

        return z
```

Differences vs ComfyUI / Forge do exist in code:

| Item | A1111 | ComfyUI / Forge |
|---|---|---|
| `output_hidden_states` | Only when `layer == "hidden"` | Always `True` |
| Layer used | `last_hidden_state` | `hidden_states[-2]` |
| `final_layer_norm` | Applied implicitly | Not applied |

### Conclusion: no fix needed

With only the CLIP-G `batch_first` fix applied, Pony and Illustrious both produce normal base generations and LoRA effects. CLIP-L layer selection and `final_layer_norm` differences were confirmed not to matter in practice.

> **Lesson:** Even when code differs, do not add fixes based on speculation if behavior is fine in practice.

---

## Issue 2: CLIP-G — RuntimeError on model load (actual fix)

### File: `repositories/generative-models/sgm/modules/encoders/modules.py`

### Root cause

In `open_clip 3.1.0`, the default for `nn.MultiheadAttention`'s `batch_first` changed to `True` (NLD layout). SGM (Stability Generative Models) was written assuming `batch_first=False` (LND) and always ran `permute(1, 0, 2)`.

### What batch_first means

`nn.MultiheadAttention` interprets input tensor layout via `batch_first`:

| `batch_first` | Expected shape | Interpretation |
|---|---|---|
| `False` (old default) | `(L, N, D)` | Length, Batch, Dim |
| `True` (new default 3.1.0+) | `(N, L, D)` | Batch, Length, Dim |

For CLIP text encoders, after token embedding the tensor is `(N=1, L=77, D=1280)` (NLD).

### What happened

SGM's `encode_with_transformer` does:

1. `token_embedding(text)` → `(N, L, D)` = `(1, 77, 1280)` — NLD
2. `x.permute(1, 0, 2)` → `(L, N, D)` = `(77, 1, 1280)` — convert to LND
3. Pass to `text_transformer_forward(x, attn_mask=...)`
4. Each `ResidualAttentionBlock` calls `nn.MultiheadAttention`

**open_clip < 3.1.0 (`batch_first=False`):**
- `(77, 1, 1280)` is read as `(L=77, N=1, D=1280)`
- `tgt_len=77, src_len=77` → attn_mask `(77, 77)` matches ✅

**open_clip >= 3.1.0 (`batch_first=True`):**
- `(77, 1, 1280)` is read as `(N=77, L=1, D=1280)`
- `tgt_len=1, src_len=1` → attn_mask `(77, 77)` mismatches ❌

```
RuntimeError: The shape of the 2D attn_mask is torch.Size([77, 77]), but should be (1, 1).
```

### Code before fix

SGM's `generative-models/sgm/modules/encoders/modules.py` has two OpenCLIP encoder classes:

#### FrozenOpenCLIPEmbedder2 (used as SDXL CLIP-G)

```python
# Before fix

def encode_with_transformer(self, text):
    x = self.model.token_embedding(text)  # [batch_size, n_ctx, d_model] → (N, L, D)
    x = x + self.model.positional_embedding
    x = x.permute(1, 0, 2)  # NLD -> LND  ← unconditional permute
    x = self.text_transformer_forward(x, attn_mask=self.model.attn_mask)
    # ...

def text_transformer_forward(self, x: torch.Tensor, attn_mask=None):
    outputs = {}
    for i, r in enumerate(self.model.transformer.resblocks):
        if i == len(self.model.transformer.resblocks) - 1:
            outputs["penultimate"] = x.permute(1, 0, 2)  # LND -> NLD  ← unconditional permute
        # ...
        x = r(x, attn_mask=attn_mask)
    outputs["last"] = x.permute(1, 0, 2)  # LND -> NLD  ← unconditional permute
    return outputs
```

#### FrozenOpenCLIPEmbedder (used for SD2, etc.)

```python
# Before fix

def encode_with_transformer(self, text):
    x = self.model.token_embedding(text)  # (N, L, D)
    x = x + self.model.positional_embedding
    x = x.permute(1, 0, 2)  # NLD -> LND  ← unconditional permute
    x = self.text_transformer_forward(x, attn_mask=self.model.attn_mask)
    x = x.permute(1, 0, 2)  # LND -> NLD  ← unconditional permute
    x = self.model.ln_final(x)
    return x
```

### Code after fix

Both classes' `encode_with_transformer` and `text_transformer_forward` check `batch_first` and permute only when needed.

#### FrozenOpenCLIPEmbedder2 (after fix)

```python
def encode_with_transformer(self, text):
    x = self.model.token_embedding(text)  # [batch_size, n_ctx, d_model]
    x = x + self.model.positional_embedding
    batch_first = getattr(self.model.transformer, 'batch_first', False)
    if not batch_first:
        x = x.permute(1, 0, 2)  # NLD -> LND
    x = self.text_transformer_forward(x, attn_mask=self.model.attn_mask)
    if self.legacy:
        x = x[self.layer]
        x = self.model.ln_final(x)
        return x
    else:
        o = x["last"]
        o = self.model.ln_final(o)
        pooled = self.pool(o, text)
        x["pooled"] = pooled
        return x

def text_transformer_forward(self, x: torch.Tensor, attn_mask=None):
    batch_first = getattr(self.model.transformer, 'batch_first', False)
    outputs = {}
    for i, r in enumerate(self.model.transformer.resblocks):
        if i == len(self.model.transformer.resblocks) - 1:
            outputs["penultimate"] = x if batch_first else x.permute(1, 0, 2)  # LND -> NLD
        if (
            self.model.transformer.grad_checkpointing
            and not torch.jit.is_scripting()
        ):
            x = checkpoint(r, x, attn_mask)
        else:
            x = r(x, attn_mask=attn_mask)
    outputs["last"] = x if batch_first else x.permute(1, 0, 2)  # LND -> NLD
    return outputs
```

#### FrozenOpenCLIPEmbedder (after fix)

```python
def encode_with_transformer(self, text):
    x = self.model.token_embedding(text)  # [batch_size, n_ctx, d_model]
    x = x + self.model.positional_embedding
    batch_first = getattr(self.model.transformer, 'batch_first', False)
    if not batch_first:
        x = x.permute(1, 0, 2)  # NLD -> LND
    x = self.text_transformer_forward(x, attn_mask=self.model.attn_mask)
    if not batch_first:
        x = x.permute(1, 0, 2)  # LND -> NLD
    x = self.model.ln_final(x)
    return x
```

### Meaning of `getattr(self.model.transformer, 'batch_first', False)`

The third argument `False` makes older open_clip without `batch_first` behave as before:

- **open_clip < 3.1.0** (no attribute or `False`) → permute runs (legacy path)
- **open_clip >= 3.1.0** (`batch_first=True`) → skip permute (stay NLD)

---

## Fix 3: Remove unnecessary workaround

### File: `modules/sd_hijack_open_clip.py`

### Background

Before the CLIP-G `batch_first` fix, `FrozenOpenCLIPEmbedder2WithCustomWords.encode_with_transformers` had a temporary workaround that set `attn_mask` to `None` to avoid the error.

### Before fix

```python
# Before fix — hack that temporarily disables attn_mask

def encode_with_transformers(self, tokens):
    original_attn_mask = None
    if hasattr(self.wrapped.model, 'attn_mask') and self.wrapped.model.attn_mask is not None:
        original_attn_mask = self.wrapped.model.attn_mask
        if original_attn_mask.shape == (77, 77):
            self.wrapped.model.attn_mask = None  # Temporarily disable mask

    try:
        d = self.wrapped.encode_with_transformer(tokens)
        z = d[self.wrapped.layer]
        pooled = d.get("pooled")
        if pooled is not None:
            z.pooled = pooled
        return z
    finally:
        if original_attn_mask is not None:
            self.wrapped.model.attn_mask = original_attn_mask
```

### After fix

```python
# After fix — hack removed, simple call

def encode_with_transformers(self, tokens):
    d = self.wrapped.encode_with_transformer(tokens)
    z = d[self.wrapped.layer]

    pooled = d.get("pooled")
    if pooled is not None:
        z.pooled = pooled

    return z
```

With `batch_first` handled in `modules.py`, disabling `attn_mask` is no longer needed.

---

## Comparison with other stacks: why Forge and ComfyUI were unaffected

### ComfyUI — does not depend on open_clip

ComfyUI has its own CLIP transformer in `comfy/sd1_clip.py` + `comfy/clip_model.py`. It does not use the open_clip package; it uses a HuggingFace Transformers `CLIPTextModel`-based wrapper. It is unaffected by the `batch_first` change.

CLIP-L is also designed to work correctly with `layer_norm_hidden_state=False`.

### Forge Nunchaku — open_clip weights only, different execution path

Forge encodes text via `ClassicTextProcessingEngine` in `backend/text_processing/classic_engine.py`:

- Model structure (weight layout) borrows from open_clip
- Transformer execution calls HuggingFace Transformers `text_encoder.transformer(tokens, output_hidden_states=True)` directly
- It does not go through `open_clip.transformer.ResidualAttentionBlock` → `nn.MultiheadAttention`

```python
# Forge Nunchaku — backend/text_processing/classic_engine.py

def encode_with_transformers(self, tokens):
    outputs = self.text_encoder.transformer(tokens, output_hidden_states=True)
    layer_id = -max(self.clip_skip, self.minimal_clip_skip)
    z = outputs.hidden_states[layer_id]
    if self.final_layer_norm:
        z = self.text_encoder.transformer.final_layer_norm(z)
    # ...
```

- `batch_first` issue: none (HF Transformers API)
- CLIP-L layer choice: can match ComfyUI via `minimal_clip_skip` and `final_layer_norm`

### A1111 — fully depends on open_clip

A1111 goes through SGM's `generative-models` and uses open_clip internals as-is. SGM uses open_clip's `ResidualAttentionBlock` → `nn.MultiheadAttention` path, so open_clip internal changes hit A1111 directly.

| | open_clip model def | open_clip transformer exec | batch_first impact |
|---|---|---|---|
| **ComfyUI** | No | No (custom impl) | None |
| **Forge** | Yes (weight load) | No (via HF Transformers) | None |
| **A1111 (before)** | Yes | Yes (via SGM) | **Direct hit** |
| **A1111 (after)** | Yes | Yes (conditional permute) | **Handled** |

---

## Changed files

| File | Change |
|---|---|
| `repositories/generative-models/sgm/modules/encoders/modules.py` | Add `batch_first` branches in three methods on `FrozenOpenCLIPEmbedder2` and `FrozenOpenCLIPEmbedder` |
| `modules/sd_hijack_open_clip.py` | Remove obsolete temporary `attn_mask` disable hack |

> **Note:** `modules/sd_hijack_clip.py` was investigated but not changed. Only the CLIP-G `batch_first` fix was needed; Pony and Illustrious base generation and LoRA effects were verified normal.

---

## Verification

After the fix, normal behavior was confirmed for:

- ✅ **Pony and Illustrious family base models** — clean images (no noise), LoRAs work
- ✅ RuntimeError on model load resolved
- ✅ Runs on Flash-Attention 2.9.1 + PyTorch 2.12.1+cu132

---

## Environment

| Item | Value |
|---|---|
| A1111 version | v1.15 (commit f175fb0) |
| Python | 3.12.10 |
| PyTorch | 2.12.1+cu132 |
| CUDA | 13.2 |
| Flash-Attention | 2.9.1 |
| open_clip | 3.1.0 |
| Repository | `ussoewwin/A1111-for-Python3.12` (Python 3.12 fork) |

---

## Reference repositories

- Forge Nunchaku: `backend/text_processing/classic_engine.py` — `ClassicTextProcessingEngine`
- ComfyUI: `comfy/sdxl_clip.py`, `comfy/sd1_clip.py`, `comfy/clip_model.py`
- A1111: `modules/sd_hijack_clip.py`, `modules/sd_hijack_open_clip.py`
- SGM: `repositories/generative-models/sgm/modules/encoders/modules.py`

---

## Investigation timeline

### Detours

After noisy images and ineffective LoRAs on Pony / Illustrious, investigation ran for about a year. Early hypotheses were wrong:

- **Wrong v-prediction detection** — Pony / Illustrious do not use v-prediction, but filename-based v-pred auto-detection was added (later reverted)
- **GroupNorm / LayerNorm dtype** — suspected bfloat16 NaN and mixed dtype; tried in-place `self.float()` casts (later reverted)
- **VAE load optimization** — suspected VAE behavior; unrelated

All of this assumed a UNet (denoising) problem; the real issue was text encoders (CLIP).

### Breakthrough

Strict comparison of Forge and ComfyUI code revealed:

1. ComfyUI does not depend on open_clip (custom implementation)
2. Forge uses open_clip model definitions only; execution goes through HuggingFace Transformers
3. Only A1111 fully depends on open_clip internals via SGM

That structural gap meant open_clip 3.1.0's `batch_first` default change hit A1111 directly. Adding conditional `batch_first` handling fixed it.

### Lessons

- **Read the code, do not guess** — comparing Forge and ComfyUI was the breakthrough
- **Code difference ≠ must fix** — CLIP-L differed from ComfyUI but needed no change once verified
- **Do not infer cause from symptoms** — noise suggested v-prediction or dtype; the cause was CLIP-G `batch_first`

---

## Summary

The core issue was **A1111's legacy dependency on SGM + open_clip without keeping up with open_clip version changes**.

The only applied fix was CLIP-G `batch_first` handling. CLIP-L was investigated and left unchanged. Forge and ComfyUI use separate text-encoding paths and were unaffected.

The fix uses `getattr` with a `False` fallback so both old and new open_clip work, preserving backward compatibility.
