# RES4LYF Hybrid Sampler — `data_prev_` IndexError Fix

**Repository:** `ussoewwin/A1111-for-Python3.12`  
**Fix commit:** `302589c0`  
**Modified file:** `modules/a1111_res4lyf_shim.py`  
**Scope:** A1111-side shim only. **`modules/RES4LYF/` is not edited.**  
**Out of scope:** Unimplemented sampler names such as `abnorsett4_3h2s` (name registered, no `case` / `rk_coeff` body) — that is a separate RES4LYF upstream gap, not this fix.

**Typical failing pipeline:** txt2img, SDXL, RES4LYF sampler `lawson45-gen-mod_4h4s` / `lawson45-gen-mod_4h4s_ode`, scheduler `Beta57`, ControlNet IP-Adapter FaceID Plus V2, MultiDiffusion, ADetailer.

---

## 1. Error Content

### 1-1. Console / progress

Sampling starts and advances for several steps, then aborts mid-run (example: **5/16**):

```text
(RES4LYF) rk_type: lawson45-gen-mod_4h4s
  0%|                                                                                           | 0/16 [00:00<?, ?it/s]
[A1111] FA-2 (Flash-Attention 2.9.1) called directly
 31%|█████████████████████████▉                                                         | 5/16 [00:17<00:34,  3.12s/it]
*** Error completing request
```

### 1-2. Exception

```text
IndexError: index 4 is out of bounds for dimension 0 with size 4
```

### 1-3. Stack (abbreviated)

```text
File "modules/call_queue.py", line 74, in f
File "modules/txt2img.py", line 109, in txt2img
File "modules/processing.py", line 848 / 1005, in process_images / process_images_inner
File "extensions-builtin/sd-webui-controlnet/scripts/hook.py", line 470, in process_sample
File "modules/sd_samplers_kdiffusion.py", line 230, in sample
File "modules/a1111_res4lyf_samplers.py", line 274, in wrapped_func
File "modules/a1111_res4lyf_samplers.py", line 197, in sample_ode_fn
File "modules/RES4LYF/beta/rk_sampler_beta.py", line 2029, in sample_rk_beta
    data_prev_[recycled_stages - ms] = data_prev_[recycled_stages - ms - 1]
    ~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^
IndexError: index 4 is out of bounds for dimension 0 with size 4
```

### 1-4. Affected samplers (this bug class)

Any RES4LYF **hybrid** RK type whose name ends in `NhMs` with **`N >= 4`**, so that:

```text
hybrid_stages = int(rk_type[-4])  # e.g. lawson45-gen-mod_4h4s → 4
```

and `recycled_stages` becomes **4** while `data_prev_` still has only **4** slots (valid indices **0..3**).

Confirmed failing example: **`lawson45-gen-mod_4h4s`** / **`lawson45-gen-mod_4h4s_ode`**.

Early steps may still run because hybrid methods often fall back to `res_*` for the first few steps (`rk_coefficients_beta.py` hybrid gate). The crash appears once the real hybrid stage is active and the history-rotation loop runs with `recycled_stages == 4`.

---

## 2. Essential Root Cause

### 2-1. Hardcoded buffer size 4

In `modules/RES4LYF/beta/rk_sampler_beta.py`, multistep / hybrid history buffers are allocated as:

```python
data_prev_ = torch.zeros(4, *x.shape, dtype=default_dtype, device=work_device)
# comment in source: "multistep max is 4m... so 4 needed"
```

Same pattern for sync-guide buffers:

```python
data_prev_x_ = torch.zeros(4, *x.shape, ...)
data_prev_y_ = torch.zeros(4, *x.shape, ...)
```

(lines **744**, **746**, **1134**, **1135**).

### 2-2. `hybrid_stages` can be 4

In `modules/RES4LYF/beta/rk_coefficients_beta.py`:

```python
if rk_type[-3] == "h" and rk_type[-1] == "s":  # hybrid method
    ...
    hybrid_stages = int(rk_type[-4])
```

For `lawson45-gen-mod_4h4s`, the digit before `h` is **4** → `hybrid_stages = 4`.

### 2-3. `recycled_stages` takes the max

In `rk_sampler_beta.py` (around line **725**):

```python
recycled_stages = max(rk_swap_stages, RK.multistep_stages, RK.hybrid_stages, data_prev_len)
```

When `RK.hybrid_stages == 4`, `recycled_stages` becomes **4** (after the first init loop that temporarily sets `recycled_stages = len(data_prev_)-1 == 3`).

### 2-4. Rotation loop writes index 4

At the end of each step (lines **2027–2029**):

```python
data_prev_[0] = data_[0]
for ms in range(recycled_stages):
    data_prev_[recycled_stages - ms] = data_prev_[recycled_stages - ms - 1]
```

With `recycled_stages = 4` and `ms = 0`:

```text
data_prev_[4] = data_prev_[3]
```

But `data_prev_.shape[0] == 4` → valid indices are only **0, 1, 2, 3** → **`IndexError`**.

### 2-5. Why A1111 cannot “just edit RES4LYF”

`modules/RES4LYF/` is third-party / AGPL-attributed code kept unmodified in this fork’s policy for this fix. The correct place for the workaround is the existing A1111 bridge: `modules/a1111_res4lyf_shim.py`, which already wraps every RES4LYF sampler call via `res4lyf_shim_context`.

### 2-6. Causal chain (summary)

```text
UI sampler: lawson45-gen-mod_4h4s(_ode)
    → hybrid_stages = 4
    → recycled_stages = 4
    → data_prev_ allocated as torch.zeros(4, ...)
    → rotation writes data_prev_[4]
    → IndexError
```

---

## 3. Countermeasure

### ① Overview

Do **not** change RES4LYF sources. Inside `res4lyf_shim_context` (already entered for every RES4LYF sample via `a1111_res4lyf_samplers.wrapped_func`):

1. Resolve the live module object `RES4LYF.beta.rk_sampler_beta` (with fallbacks).
2. Temporarily replace **that module’s** `torch` attribute with a thin proxy.
3. The proxy forwards all attributes to real `torch`, but intercepts `zeros`: if the first positional argument is the integer **`4`**, allocate with first dim **`8`** instead.
4. On context exit (`finally`), restore the original `torch` on the module.

Effect: `data_prev_` / `data_prev_x_` / `data_prev_y_` become size **8** on dim 0 → index **4** is valid. Other `torch.zeros` calls with a different leading dim are unchanged. Global `torch` for the rest of A1111 is not patched.

**Why 8:** `hybrid_stages` max in current name list is 4 → need at least **5** slots (`0..4`). **8** gives headroom without touching RES4LYF.

**Module name note:** An early draft looked up `beta.rk_sampler_beta` only; that name does not resolve at A1111 runtime. The live name is `RES4LYF.beta.rk_sampler_beta`. Without the correct name, the proxy never installed and the IndexError remained.

### ② Modified file names

| Path | Role |
|------|------|
| `modules/a1111_res4lyf_shim.py` | Runtime fix (patch inside `res4lyf_shim_context`) |
| `md/A1111_RES4LYF_INTEGRATION.md` | Integration doc synced to include the same patch block |

**Not modified:** anything under `modules/RES4LYF/`.

### ③ Full text of the added / modified code

#### Import addition (top of `a1111_res4lyf_shim.py`)

```python
import sys
```

(alongside existing `inspect`, `logging`, `contextmanager`, `torch`).

#### Patch block inside `res4lyf_shim_context` (after diffusion_model alias, before `yield`)

```python
    # --- 3. patch RES4LYF rk_sampler_beta torch.zeros for data_prev_* buffers ---
    #
    # RES4LYF allocates ``data_prev_`` / ``data_prev_x_`` / ``data_prev_y_``
    # as ``torch.zeros(4, *x.shape, ...)`` (rk_sampler_beta.py lines 744, 746,
    # 1134, 1135) — hardcoded size 4. For hybrid samplers like
    # ``lawson45-gen-mod_4h4s`` where ``hybrid_stages = 4``, ``recycled_stages``
    # becomes 4 (rk_sampler_beta.py line 725), and the rotation loop at line
    # 2028-2029 writes ``data_prev_[4]`` → IndexError (size 4, valid 0..3).
    #
    # We cannot edit ``modules/RES4LYF/``. Instead, we replace ``torch.zeros``
    # **only in the ``rk_sampler_beta`` module's globals** (not the global
    # ``torch`` package) with a wrapper that grows any size-4 first-dim
    # allocation to 8 — enough for all current hybrid_stages values
    # (max 4 → need 5 slots, 8 gives headroom). Other torch.zeros calls
    # (different leading dim) pass through unchanged.
    rk_sampler_beta_mod = None
    try:
        import importlib
        # RES4LYF is imported as a subpackage of ``modules``; the fully
        # qualified name is ``RES4LYF.beta.rk_sampler_beta``. The bare
        # ``beta.rk_sampler_beta`` form only resolves when CWD is inside
        # the RES4LYF package, which is not the case at runtime.
        for _modname in (
            "RES4LYF.beta.rk_sampler_beta",
            "modules.RES4LYF.beta.rk_sampler_beta",
            "beta.rk_sampler_beta",
        ):
            _m = sys.modules.get(_modname)
            if _m is not None:
                rk_sampler_beta_mod = _m
                break
        if rk_sampler_beta_mod is None:
            for _modname in (
                "RES4LYF.beta.rk_sampler_beta",
                "modules.RES4LYF.beta.rk_sampler_beta",
                "beta.rk_sampler_beta",
            ):
                try:
                    rk_sampler_beta_mod = importlib.import_module(_modname)
                    break
                except Exception:
                    continue
    except Exception:
        rk_sampler_beta_mod = None

    zeros_patched = False
    _rk_torch = None
    if rk_sampler_beta_mod is not None:
        # Inject a module-level helper that shadows torch.zeros only inside
        # rk_sampler_beta's global scope. We do this by replacing the
        # ``torch`` attribute the module sees with a lightweight namespace
        # that proxies all torch.* calls but intercepts ``zeros``.
        try:
            _rk_torch = rk_sampler_beta_mod.torch
            _orig_torch_zeros = _rk_torch.zeros

            def _patched_zeros(*args, **kwargs):
                if args and isinstance(args[0], int) and args[0] == 4:
                    args = (8,) + args[1:]
                return _orig_torch_zeros(*args, **kwargs)

            # Build a proxy that forwards every attribute to real torch except
            # ``zeros`` which goes through our wrapper.
            class _TorchProxy:
                def __getattr__(self, name):
                    return getattr(_rk_torch, name)

                def zeros(self, *args, **kwargs):
                    return _patched_zeros(*args, **kwargs)

            rk_sampler_beta_mod.torch = _TorchProxy()
            zeros_patched = True
        except Exception:
            logger.debug("[RES4LYF shim] Failed to patch rk_sampler_beta.torch.zeros", exc_info=True)

    try:
        yield
    finally:
        # --- restore rk_sampler_beta.torch ---
        if zeros_patched and rk_sampler_beta_mod is not None:
            try:
                rk_sampler_beta_mod.torch = _rk_torch
            except Exception:
                logger.debug("[RES4LYF shim] restore rk_sampler_beta.torch failed", exc_info=True)
        # --- restore diffusion_model ---
        # ... (existing cleanup unchanged) ...
```

(The `finally` block continues with the pre-existing `diffusion_model` and `model_sampling` restore logic.)

### ④ Meaning of the code

| Piece | Meaning |
|-------|---------|
| `sys.modules.get("RES4LYF.beta.rk_sampler_beta")` | Prefer the already-imported sampler module used by `a1111_res4lyf_samplers` / RES4LYF registration. |
| Fallback `importlib.import_module(...)` | If not yet cached, import under the same candidate names. |
| `_TorchProxy` | Module-local stand-in for `torch`: every attribute except `zeros` is delegated to the real torch module. |
| `_patched_zeros`: `4 → 8` | Only the hardcoded history-buffer allocations grow; unrelated zeros keep original shapes. |
| `rk_sampler_beta_mod.torch = _TorchProxy()` | RES4LYF code does `import torch` then `torch.zeros(...)`. Binding `torch` in **that module’s** globals makes subsequent `torch.zeros` hit the proxy. |
| `finally` restore | After the sampler returns (success or exception), put the real `torch` back so later non-RES4LYF work is unaffected. |
| No edit under `modules/RES4LYF/` | Policy: fix via A1111 shim, not by rewriting third-party RES4LYF. |

---

## 4. Verification

After restarting A1111 with the patched shim:

1. Select sampler **`lawson45-gen-mod_4h4s`** or **`lawson45-gen-mod_4h4s_ode`**.
2. Run a short txt2img (e.g. 16 steps) that previously died around step 5.
3. Expect: progress reaches completion **without** `IndexError: index 4 is out of bounds for dimension 0 with size 4`.

If the IndexError still appears, confirm the running process loaded `modules/a1111_res4lyf_shim.py` from commit `302589c0` or later (full WebUI restart required; Gradio reload alone may keep an old module in memory).

---

## 5. Explicitly excluded from this document

| Issue | Why excluded |
|-------|----------------|
| `abnorsett4_3h2s` → `UnboundLocalError: ci` | Sampler name is listed in RES4LYF’s UI name table, but there is **no** `case "abnorsett4_3h2s"` and **no** `rk_coeff` entry. The method body does not exist. That is not a buffer-size bug and is not fixed by this shim patch. |

---

## 6. Related files (read-only context)

| Path | Relevance |
|------|-----------|
| `modules/RES4LYF/beta/rk_sampler_beta.py` | Allocates `data_prev_*`; rotation at ~2029 |
| `modules/RES4LYF/beta/rk_coefficients_beta.py` | Sets `hybrid_stages` from name digit |
| `modules/a1111_res4lyf_samplers.py` | Calls `res4lyf_shim_context` around each RES4LYF sample |
| `md/A1111_RES4LYF_INTEGRATION.md` | Broader integration notes; includes mirrored patch text |

---

## 7. Commit reference

```text
302589c0 — fix: patch RES4LYF rk_sampler_beta data_prev_ buffer IndexError for 4h4s hybrid samplers
```
