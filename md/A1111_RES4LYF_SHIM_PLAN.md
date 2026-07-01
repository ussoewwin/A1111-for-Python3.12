# A1111 RES4LYF Integration — Shim Plan

Path: `md/A1111_RES4LYF_SHIM_PLAN.md`
Status: draft (before Phase 1 implementation)
Related files already in place:
- `modules/a1111_res4lyf_samplers.py` (new)
- `modules/initialize.py` (hook added after `sd_samplers.set_samplers()`)

## 1. Goal

Make `modules/RES4LYF` samplers usable from the A1111 native `Sampling method` dropdown, in the same way as they work under `Stable-Diffusion-WebUI-Forge-Nunchaku`. Registration side is already done in `modules/a1111_res4lyf_samplers.py`; this plan covers the runtime bridging needed so the samplers actually run and produce images.

## 2. Confirmed observations (do not re-litigate)

1. RES4LYF loads successfully into A1111 after fixing the `einops==0.4.1` pin. `ComfyUI-master`, `pywavelets`, `comfy-kitchen`, `av`, `torchaudio` are all installed.
2. The dropdown shows exactly the sampler set that `modules/RES4LYF/beta/__init__.py` (17) + `modules/RES4LYF/legacy/__init__.py` (2) explicitly add to `extra_samplers`. That is the true upper bound in the current source tree: `res_2m/3m/2s/3s/5s/6s`, `..._ode` variants, `deis_2m/3m(_ode)`, `rk_beta`, `rk`, `legacy_rk`. Total 19.
3. Selecting any of these samplers and running txt2img currently crashes with:
   ```
   AttributeError: 'LatentDiffusion' object has no attribute 'model_sampling'
   ```
   at `modules/RES4LYF/beta/rk_sampler_beta.py:254`.

## 3. Root cause

RES4LYF was written against ComfyUI's model wrapper stack. It assumes:

- `model` = `CFGDenoiser` (ok on A1111 too, name matches)
- `model.inner_model` = k-diffusion `CompVisDenoiser` (ok on A1111 too)
- **`model.inner_model.inner_model`** = ComfyUI `comfy.model_base.BaseModel`, which exposes:
  - `.device`
  - `.model_sampling` — an instance of `comfy.model_sampling.EPS` / `V_PREDICTION` / `EDM` / etc.
  - `.diffusion_model` — the raw U-Net (used for Flux/HiDream specific paths)

On A1111 `model.inner_model.inner_model` is the pytorch-lightning `ldm.models.diffusion.ddpm.LatentDiffusion`. It has none of the above attributes.

Forge-Nunchaku works because its backend already substitutes a ComfyUI-style `BaseModel` at that layer. A1111 does not.

## 4. Required API surface (from grep of `rk_sampler_beta.py`)

Attributes/methods that RES4LYF touches on `model.inner_model.inner_model`:

| Symbol | Used at (line) | Priority | Notes |
|---|---|---|---|
| `.device` | 248 | P0 | scalar attribute |
| `.model_sampling` | 254, 578, 2109 | P0 | object; see below |
| `.diffusion_model.double_stream_blocks` | 492 | P2 | Flux only |
| `.diffusion_model.single_stream_blocks` | 497 | P2 | Flux only |
| `.diffusion_model.Retrojector` | 571 | P2 | HiDream-ish |
| `.diffusion_model.y0_standard_guide` | 1142, 1705 | P2 | RES4LYF-internal guide, safe to `hasattr()`-gate |
| `.diffusion_model.y0_inv_standard_guide` | 1148, 1711 | P2 | same |
| `.diffusion_model.eps_out` | 1685 | P3 | test-only override |

On `.model_sampling` itself, ComfyUI's `EPS` / `V_PREDICTION` etc. expose:

| Method / attr | Used at | Priority | A1111 equivalent |
|---|---|---|---|
| `isinstance(_, EPS)` | 254 | P0 | none — need a real ComfyUI class |
| `.calculate_denoised(sigma, model_output, model_input)` | 2109 | P0 | k-diffusion does the same math inside `CompVisDenoiser`, but as a different call path |
| `.calculate_input(sigma, noise)` | (transitive via ComfyUI helpers if reached) | P1 | k-diffusion `sigma * noise` etc. |
| `.timestep(sigma)` | via `RK.update_transformer_options` chain | P1 | `CompVisDenoiser.sigma_to_t` |
| `.sigma(timestep)` | same | P1 | `CompVisDenoiser.t_to_sigma` |
| `.sigmas` (tensor) | schedulers already using it | P1 | `CompVisDenoiser.sigmas` |
| `.sigma_min` / `.sigma_max` | scheduler paths | P1 | `sigmas[0]` / `sigmas[-1]` |

Priority key: P0 blocks *any* run. P1 blocks first non-trivial code path. P2 only fires for models that A1111 does not currently support (Flux/HiDream via A1111 is out of scope for now). P3 is dev-only.

## 5. Shim architecture

### 5.1 Placement

The shim is applied *inside* the sampler constructor (`_build_res4lyf_constructor`) in `modules/a1111_res4lyf_samplers.py`, right before `KDiffusionSampler.__init__`. That way the shim is scoped to RES4LYF calls only and does not touch A1111 code paths for stock samplers.

### 5.2 Structure

Two thin classes in a new file `modules/a1111_res4lyf_shim.py`:

1. `A1111ModelSamplingShim` — wraps A1111's k-diffusion `CompVisDenoiser` and re-exports:
   - `calculate_denoised(sigma, eps, x)` — translate to `CompVisDenoiser` math
   - `calculate_input(sigma, noise)` — same
   - `timestep(sigma)` — via `CompVisDenoiser.sigma_to_t`
   - `sigma(timestep)` — via `CompVisDenoiser.t_to_sigma`
   - `sigmas` / `sigma_min` / `sigma_max` — from `CompVisDenoiser.sigmas`
   - Registered as a subclass of `comfy.model_sampling.EPS` (SD1/SDXL) or `V_PREDICTION` (SD2-v) so `isinstance` checks pass.
2. `A1111InnerModelShim` — thin wrapper attached to `model.inner_model.inner_model` at call time, adding:
   - `.model_sampling` = `A1111ModelSamplingShim`
   - `.device` = actual device (pass-through)
   - `.diffusion_model` = pass-through to the real U-Net (attribute forwarding); Flux-specific attributes are simply absent, so `hasattr()`-guarded paths skip.

Rather than mutating the singleton `LatentDiffusion`, the constructor **monkeypatches `model.inner_model.inner_model.model_sampling` and `.device` as instance attributes** just before invoking `sample_XXX(model, ...)`, and restores/removes them in a `try/finally`. This keeps the A1111 model object free of dangling RES4LYF state between calls.

### 5.3 EPS vs V_PREDICTION selection

Detect once per constructor call:

```python
from comfy import model_sampling as cms
if getattr(shared.sd_model, 'parameterization', 'eps') == 'v':
    parent_cls = cms.V_PREDICTION
else:
    parent_cls = cms.EPS
```

## 6. Phased implementation

**Phase 1 — Minimum viable shim (SD1 / SDXL, EPS parameterization)**
- Create `modules/a1111_res4lyf_shim.py` with `A1111ModelSamplingShim(EPS)` and `A1111InnerModelShim`.
- Modify `_build_res4lyf_constructor` in `modules/a1111_res4lyf_samplers.py`:
  - subclass `KDiffusionSampler` locally as `RES4LYFSampler`
  - override `sample()` / `sample_img2img()` to install the shim onto `model.inner_model.inner_model` before calling `super().sample(...)` and remove it after
- Verify with `res_2s` + `Beta57` on an SDXL and an SD1.5 checkpoint.

**Phase 2 — V-prediction and any missing API discovered by Phase 1 testing**
- Add V_PREDICTION path.
- Any additional `AttributeError` shown by the first real run gets added to the shim.
- Iterate until at least the P0/P1 surface is stable.

**Phase 3 (deferred, out of scope unless requested)**
- Flux/HiDream-specific attributes.
- `y0_standard_guide` and related guide extensions.

## 7. Problem-1 side (dropdown count)

Not a defect. `add_beta` / `add_legacy` register 19 names by design. If Forge is showing more, it is a difference in the RES4LYF fork/version bundled there, not something A1111 side can synthesize without editing RES4LYF itself. We do not edit RES4LYF. If the user later wants the fully-implicit (`gauss-legendre_*`, `radau_*`, `lobatto_*`) names as first-class dropdown entries, that is a separate task requiring `beta/__init__.py` changes on the RES4LYF side.

## 8. Files this plan will touch

| Path | Kind | Purpose |
|---|---|---|
| `modules/a1111_res4lyf_shim.py` | new | The shim classes described in §5.2 |
| `modules/a1111_res4lyf_samplers.py` | edit | Route sampler constructor through the shim; small change |
| `modules/initialize.py` | no change | Already hooks the registration |
| `modules/RES4LYF/**` | no change | RES4LYF stays untouched (per user policy) |
| `md/A1111_RES4LYF_SHIM_PLAN.md` | this file | Plan of record |

## 9. Risks

- **Numerical divergence.** `CompVisDenoiser.calculate_denoised` and ComfyUI's `EPS.calculate_denoised` compute the same quantity but through different code paths (float dtype defaults, `.to()` behavior). Phase 1 result must be visually compared against Forge output on the same seed to catch drift.
- **Unknown-unknowns in RES4LYF internals.** Because A1111 has never been a target for RES4LYF, Phase 1 may surface additional `AttributeError`s beyond §4. The iteration cost is bounded — each error names the missing attribute — but the number of iterations is unknown until we run.
- **Model type coverage.** SD1/SDXL first. SD2-v via V_PREDICTION second. Flux / HiDream / DiT class models via A1111 are out of scope.
- **`isinstance` pitfalls.** Some RES4LYF checks use `isinstance(model_sampling, EPS)`; if we later add V-pred, be careful that V_PREDICTION *does* subclass EPS in ComfyUI (`class V_PREDICTION(EPS)`), so `isinstance` returns True for both — matches RES4LYF's intent.

## 10. Testing

Minimum acceptance for Phase 1:
1. `res_2s` + `Beta57` + SDXL checkpoint completes txt2img without exception.
2. `res_2m` + `Bong Tangent` completes txt2img without exception.
3. Result is qualitatively similar to the same setting on Forge (no obvious garbage / NaN / all-black output).
4. Selecting a stock sampler (`Euler a`, `DPM++ 2M`) still works — no regression on non-RES4LYF paths.

## 11. Rollback

If shim work is aborted:
- Revert `modules/a1111_res4lyf_samplers.py` to the current (already-committed-in-file, not in git yet) version.
- Delete `modules/a1111_res4lyf_shim.py`.
- Remove the RES4LYF hook block from `modules/initialize.py`.
- Delete `modules/RES4LYF/` if the user wants no trace (destructive; requires explicit user order per `no-unauthorized-destruction-and-scope-creep.mdc`).

## 12. Open questions for the user (before Phase 1 code)

- Confirm target parameterization coverage: SD1 + SDXL only (fastest), or include SD2-v (adds V_PREDICTION path)?
- Confirm placement of the shim file name (`modules/a1111_res4lyf_shim.py`) is acceptable.
- Confirm that Phase 3 (Flux/HiDream) is out of scope for now.
