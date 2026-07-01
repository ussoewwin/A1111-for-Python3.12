"""
Bridging shim so RES4LYF samplers can run against A1111's model stack.

RES4LYF was written against ComfyUI. Its sampler code (e.g.
``modules/RES4LYF/beta/rk_sampler_beta.py``) traverses
``model.inner_model.inner_model.model_sampling`` and
``model.inner_model.inner_model.device``. In A1111 that innermost
object is ``ldm.models.diffusion.ddpm.LatentDiffusion`` which has no
such attributes.

This module provides:

- ``A1111ModelSamplingShim``: subclass of ``comfy.model_sampling.EPS``
  that borrows sigmas / sigma-t transforms from A1111's k-diffusion
  ``CompVisDenoiser`` and passes ``isinstance(_, EPS)`` checks used by
  RES4LYF.
- ``res4lyf_shim_context``: context manager that temporarily attaches
  ``model_sampling`` and ``device`` onto the ``LatentDiffusion`` object
  for the duration of one sampler run, then restores the original state.

Scope (Phase 1): SD1 / SDXL, EPS parameterization only. See
``md/A1111_RES4LYF_SHIM_PLAN.md`` for the full plan and roadmap.
"""

from __future__ import annotations

import inspect
import logging
from contextlib import contextmanager

import torch

logger = logging.getLogger(__name__)


def _get_eps_base():
    """Import ``comfy.model_sampling.EPS`` lazily.

    Import is deferred so this module can be imported at A1111 startup
    even before ``ComfyUI-master`` has been added to ``sys.path``
    (``a1111_res4lyf_samplers._ensure_comfyui_on_path`` runs at
    registration time).
    """
    from comfy.model_sampling import EPS  # noqa: WPS433
    return EPS


class _ShimBase:
    """Placeholder replaced by the real EPS-derived class at first call.

    We can't inherit from ``EPS`` at module import time because
    ``comfy.model_sampling`` may not yet be importable. Instead we build
    the concrete class lazily inside :func:`build_shim`.
    """


_shim_cls_cache = None


def build_shim(comp_vis_denoiser):
    """Return an instance of the EPS-derived shim.

    Class is cached so ``isinstance(shim, EPS)`` returns True consistently
    across calls (a fresh class per call would break equality-style
    checks in RES4LYF).
    """
    global _shim_cls_cache
    if _shim_cls_cache is None:
        eps_cls = _get_eps_base()

        class A1111ModelSamplingShim(eps_cls):
            """
            Bridges A1111's ``CompVisDenoiser`` into ComfyUI's
            ``ModelSamplingDiscrete + EPS`` API.

            Only EPS parameterization (SD1 / SDXL). V_PREDICTION and
            later families are Phase 2+.
            """

            sigma_data = 1.0

            def __init__(self, cvd):
                self._cvd = cvd

            @property
            def sigmas(self):
                return self._cvd.sigmas

            @property
            def log_sigmas(self):
                return self._cvd.log_sigmas

            @property
            def sigma_min(self):
                return self._cvd.sigmas[0]

            @property
            def sigma_max(self):
                return self._cvd.sigmas[-1]

            def timestep(self, sigma):
                return self._cvd.sigma_to_t(sigma)

            def sigma(self, timestep):
                return self._cvd.t_to_sigma(timestep)

            def percent_to_sigma(self, percent):
                if percent <= 0.0:
                    return 999999999.9
                if percent >= 1.0:
                    return 0.0
                percent = 1.0 - percent
                return self.sigma(torch.tensor(percent * 999.0)).item()

            # calculate_input / calculate_denoised / noise_scaling /
            # inverse_noise_scaling are inherited from EPS.

        _shim_cls_cache = A1111ModelSamplingShim

    return _shim_cls_cache(comp_vis_denoiser)


_cfg_forward_patched = False


def patch_cfg_denoiser_forward():
    """Idempotently wrap ``CFGDenoiser.forward`` to silently drop kwargs
    it does not accept.

    RES4LYF sampler internals do ``self.model(x, sigma, **extra_args)``.
    Once :func:`ensure_res4lyf_extra_args` has added ``model_options``
    (etc.) to ``extra_args`` on the sampler side, those keys leak into
    ``CFGDenoiser.forward`` and blow up because A1111's forward has a
    strict signature (``x, sigma, uncond, cond, cond_scale,
    s_min_uncond, image_cond``).

    The wrapper filters keyword arguments to only those accepted by the
    real ``forward`` signature. For all non-RES4LYF sampler paths on
    A1111 the ``extra_args`` dict never contains foreign keys, so the
    wrapper is a no-op there.

    Safe to call multiple times. First call installs the wrapper;
    subsequent calls are no-ops.
    """
    global _cfg_forward_patched
    if _cfg_forward_patched:
        return
    try:
        from modules.sd_samplers_cfg_denoiser import CFGDenoiser
    except ImportError:
        logger.warning("[RES4LYF shim] CFGDenoiser not importable; skip forward patch")
        return

    original_forward = CFGDenoiser.forward
    sig = inspect.signature(original_forward)
    has_var_keyword = any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )
    if has_var_keyword:
        _cfg_forward_patched = True
        return
    accepted = {
        name
        for name, p in sig.parameters.items()
        if p.kind
        in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
        and name != "self"
    }

    def patched_forward(self, *args, **kwargs):
        if kwargs:
            filtered = {k: v for k, v in kwargs.items() if k in accepted}
        else:
            filtered = kwargs
        return original_forward(self, *args, **filtered)

    CFGDenoiser.forward = patched_forward
    _cfg_forward_patched = True
    logger.info("[RES4LYF shim] Patched CFGDenoiser.forward to drop unknown kwargs")


def ensure_res4lyf_extra_args(extra_args):
    """Ensure ``extra_args`` has the ComfyUI-style nesting RES4LYF expects.

    RES4LYF assigns into ``extra_args['model_options']['transformer_options']``
    directly. A1111's ``sampler_extra_args`` only contains ``cond`` /
    ``image_cond`` / ``uncond`` / ``cond_scale`` / ``s_min_uncond``, so we
    add the missing keys with empty dicts. Idempotent; safe to call
    multiple times.

    ``extra_args`` is mutated in place. Non-dict input is a no-op.
    """
    if not isinstance(extra_args, dict):
        return
    extra_args.setdefault("model_options", {})
    mo = extra_args["model_options"]
    if isinstance(mo, dict):
        mo.setdefault("transformer_options", {})


@contextmanager
def res4lyf_shim_context(cfg_denoiser):
    """Attach ComfyUI-compatible attributes for one RES4LYF sampler run.

    Parameters
    ----------
    cfg_denoiser :
        The ``CFGDenoiser`` instance passed as the first positional
        argument to any k-diffusion ``sample_*`` function on A1111.
        We expect:
        - ``cfg_denoiser.inner_model`` -> k-diffusion ``CompVisDenoiser``
        - ``cfg_denoiser.inner_model.inner_model`` -> ``LatentDiffusion``

    Notes
    -----
    ``LatentDiffusion.device`` already exists as a read-only ``@property``,
    so we do **not** attempt to set it. RES4LYF's ``model.inner_model.inner_model.device``
    read will resolve to the property getter, which returns the model's
    parameter device.

    Yields
    ------
    None
        The context body runs with the shim installed. Original state
        is restored on exit (both normal and exceptional).
    """
    inner_ldm = cfg_denoiser.inner_model.inner_model
    comp_vis = cfg_denoiser.inner_model

    # --- 1. install model_sampling ------------------------------------
    had_model_sampling = hasattr(inner_ldm, "model_sampling")
    original_model_sampling = getattr(inner_ldm, "model_sampling", None)

    try:
        inner_ldm.model_sampling = build_shim(comp_vis)
    except Exception:
        logger.exception("[RES4LYF shim] Failed to build model_sampling shim")
        yield
        return

    # --- 2. install diffusion_model alias (A1111 LDM has model.diffusion_model,
    #        ComfyUI BaseModel has .diffusion_model directly) ---------
    #
    # We assign via ``__dict__`` directly so torch.nn.Module.__setattr__ does
    # not register the U-Net twice under _modules (which would double up in
    # .parameters()/.state_dict()). Attribute lookup on the instance still
    # finds __dict__ before Module.__getattr__ fires.
    diffusion_installed = False
    original_diffusion_present = "diffusion_model" in inner_ldm.__dict__
    original_diffusion_val = inner_ldm.__dict__.get("diffusion_model", None)
    try:
        wrapper = getattr(inner_ldm, "model", None)
        real_unet = getattr(wrapper, "diffusion_model", None) if wrapper is not None else None
        if real_unet is not None:
            inner_ldm.__dict__["diffusion_model"] = real_unet
            diffusion_installed = True
    except Exception:
        logger.exception("[RES4LYF shim] Failed to alias diffusion_model")

    try:
        yield
    finally:
        # --- restore diffusion_model ---
        if diffusion_installed:
            try:
                if original_diffusion_present:
                    inner_ldm.__dict__["diffusion_model"] = original_diffusion_val
                else:
                    inner_ldm.__dict__.pop("diffusion_model", None)
            except Exception:
                logger.debug(
                    "[RES4LYF shim] cleanup diffusion_model failed",
                    exc_info=True,
                )
        # --- restore model_sampling ---
        try:
            if had_model_sampling:
                inner_ldm.model_sampling = original_model_sampling
            else:
                try:
                    delattr(inner_ldm, "model_sampling")
                except AttributeError:
                    pass
        except Exception:
            logger.debug(
                "[RES4LYF shim] cleanup: could not restore model_sampling",
                exc_info=True,
            )
