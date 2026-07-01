# A1111 RES4LYF 統合 — 完全作業解説書

**対象リポジトリ:** `D:\USERFILES\A1111`  
**作成日:** 2026-07-01  
**目的:** ComfyUI カスタムノード RES4LYF のサンプラー／スケジューラーを、Automatic1111 (A1111) のネイティブ UI から利用可能にする。  
**前提:** `modules/RES4LYF/` と `ComfyUI-master/` は **本統合では無改変**（配置のみ）。

---

## 目次

1. [概要と設計方針](#1-概要と設計方針)
2. [変更ファイル一覧](#2-変更ファイル一覧)
3. [アーキテクチャ](#3-アーキテクチャ)
4. [起動から生成までのデータフロー](#4-起動から生成までのデータフロー)
5. [追加ファイル全文と技術解説](#5-追加ファイル全文と技術解説)
6. [修正ファイル全文と技術解説](#6-修正ファイル全文と技術解説)
7. [遭遇したエラーと対処](#7-遭遇したエラーと対処)
8. [Forge との差分まとめ](#8-forge-との差分まとめ)
9. [制限事項と今後の拡張](#9-制限事項と今後の拡張)

---

## 1. 概要と設計方針

### 1.1 何を達成したか

- A1111 の **Sampling method** ドロップダウンに RES4LYF 由来のサンプラー（約 100 件超、Forge と同等の RK 動的登録を含む）を追加
- A1111 の **Schedule type** に `beta57` と `bong_tangent` を追加
- RES4LYF サンプラー選択時に **画像生成が正常完了**する（`model_sampling` 欠如・`CFGDenoiser` 引数不一致・名前衝突などを解消）

### 1.2 設計原則

| 原則 | 内容 |
|------|------|
| RES4LYF 本体を触らない | `modules/RES4LYF/**` は ComfyUI ノードのコピーのまま |
| A1111 コアを最小変更 | `initialize.py` に起動フック数行のみ。`sd_samplers.py` 等は未改変 |
| 接着剤パターン | Forge の `modules_forge/forge_res4lyf_samplers.py` と同型の **glue + shim** |
| 実行時のみ互換化 | `res4lyf_shim_context` で ComfyUI 想定属性を一時注入し、終了時に復元 |
| 標準サンプラー保護 | `dpmpp_2m` / `euler` 等の名前衝突時は RES4LYF 側をスキップ |

### 1.3 外部依存（配置のみ・本文書のコード変更対象外）

- `ComfyUI-master/` — `comfy.*` ランタイム（`sys.path` へ手動追加）
- `modules/RES4LYF/` — RES4LYF 実装本体
- venv 追加パッケージ例: `torchaudio`, `av`, `pywavelets`, `comfy-kitchen`（不足時は glue 側が pip 試行または手動導入）

### 1.4 環境側の手動変更（コード外）

`requirements_versions_py312.txt` / `requirements_versions_py312_windows.txt` において、  
`einops==0.4.1` 固定を **`einops>=0.4.1`** に緩和（`from einops import einsum` が ComfyUI / spandrel 側で必要なため）。

---

## 2. 変更ファイル一覧

### 2.1 新規追加（3 ファイル）

| パス | 行数 | 役割 |
|------|------|------|
| `modules/a1111_res4lyf_samplers.py` | 426 | 登録 glue（path・mock・RK 動的追加・UI 登録・スケジューラ） |
| `modules/a1111_res4lyf_shim.py` | 292 | 実行時 shim（`model_sampling`・`diffusion_model` エイリアス・`CFGDenoiser` パッチ） |
| `md/A1111_RES4LYF_SHIM_PLAN.md` | — | Phase 1 実装前の技術計画書（参照用） |

### 2.2 修正（1 ファイル）

| パス | 変更内容 |
|------|----------|
| `modules/initialize.py` | `sd_samplers.set_samplers()` 直後に RES4LYF 登録フックを追加（10 行） |

### 2.3 本書の対象外

- `modules/RES4LYF/**`
- `ComfyUI-master/**`

---

## 3. アーキテクチャ

### 3.1 二層構造（Forge と同型）

```
┌─────────────────────────────────────────────────────────────┐
│  A1111 UI (txt2img / img2img)                                │
│    Sampling method  ← sd_samplers.all_samplers               │
│    Schedule type    ← sd_schedulers.schedulers               │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│  modules/a1111_res4lyf_samplers.py  (GLUE)                     │
│    · ComfyUI-master を sys.path へ                             │
│    · folder_paths / server を mock                             │
│    · import modules.RES4LYF → extra_samplers 構築            │
│    · _register_extra_rk_beta_samplers (Forge 同等の動的 RK)    │
│    · comfy.k_diffusion → k_diffusion.sampling 同期             │
│    · SamplerData を all_samplers へ append                   │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│  modules/a1111_res4lyf_shim.py  (SHIM)                        │
│    · res4lyf_shim_context: LatentDiffusion へ一時属性注入      │
│    · patch_cfg_denoiser_forward: 未知 kwargs を除去          │
│    · ensure_res4lyf_extra_args: model_options ネスト確保     │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│  modules/RES4LYF/  (ComfyUI ノード・無改変)                    │
│    rk_sampler_beta.sample_rk_beta 等                           │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 モデルオブジェクト階層の差（核心）

RES4LYF は ComfyUI の `BaseModel` を想定する。A1111 は最内層が `LatentDiffusion` であり、属性が一致しない。

| パス | ComfyUI / Forge | A1111 |
|------|-----------------|-------|
| `model` | `CFGDenoiser` | `CFGDenoiser` ✓ |
| `model.inner_model` | `CompVisDenoiser` | `CompVisDenoiser` ✓ |
| `model.inner_model.inner_model` | `BaseModel` | **`LatentDiffusion`** ✗ |
| `.model_sampling` | `comfy.model_sampling.EPS` 等 | **無い** → shim で注入 |
| `.diffusion_model` | U-Net 直参照 | **`model.diffusion_model` にネスト** → `__dict__` エイリアス |
| `.device` | 属性 | **読み取り専用 `@property`**（設定不要） |

---

## 4. 起動から生成までのデータフロー

### 4.1 起動時（`initialize_rest`）

1. A1111 標準の `sd_samplers.set_samplers()`
2. `a1111_res4lyf_samplers.register_res4lyf_samplers()`
   - `_ensure_comfyui_on_path()` → `ComfyUI-master` を `sys.path[0]` に
   - `_mock_comfyui_globals()` → 不足モジュールをスタブ
   - `from modules import RES4LYF` → `__init__.py` 内 `add_samplers()` が走り `extra_samplers` 構築
   - `_register_extra_rk_beta_samplers()` → Forge 同等の RK 名を `extra_samplers` に追加（衝突名はスキップ）
   - `comfy.k_diffusion.sampling` の `sample_*` を `k_diffusion.sampling` にコピー
   - 各名前で `SamplerData` を `all_samplers` に追加 → `set_samplers()` 再実行
   - `patch_cfg_denoiser_forward()` を一度だけ適用
3. `register_res4lyf_schedulers()` → `beta57`, `bong_tangent` を `sd_schedulers` に追加

### 4.2 生成時（RES4LYF サンプラー選択）

1. UI → `SamplerData.constructor(model)` → `_build_res4lyf_constructor` が返したクロージャ
2. `KDiffusionSampler(wrapped_func, model)` 生成
3. `wrapped_func(cfg_denoiser, x, ...)` 呼び出し時:
   - `ensure_res4lyf_extra_args(kwargs["extra_args"])`
   - `with res4lyf_shim_context(cfg_denoiser):` で `model_sampling` / `diffusion_model` 注入
   - `k_diffusion.sampling.sample_<name>(...)` → 内部で `rk_sampler_beta.sample_rk_beta`
4. RES4LYF が `self.model(x, sigma, **extra_args)` を呼ぶ
5. `CFGDenoiser.forward` はパッチ済み → `model_options` 等の余分な kwargs を無視して実行

---

## 5. 追加ファイル全文と技術解説

### 5.1 `modules/a1111_res4lyf_samplers.py`（全文）

```python
"""
A1111 integration for RES4LYF samplers and schedulers.

Forge (Stable-Diffusion-WebUI-Forge-Nunchaku) 側の
`modules_forge/forge_res4lyf_samplers.py` を A1111 向けに書き直したもの。

Forge との主な差分:
- A1111 には ``sd_samplers.add_sampler()`` が無いため
  ``all_samplers`` を直接拡張し ``set_samplers()`` を再実行する
- ComfyUI ランタイム (``ComfyUI-master``) は ``modules/paths.py`` では
  sys.path に追加されないため、本モジュールが自前で追加する
- ``bong_tangent`` は Forge の ``sd_schedulers.py`` にはネイティブ実装があったが
  A1111 には無いため、``beta57`` と併せて本モジュール側で登録する
- ``comfy.samplers.beta_scheduler`` / RES4LYF ``sigmas.bong_tangent_scheduler`` は
  ComfyUI 由来のシグネチャなので、A1111 の
  ``(n, sigma_min, sigma_max, inner_model, device)`` にラップする
"""

from __future__ import annotations

import logging
import os
import sys
import types
from typing import Callable, Optional

logger = logging.getLogger(__name__)


def _ensure_comfyui_on_path() -> Optional[str]:
    """``ComfyUI-master`` を sys.path に挿入。既にあれば何もしない。"""
    from modules.paths_internal import script_path
    comfyui_path = os.path.join(script_path, "ComfyUI-master")
    if not os.path.isdir(comfyui_path):
        logger.warning(f"[RES4LYF] ComfyUI-master not found at: {comfyui_path}")
        return None
    if comfyui_path not in sys.path:
        sys.path.insert(0, comfyui_path)
        logger.info(f"[RES4LYF] Added ComfyUI to sys.path: {comfyui_path}")
    return comfyui_path


def _install_optional_deps() -> None:
    """``pywavelets`` / ``comfy-kitchen`` を可能なら pip で入れる。失敗しても致命的にしない。"""
    try:
        from modules import launch_utils
    except ImportError:
        return
    is_installed = launch_utils.is_installed
    run_pip = launch_utils.run_pip
    if not is_installed("pywavelets"):
        try:
            run_pip("install pywavelets", "pywavelets")
            logger.info("[RES4LYF] Installed pywavelets")
        except Exception as e:
            logger.warning(f"[RES4LYF] Failed to install pywavelets: {e}")
    if not is_installed("comfy-kitchen") and not is_installed("comfy_kitchen"):
        try:
            run_pip("install comfy-kitchen", "comfy-kitchen")
            logger.info("[RES4LYF] Installed comfy-kitchen")
        except Exception as e:
            logger.warning(f"[RES4LYF] Failed to install comfy-kitchen: {e}")


def _mock_comfyui_globals() -> None:
    """
    ``folder_paths`` と ``server.PromptServer`` を、
    ComfyUI 本体を起動していない状態でも import できるように差し込む。
    ``ComfyUI-master`` に本物の ``folder_paths.py`` があればそれを優先する。
    """
    if 'folder_paths' not in sys.modules:
        try:
            import folder_paths  # noqa: F401
        except (ImportError, ModuleNotFoundError):
            mock_module = types.ModuleType('folder_paths')

            mock_module.get_filename_list = lambda *_a, **_kw: []

            def _raise_not_found(folder_type, filename):
                raise FileNotFoundError(f"Mock folder_paths: {folder_type}/{filename}")

            mock_module.get_full_path_or_raise = _raise_not_found
            mock_module.get_input_directory = lambda: ""
            mock_module.get_output_directory = lambda: ""
            mock_module.get_save_image_path = lambda *a, **kw: ("", "", 0, "", "")
            sys.modules['folder_paths'] = mock_module

    if 'server' not in sys.modules:
        class MockRoutes:
            def post(self, path):
                def decorator(func):
                    return func
                return decorator

        class MockPromptServerInstance:
            def __init__(self):
                self.routes = MockRoutes()
                self.client_id = None
                self.supports = set()

            def send_sync(self, event_type, data):
                pass

            async def send(self, event_type, data):
                pass

        class MockPromptServer:
            instance = MockPromptServerInstance()

        server_module = types.ModuleType('server')
        server_module.PromptServer = MockPromptServer
        sys.modules['server'] = server_module


# Sampler names that must not have an auto-generated ``_ode`` variant.
# These are implicit RK families where ``eta=0`` is meaningless / redundant.
# Mirrors the keyword list used in
# ``modules_forge/../modules/RES4LYF/beta/__init__.py`` (Forge fork).
_IMPLICIT_RK_KEYWORDS = (
    "gauss-legendre",
    "radau",
    "lobatto",
    "irk_exp_diag",
    "kraaijevanger",
    "qin_zhang",
    "pareschi",
    "crouzeix",
)


def _register_extra_rk_beta_samplers() -> list:
    """
    Add the dynamically-generated RK sampler set that the Forge fork of
    RES4LYF exposes.

    The RES4LYF copy shipped in A1111's ``modules/RES4LYF`` is closer to
    upstream and its ``beta/__init__.py`` only registers ~17 hand-picked
    names. The Forge fork instead iterates
    ``RK_SAMPLER_NAMES_BETA_NO_FOLDERS`` and creates one sampler per
    entry (plus an ``_ode`` variant for non-implicit families).

    Both copies share the same sampler-name list
    (``rk_coefficients_beta.py`` is byte-identical), so we can reproduce
    the Forge behaviour on the A1111 side without editing RES4LYF at
    all.

    Returns
    -------
    list[str]
        Sampler names newly registered by this call. Empty if all were
        already registered (which happens if the RES4LYF source has been
        updated to the Forge behaviour).
    """
    try:
        from modules.RES4LYF.beta.rk_coefficients_beta import (
            RK_SAMPLER_NAMES_BETA_NO_FOLDERS,
        )
        from modules.RES4LYF.beta import rk_sampler_beta
        from modules import RES4LYF
    except ImportError as e:
        logger.warning(f"[RES4LYF] Failed to import RK sampler internals: {e}")
        return []

    extra = getattr(RES4LYF, "extra_samplers", None)
    if extra is None:
        logger.warning("[RES4LYF] RES4LYF.extra_samplers not available")
        return []

    # A1111 の標準サンプラーが参照する ``k_diffusion.sampling.sample_<name>``
    # と衝突する名前は絶対に上書きしない。
    protected_funcnames = set()
    try:
        from modules import sd_samplers_kdiffusion
        for entry in sd_samplers_kdiffusion.samplers_k_diffusion:
            fn = entry[1]
            if isinstance(fn, str):
                protected_funcnames.add(fn)
    except Exception:
        logger.exception("[RES4LYF] Failed to snapshot A1111 standard sampler funcnames")

    def _make_sample_fn(rk_type):
        def sample_fn(model, x, sigmas, extra_args=None, callback=None, disable=None):
            return rk_sampler_beta.sample_rk_beta(
                model, x, sigmas, None, extra_args, callback, disable,
                rk_type=rk_type,
            )
        return sample_fn

    def _make_sample_ode_fn(rk_type):
        def sample_ode_fn(model, x, sigmas, extra_args=None, callback=None, disable=None):
            return rk_sampler_beta.sample_rk_beta(
                model, x, sigmas, None, extra_args, callback, disable,
                rk_type=rk_type, eta=0.0, eta_substep=0.0,
            )
        return sample_ode_fn

    added_names = []
    skipped_collisions = []
    for name in RK_SAMPLER_NAMES_BETA_NO_FOLDERS:
        if name == "none":
            continue
        if f"sample_{name}" in protected_funcnames:
            skipped_collisions.append(name)
            continue
        if name not in extra:
            extra[name] = _make_sample_fn(name)
            added_names.append(name)
        if not any(kw in name for kw in _IMPLICIT_RK_KEYWORDS):
            ode_name = f"{name}_ode"
            if f"sample_{ode_name}" in protected_funcnames:
                skipped_collisions.append(ode_name)
                continue
            if ode_name not in extra:
                extra[ode_name] = _make_sample_ode_fn(name)
                added_names.append(ode_name)
    if skipped_collisions:
        logger.info(
            f"[RES4LYF] Skipped {len(skipped_collisions)} name(s) that would "
            f"overwrite A1111 standard samplers: {skipped_collisions}"
        )

    if added_names:
        try:
            from comfy.samplers import KSampler, k_diffusion_sampling
            for name in added_names:
                if name not in KSampler.SAMPLERS:
                    try:
                        idx = KSampler.SAMPLERS.index("uni_pc_bh2")
                        KSampler.SAMPLERS.insert(idx + 1, name)
                    except ValueError:
                        pass
                setattr(k_diffusion_sampling, f"sample_{name}", extra[name])
        except Exception:
            logger.exception("[RES4LYF] Failed to publish extra RK samplers to comfy.k_diffusion.sampling")

    return added_names


def _build_res4lyf_constructor(sampler_key: str) -> Callable:
    """``SamplerData.constructor`` 用のクロージャを返す。"""
    def constructor(model):
        import functools

        from modules import sd_samplers_kdiffusion
        import k_diffusion.sampling
        from modules.a1111_res4lyf_shim import (
            ensure_res4lyf_extra_args,
            res4lyf_shim_context,
        )

        original_func = getattr(k_diffusion.sampling, f"sample_{sampler_key}", None)
        if original_func is None:
            raise ValueError(f"Unknown RES4LYF sampler: {sampler_key}")

        def wrapped_func(cfg_denoiser, x, *args, **kwargs):
            ensure_res4lyf_extra_args(kwargs.get("extra_args"))
            with res4lyf_shim_context(cfg_denoiser):
                return original_func(cfg_denoiser, x, *args, **kwargs)

        functools.update_wrapper(wrapped_func, original_func)

        return sd_samplers_kdiffusion.KDiffusionSampler(wrapped_func, model)
    return constructor


def register_res4lyf_samplers() -> None:
    """RES4LYF のサンプラーを A1111 の ``all_samplers`` に追加する。"""
    try:
        _install_optional_deps()
        _mock_comfyui_globals()
        comfyui_path = _ensure_comfyui_on_path()
        if comfyui_path is None:
            logger.warning("[RES4LYF] Aborting sampler registration (ComfyUI-master missing)")
            return

        from modules import RES4LYF
        import comfy.k_diffusion.sampling as comfy_k_diffusion_sampling
        import k_diffusion.sampling

        extra_added = _register_extra_rk_beta_samplers()
        if extra_added:
            logger.info(f"[RES4LYF] Added {len(extra_added)} extra RK samplers (Forge parity)")

        extra_samplers = getattr(RES4LYF, 'extra_samplers', {})
        if not extra_samplers:
            logger.info("[RES4LYF] No samplers found in extra_samplers")
            return

        if k_diffusion.sampling is not comfy_k_diffusion_sampling:
            for sampler_name in extra_samplers:
                fn = getattr(comfy_k_diffusion_sampling, f"sample_{sampler_name}", None)
                if fn is not None:
                    setattr(k_diffusion.sampling, f"sample_{sampler_name}", fn)

        from modules import sd_samplers, sd_samplers_common

        existing_names = {x.name for x in sd_samplers.all_samplers}
        added = 0
        for sampler_name in extra_samplers:
            if sampler_name in existing_names:
                continue
            if getattr(k_diffusion.sampling, f"sample_{sampler_name}", None) is None:
                logger.warning(
                    f"[RES4LYF] sample_{sampler_name} not found in k_diffusion.sampling after sync"
                )
                continue
            data = sd_samplers_common.SamplerData(
                sampler_name,
                _build_res4lyf_constructor(sampler_key=sampler_name),
                [sampler_name],
                {}
            )
            sd_samplers.all_samplers.append(data)
            sd_samplers.all_samplers_map[data.name] = data
            added += 1

        if added > 0:
            sd_samplers.set_samplers()
            logger.info(f"[RES4LYF] Registered {added} samplers")
        else:
            logger.info("[RES4LYF] No new samplers to register")

        try:
            from modules.a1111_res4lyf_shim import patch_cfg_denoiser_forward
            patch_cfg_denoiser_forward()
        except Exception as e:
            logger.warning(f"[RES4LYF] Failed to patch CFGDenoiser.forward: {e}")

    except ImportError as e:
        logger.warning(f"[RES4LYF] Failed to import RES4LYF: {e}")
    except Exception as e:
        logger.error(f"[RES4LYF] Error registering samplers: {e}", exc_info=True)


def register_res4lyf_schedulers() -> None:
    """RES4LYF スケジューラーを A1111 の ``sd_schedulers`` に登録する。"""
    try:
        _ensure_comfyui_on_path()
        _mock_comfyui_globals()

        from modules import sd_schedulers, RES4LYF  # noqa: F401
        from modules.RES4LYF import sigmas as res4lyf_sigmas
        from comfy import samplers as comfy_samplers
        import torch

        def _to_tensor(sigs, device):
            if isinstance(sigs, torch.Tensor):
                return sigs.to(device)
            return torch.as_tensor(sigs, dtype=torch.float32).to(device)

        def beta57_scheduler(n, sigma_min, sigma_max, inner_model, device):
            sigs = comfy_samplers.beta_scheduler(inner_model, n, alpha=0.5, beta=0.7)
            return _to_tensor(sigs, device)

        def bong_tangent(n, sigma_min, sigma_max, inner_model, device):
            sigs = res4lyf_sigmas.bong_tangent_scheduler(inner_model, n)
            return _to_tensor(sigs, device)

        existing = {s.name for s in sd_schedulers.schedulers}
        registered = 0

        if "beta57" not in existing:
            sch = sd_schedulers.Scheduler(
                "beta57", "Beta57", beta57_scheduler, need_inner_model=True
            )
            sd_schedulers.schedulers.append(sch)
            sd_schedulers.schedulers_map[sch.name] = sch
            sd_schedulers.schedulers_map[sch.label] = sch
            registered += 1
            logger.info("[RES4LYF] Registered scheduler: beta57")

        if "bong_tangent" not in existing:
            sch = sd_schedulers.Scheduler(
                "bong_tangent", "Bong Tangent", bong_tangent, need_inner_model=True
            )
            sd_schedulers.schedulers.append(sch)
            sd_schedulers.schedulers_map[sch.name] = sch
            sd_schedulers.schedulers_map[sch.label] = sch
            registered += 1
            logger.info("[RES4LYF] Registered scheduler: bong_tangent")

        if registered > 0:
            logger.info(f"[RES4LYF] Registered {registered} schedulers")
        else:
            logger.info("[RES4LYF] No new schedulers to register")

    except ImportError as e:
        logger.warning(f"[RES4LYF] Failed to import RES4LYF for schedulers: {e}")
    except Exception as e:
        logger.error(f"[RES4LYF] Error registering schedulers: {e}", exc_info=True)
```

#### 5.1.1 関数別の意味

| 関数 | 技術的意味 |
|------|------------|
| `_ensure_comfyui_on_path` | RES4LYF は `import comfy.*` 前提。A1111 は `ComfyUI-master` を自動で path に入れないため、glue が `script_path/ComfyUI-master` を `sys.path.insert(0, ...)` する |
| `_install_optional_deps` | RES4LYF の一部機能が `pywavelets` / `comfy-kitchen` を要求。Forge と同様に起動時 pip を試行（失敗しても登録自体は続行） |
| `_mock_comfyui_globals` | RES4LYF import 時に `folder_paths` / `server.PromptServer` が無いと落ちる箇所へのスタブ。本物が import できれば本物を優先 |
| `_register_extra_rk_beta_samplers` | **Forge パリティの要**。`RK_SAMPLER_NAMES_BETA_NO_FOLDERS` 全件から `sample_rk_beta` クロージャを `extra_samplers` に登録。implicit RK 系は `_ode` を付けない。A1111 標準の `sample_*` 名と衝突するものは **登録しない**（後述の致命バグ修正） |
| `_build_res4lyf_constructor` | A1111 の `SamplerData` は `constructor(model) -> Sampler` 形式。返す `KDiffusionSampler` の `func` を shim で包み、RES4LYF 実行時だけ ComfyUI 互換属性を付与 |
| `register_res4lyf_samplers` | 上記を束ね、`all_samplers` / `all_samplers_map` を拡張し `set_samplers()` で UI 用 alias リストを再構築 |
| `register_res4lyf_schedulers` | ComfyUI シグネチャの scheduler を A1111 の `(n, sigma_min, sigma_max, inner_model, device)` にラップして登録 |

#### 5.1.2 `k_diffusion` 二重モジュール問題

- RES4LYF は **`comfy.k_diffusion.sampling`** に `sample_<name>` を定義
- A1111 の `KDiffusionSampler` は **`k_diffusion.sampling`**（Crowsonkb 版、別モジュールオブジェクト）を参照
- 同一プロセスでも **別インスタンス** のため、`register_res4lyf_samplers` 内で属性コピーが必須:

```python
setattr(k_diffusion.sampling, f"sample_{sampler_name}", fn)
```

#### 5.1.3 名前衝突保護（標準サンプラー破壊の防止）

`RK_SAMPLER_NAMES_BETA_NO_FOLDERS` には `dpmpp_2m`, `euler`, `ddim` 等が含まれる。  
これらを `k_diffusion.sampling` に上書きすると、A1111 標準の DPM++ 2M / Euler が RES4LYF 実装に差し替わり、**標準サンプラーが壊れる**。

対策: `sd_samplers_kdiffusion.samplers_k_diffusion` から既存 `sample_*` 名をスナップショットし、衝突する RES4LYF エントリは **スキップ**（ログに `Skipped N name(s)...`）。

---

### 5.2 `modules/a1111_res4lyf_shim.py`（全文）

```python
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
    """Import ``comfy.model_sampling.EPS`` lazily."""
    from comfy.model_sampling import EPS  # noqa: WPS433
    return EPS


class _ShimBase:
    """Placeholder — real EPS subclass is built lazily in build_shim."""


_shim_cls_cache = None


def build_shim(comp_vis_denoiser):
    """Return an EPS-derived shim instance (class cached for isinstance checks)."""
    global _shim_cls_cache
    if _shim_cls_cache is None:
        eps_cls = _get_eps_base()

        class A1111ModelSamplingShim(eps_cls):
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

        _shim_cls_cache = A1111ModelSamplingShim

    return _shim_cls_cache(comp_vis_denoiser)


_cfg_forward_patched = False


def patch_cfg_denoiser_forward():
    """Wrap CFGDenoiser.forward to drop kwargs A1111 does not accept."""
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
    """Add ComfyUI-style model_options nesting if missing."""
    if not isinstance(extra_args, dict):
        return
    extra_args.setdefault("model_options", {})
    mo = extra_args["model_options"]
    if isinstance(mo, dict):
        mo.setdefault("transformer_options", {})


@contextmanager
def res4lyf_shim_context(cfg_denoiser):
    """Temporarily attach model_sampling and diffusion_model alias on LatentDiffusion."""
    inner_ldm = cfg_denoiser.inner_model.inner_model
    comp_vis = cfg_denoiser.inner_model

    had_model_sampling = hasattr(inner_ldm, "model_sampling")
    original_model_sampling = getattr(inner_ldm, "model_sampling", None)

    try:
        inner_ldm.model_sampling = build_shim(comp_vis)
    except Exception:
        logger.exception("[RES4LYF shim] Failed to build model_sampling shim")
        yield
        return

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
```

#### 5.2.1 `A1111ModelSamplingShim` の役割

- `rk_sampler_beta.py` は `isinstance(model_sampling, EPS)` および `calculate_denoised` 等を **ComfyUI の EPS クラス**経由で呼ぶ
- A1111 の `CompVisDenoiser` は同等数学を持つが **API 形状が異なる**
- shim は `CompVisDenoiser.sigmas` / `sigma_to_t` / `t_to_sigma` を委譲し、EPS 継承クラスとして振る舞う
- `sigma_data = 1.0` は ComfyUI EPS の慣例値（SD1/SDXL EPS 想定）

#### 5.2.2 `__dict__` による `diffusion_model` エイリアス

`inner_ldm.diffusion_model = real_unet` と通常代入すると、`torch.nn.Module.__setattr__` がサブモジュールとして **二重登録**し、`.parameters()` が壊れる可能性がある。

```python
inner_ldm.__dict__["diffusion_model"] = real_unet
```

インスタンス `__dict__` への直接代入により、属性ルックアップは成功するが `_modules` には載らない。

#### 5.2.3 `patch_cfg_denoiser_forward`

RES4LYF 内部: `self.model(x, sigma, **extra_args)`  
`ensure_res4lyf_extra_args` 後の `extra_args` には `model_options` が含まれる。  
A1111 の `CFGDenoiser.forward` は `**kwargs` を受け取らないため `TypeError` になる。

`inspect.signature` で許可パラメータ集合を取り、フィルタする **冪等** モンキーパッチ。標準サンプラーでは `extra_args` に foreign key が無いため実質 no-op。

#### 5.2.4 `functools.update_wrapper` との関係

`_build_res4lyf_constructor` 内で `update_wrapper(wrapped_func, original_func)` により、A1111 の `KDiffusionSampler` が `inspect.signature(self.func)` で検査する `n` / `sigmas` / `sigma_min` / `sigma_max` 等のシグネチャメタデータを維持。

---

## 6. 修正ファイル全文と技術解説

### 6.1 `modules/initialize.py`（変更箇所のみ）

**変更前（概念）:**

```python
    from modules import sd_samplers
    sd_samplers.set_samplers()
    startup_timer.record("set samplers")

    from modules import extensions
```

**変更後（実ファイル）:**

```python
    from modules import sd_samplers
    sd_samplers.set_samplers()
    startup_timer.record("set samplers")

    # Register RES4LYF samplers and schedulers (mirrors Forge's forge_res4lyf_samplers hook)
    try:
        from modules import a1111_res4lyf_samplers
        a1111_res4lyf_samplers.register_res4lyf_samplers()
        a1111_res4lyf_samplers.register_res4lyf_schedulers()
        startup_timer.record("register RES4LYF")
    except Exception:
        import traceback
        traceback.print_exc()

    from modules import extensions
```

#### 6.1.1 なぜこの位置か

- `sd_samplers.set_samplers()` **の後** — 標準サンプラー一覧が確定してから RES4LYF を **追加**
- `extensions.list_extensions()` **の前** — Extension より先にネイティブ登録しておくと、UI 再構築タイミングが安定
- `try/except` + `traceback` — RES4LYF 失敗時も **A1111 本体起動は継続**（Forge フックと同趣旨）
- `initialize_rest(reload_script_modules=True)` 経由の **リロード時も同じ経路**で再登録される

---

## 7. 遭遇したエラーと対処

| # | エラー | 原因 | 対処 |
|---|--------|------|------|
| 1 | `ModuleNotFoundError: torchaudio` | RES4LYF → `comfy.sd` 経由の依存 | venv に `pip install torchaudio`（ユーザー手動） |
| 2 | `ImportError: cannot import name 'einsum' from 'einops'` | `einops==0.4.1` 固定が古すぎる | requirements の pin を `einops>=0.4.1` に緩和 |
| 3 | `ModuleNotFoundError: av` | ComfyUI 依存 | `pip install av`（ユーザー手動） |
| 4 | `AttributeError: ... has no attribute 'model_sampling'` | `LatentDiffusion` に ComfyUI 属性無し | `a1111_res4lyf_shim.py` + `res4lyf_shim_context` |
| 5 | `AttributeError: property 'device' ... has no setter` | shim が `device` を代入しようとした | 代入をやめ、`LatentDiffusion.device` プロパティをそのまま利用 |
| 6 | `TypeError: CFGDenoiser.forward() got an unexpected keyword argument 'model_options'` | A1111 forward が厳格シグネチャ | `patch_cfg_denoiser_forward()` |
| 7 | `AttributeError: ... has no attribute 'diffusion_model'` | U-Net のパスが1段深い | `__dict__["diffusion_model"]` エイリアス |
| 8 | 標準 DPM++ / Euler が壊れる | RES4LYF が `sample_dpmpp_2m` 等を上書き | `_register_extra_rk_beta_samplers` の衝突スキップ |

---

## 8. Forge との差分まとめ

| 項目 | Forge-Nunchaku | A1111（本統合） |
|------|----------------|-----------------|
| glue ファイル | `modules_forge/forge_res4lyf_samplers.py` | `modules/a1111_res4lyf_samplers.py` |
| 起動フック | `initialize_forge()` 内 | `initialize_rest()` 内 |
| ComfyUI path | Forge backend が既に整備 | `_ensure_comfyui_on_path()` で自前追加 |
| サンプラー登録 API | `all_samplers` 直接追加 + `set_samplers()` | 同左 |
| サンプラークラス | `RES4LYFSampler(KDiffusionSampler)` | `_build_res4lyf_constructor` + shim 付き `wrapped_func` |
| RK 動的登録 | Forge 版 `RES4LYF/beta/__init__.py` 内 | A1111 glue の `_register_extra_rk_beta_samplers()` |
| `bong_tangent` | `sd_schedulers.py` にネイティブあり | glue でラップ登録 |
| モデル shim | Forge は Comfy 型 `BaseModel` | A1111 専用 `a1111_res4lyf_shim.py` が必須 |
| 名前衝突対策 | Forge 環境では問題化しにくい | **明示的スキップ**が必須 |

---

## 9. 制限事項と今後の拡張

### 9.1 現状のスコープ（Phase 1）

- **SD1 / SDXL、EPS パラメータ化** を主対象とした shim
- Flux / HiDream 向けの `double_stream_blocks` 等のパスは **未対応**（`rk_sampler_beta.py` 内の該当分岐は A1111 では通常到達しない）

### 9.2 衝突スキップにより UI に出ない名前

`RK_SAMPLER_NAMES_BETA_NO_FOLDERS` のうち A1111 標準と同名のもの（例: `dpmpp_2m`, `euler`, `ddim`）は、RES4LYF 版は **意図的に未登録**。標準サンプラーを使用すること。

### 9.3 関連ドキュメント

- `md/A1111_RES4LYF_SHIM_PLAN.md` — shim 設計・API 表面一覧・フェーズ計画
- Forge 参照実装: `Stable-Diffusion-WebUI-Forge-Nunchaku/modules_forge/forge_res4lyf_samplers.py`

---

## 付録 A: ファイルツリー（本統合で触った部分のみ）

```
D:\USERFILES\A1111\
├── modules\
│   ├── a1111_res4lyf_samplers.py   [新規]
│   ├── a1111_res4lyf_shim.py         [新規]
│   ├── initialize.py                 [修正: +10行]
│   └── RES4LYF\                    [無改変・配置のみ]
├── ComfyUI-master\                 [無改変・配置のみ]
└── md\
    ├── A1111_RES4LYF_INTEGRATION.md  [本書]
    └── A1111_RES4LYF_SHIM_PLAN.md    [計画書]
```

---

## 付録 B: 動作確認の目安

起動ログに以下が出れば登録成功の目安:

```
[RES4LYF] Added ComfyUI to sys.path: ...
[RES4LYF] Added N extra RK samplers (Forge parity)
[RES4LYF] Registered M samplers
[RES4LYF] Registered scheduler: beta57
[RES4LYF] Registered scheduler: bong_tangent
[RES4LYF shim] Patched CFGDenoiser.forward to drop unknown kwargs
```

衝突スキップ時:

```
[RES4LYF] Skipped K name(s) that would overwrite A1111 standard samplers: [...]
```

---

*本文書は `modules/a1111_res4lyf_samplers.py` / `modules/a1111_res4lyf_shim.py` / `modules/initialize.py` の実ファイル内容に基づく。*
