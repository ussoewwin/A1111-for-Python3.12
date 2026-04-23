# Release Notes (v1.01 to v1.04)

This document contains release notes for versions v1.01 through v1.04 of `ussoewwin/A1111-for-Python3.12`.

---

## v1.04

*Tag range: `1.03`..`1.04` (2026-04-23)*

- **Added**: ADetailer is now vendored as a built-in extension under `extensions-builtin/adetailer` (no nested `.git`), so it ships with the main repository.
- **Updated**: ADetailer runtime dependencies are integrated into main Python 3.12 requirement files (`requirements_versions_py312.txt` and `requirements_versions_py312_windows.txt`) to ensure install-time consistency.
- **Updated**: `extensions-builtin/adetailer/install.py` now skips its own pip install path when loaded from `extensions-builtin`, preventing duplicate dependency installation.
- **Fixed**: Startup no longer crashes when the default `localizations` directory is missing (`modules/localization.py` now checks directory existence before listing).
- **Fixed**: Removed `mediapipe` from global requirements to resolve pip dependency conflict with pinned `protobuf==7.34.1` + `tensorflow==2.20.0`; mediapipe is now ensured via no-deps install path so ADetailer mediapipe detectors remain usable without downgrading protobuf.

---

## v1.03

*Tag range: `1.02`..`1.03` (2026-04-22 to 2026-04-23)*

- **Added**: Cross-platform Python 3.12 startup support — created `requirements_versions_py312.txt` for Linux / macOS (byte-identical to the Windows file), resolving the `FileNotFoundError` that blocked non-Windows Python 3.12 users.
- **Added**: Flash-Attention 2 platform branching in `modules/launch_utils.py` — Windows uses prebuilt wheel `flash_attn-2.8.3+cu130torch2.10.0`, Linux source-builds `flash-attn==2.8.3` via PyPI (`--no-build-isolation`, ~30 min, requires CUDA toolkit), macOS is skipped (FA2 requires CUDA).
- **Added**: SciPy platform branching — Windows uses HuggingFace prebuilt `scipy-1.16.1-cp312-cp312-win_amd64` wheel forced via `--no-deps --no-index`; Linux / macOS use PyPI `scipy==1.16.1`.
- **Added**: `clip.py` `pkg_resources` auto-fix path branched by platform — Windows `venv/Lib/...`, Linux / macOS `venv/lib/pythonX.Y/...` (dynamic `sys.version_info`).
- **Added**: OOM prevention + Flash-Attention 2 direct-load optimizations promoted from the A1111 working tree into the fork.
- **Added**: `md/FA2_direct_load_design.md` — English design document for direct Flash-Attention 2 kernel loading (bypassing xformers).
- **Added**: `md/LINUX_MAC_PY312_STARTUP_FIX.md` — English design document for the Linux / macOS startup fix.
- **Added**: `md/INCIDENT_2026-04-22_A1111_Cursor_git_disabled.md` — incident record for the `.git` → `.git_disabled` rename event.
- **Added**: `md/CHANGELOG.md` — per-release notes (this document) covering v1.01 through v1.03, linked from README.
- **Updated**: PyTorch **2.9.1+cu130 → 2.10.0+cu130**.
- **Updated**: Flash-Attention 2 wheel **`2.8.3+cu130torch2.9.1` → `2.8.3+cu130torch2.10.0`** (Windows).
- **Updated**: transformers baseline raised to **5.4.0+** (4.x shims dropped throughout).
- **Updated**: protobuf **4.25.2 → 7.34.1**.
- **Updated**: `configs/v1-inference.yaml` normalized to 2-space indentation while preserving `use_checkpoint: True` (gradient checkpointing, VRAM saving).
- **Updated**: `md/PYTHON312_COMPATIBILITY.md` rewritten for the 1.03 codebase, narrowed strictly to Python 3.12 scope, and translated to English.
- **Updated**: README rewritten for 1.03 — per-OS Installation sections (Windows / Linux / macOS), Default Package Versions table, How Linux / macOS support works, Changelog link.
- **Fixed**: `AttributeError` during quick model load caused by transformers ≥ 5 API changes (`sd_disable_initialization.py` shim removed, `import_hook.py` / `initialize.py` 4.x shims removed).
- **Removed**: Reference to non-existent `requirements_versions_py312_fallback.txt` from documentation.
- **Guarantee**: Windows install flow is byte-identical to pre-1.03 — all Linux / macOS branches are additive.

---

## v1.02

*Tagged commit: `80729aa` (2026-04-22)*

- **Added**: Self-contained installation — all repository dependencies (`ldm`, `sgm`, `k_diffusion`, `BLIP`, `assets`) vendored under `repositories/` so first-time setup performs **no external `git clone`** calls.
- **Fixed**: `RuntimeError: Couldn't fetch assets.` and `fatal: not a git repository` failures on restricted / offline environments.
- **Updated**: README release-notes section now links to GitHub Releases instead of inlined content.

---

## v1.01

*Tagged commit: `b456793` (2025-11-18)*

- **Added**: Direct Flash-Attention 2 kernel load path — FA2 is loaded directly without going through xformers.
- **Updated**: PyTorch baseline to **2.9.1+cu130**.
- **Updated**: Flash-Attention 2 pinned to **`2.8.3+cu130torch2.9.1`**.
- **Fixed**: transformers `GenerationMixin` import corrected for newer releases.
- **Removed**: `xformers` auto-install block (FA2 is loaded directly; xformers no longer required).
- **Removed**: Japanese text from README in favor of English.

---

## Related documents

- [`PYTHON312_COMPATIBILITY.md`](PYTHON312_COMPATIBILITY.md) — Python 3.12 compatibility overview.
- [`LINUX_MAC_PY312_STARTUP_FIX.md`](LINUX_MAC_PY312_STARTUP_FIX.md) — Linux / macOS startup fix design.
- [`FA2_direct_load_design.md`](FA2_direct_load_design.md) — FA2 direct kernel load design.
- [`INCIDENT_2026-04-22_A1111_Cursor_git_disabled.md`](INCIDENT_2026-04-22_A1111_Cursor_git_disabled.md) — `.git` rename incident record.
