# Release Notes (v1.01 to v1.03)

This document contains release notes for versions v1.01 through v1.03 of `ussoewwin/A1111-for-Python3.12`.

---

## v1.03

*Tagged commits: `0346830`..`dcadfd5` (11 commits, 2026-04-23)*

- **Added**: Cross-platform Python 3.12 startup support — created `requirements_versions_py312.txt` for Linux / macOS (byte-identical to the Windows file), resolving the `FileNotFoundError` that blocked non-Windows Python 3.12 users.
- **Added**: Flash-Attention 2 platform branching in `modules/launch_utils.py` — Windows uses prebuilt wheel `flash_attn-2.8.3+cu130torch2.10.0`, Linux source-builds `flash-attn==2.8.3` via PyPI (`--no-build-isolation`, ~30 min, requires CUDA toolkit), macOS is skipped (FA2 requires CUDA).
- **Added**: SciPy platform branching — Windows uses HuggingFace prebuilt `scipy-1.16.1-cp312-cp312-win_amd64` wheel forced via `--no-deps --no-index`; Linux / macOS use PyPI `scipy==1.16.1`.
- **Added**: `clip.py` `pkg_resources` auto-fix path branched by platform — Windows `venv/Lib/...`, Linux / macOS `venv/lib/pythonX.Y/...` (dynamic `sys.version_info`).
- **Added**: OOM prevention + Flash-Attention 2 direct-load optimizations promoted from the A1111 working tree into the fork.
- **Added**: `md/LINUX_MAC_PY312_STARTUP_FIX.md` — English design document for the Linux / macOS startup fix.
- **Added**: `md/INCIDENT_2026-04-22_A1111_Cursor_git_disabled.md` — incident record for the `.git` → `.git_disabled` rename event.
- **Updated**: PyTorch **2.9.1+cu130 → 2.10.0+cu130**.
- **Updated**: Flash-Attention 2 wheel **`2.8.3+cu130torch2.9.1` → `2.8.3+cu130torch2.10.0`** (Windows).
- **Updated**: transformers baseline raised to **5.4.0+** (4.x shims dropped throughout).
- **Updated**: protobuf **4.25.2 → 7.34.1**.
- **Updated**: `configs/v1-inference.yaml` normalized to 2-space indentation while preserving `use_checkpoint: True` (gradient checkpointing, VRAM saving).
- **Updated**: `md/PYTHON312_COMPATIBILITY.md` rewritten for the 1.03 codebase and narrowed strictly to Python 3.12 scope.
- **Fixed**: `AttributeError` during quick model load caused by transformers ≥ 5 API changes (`sd_disable_initialization.py` shim removed, `import_hook.py` / `initialize.py` 4.x shims removed).
- **Removed**: Reference to non-existent `requirements_versions_py312_fallback.txt` from documentation.
- **Guarantee**: Windows install flow is byte-identical to pre-1.03 — all Linux / macOS branches are additive.
- **Technical Details**:
  - [`md/LINUX_MAC_PY312_STARTUP_FIX.md`](LINUX_MAC_PY312_STARTUP_FIX.md)
  - [`md/PYTHON312_COMPATIBILITY.md`](PYTHON312_COMPATIBILITY.md)
  - [`md/FA2_direct_load_design.md`](FA2_direct_load_design.md)

---

## v1.02

*Tagged commit: `80729aa` (2026-04-22)*

- **Added**: Self-contained installation — all repository dependencies (`ldm`, `sgm`, `k_diffusion`, `BLIP`, `assets`) vendored under `repositories/` so first-time setup performs **no external `git clone`** calls.
- **Added**: `md/FA2_direct_load_design.md` — English design document for direct Flash-Attention 2 kernel loading (bypassing xformers).
- **Fixed**: `RuntimeError: Couldn't fetch assets.` and `fatal: not a git repository` failures on restricted / offline environments.
- **Updated**: README release-notes section now links to GitHub Releases instead of inlined content.
- **Technical Details**: See [v1.02 release](https://github.com/ussoewwin/A1111-for-Python3.12/releases/tag/1.02) for complete explanation.

---

## v1.01

*Tagged commit: `b456793` (2025-11-18)*

- **Added**: Direct Flash-Attention 2 kernel load path — FA2 is loaded directly without going through xformers.
- **Updated**: PyTorch baseline to **2.9.1+cu130**.
- **Updated**: Flash-Attention 2 pinned to **`2.8.3+cu130torch2.9.1`**.
- **Fixed**: transformers `GenerationMixin` import corrected for newer releases.
- **Removed**: `xformers` auto-install block (FA2 is loaded directly; xformers no longer required).
- **Removed**: Japanese text from README in favor of English.
- **Technical Details**: See [v1.01 release](https://github.com/ussoewwin/A1111-for-Python3.12/releases/tag/1.01) for complete explanation.

---

## Related documents

- [`PYTHON312_COMPATIBILITY.md`](PYTHON312_COMPATIBILITY.md) — Python 3.12 compatibility overview.
- [`LINUX_MAC_PY312_STARTUP_FIX.md`](LINUX_MAC_PY312_STARTUP_FIX.md) — Linux / macOS startup fix design.
- [`FA2_direct_load_design.md`](FA2_direct_load_design.md) — FA2 direct kernel load design.
- [`INCIDENT_2026-04-22_A1111_Cursor_git_disabled.md`](INCIDENT_2026-04-22_A1111_Cursor_git_disabled.md) — `.git` rename incident record.
