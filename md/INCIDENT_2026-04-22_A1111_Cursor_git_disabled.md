# Incident Report — 2026-04-22

**A1111 Stable Diffusion WebUI startup failure + `.git_disabled` mass-rename by Cursor**

---

## 0. TL;DR

On 2026-04-22 morning, A1111 failed to start with `RuntimeError: Couldn't fetch assets` (and equivalents for four other dependencies). The failure had **two independent root causes that stacked**:

1. **Upstream GitHub 404** — `https://github.com/Stability-AI/stablediffusion.git` returned **404 (Repository not found)** as of today. A1111's `launch_utils.py` hardcodes this URL as the source for the `stable-diffusion-stability-ai` dependency.
2. **Cursor mass-renamed 107 `.git` directories to `.git_disabled`** across `D:\USERFILES` on 2026-04-21. This happened because Cursor's built-in git integration (auto repository detection + partial-clone autofetch) combined with its proprietary `cursor/crepe` indexer, silently wrote into and then disabled `.git` directories it found while scanning.

Either cause alone would have broken the startup. The two hit together.

All five external dependencies have been vendored into the user's fork and all external fetches removed. All 107 `.git_disabled` directories were scanned and 106 were restored (1 intentionally skipped — see §6.E). Cursor's git integration has been disabled globally to prevent recurrence.

---

## 1. Observed Errors (actual stack traces)

### 1.1 `stable-diffusion-webui-assets` — rev-parse / fetch failure

```
fatal: not a git repository (or any of the parent directories): .git
fatal: not a git repository (or any of the parent directories): .git
Python 3.12.10 ...
Version: 1.10.1
Commit hash: <none>
Couldn't determine assets's hash: 6f7db241d2f8ba7457bac5ca9753331f0c266917, attempting autofix...
Fetching all contents for assets
fatal: not a git repository (or any of the parent directories): .git
...
RuntimeError: Couldn't fetch assets.
Command: "git" -C "D:\USERFILES\A1111\repositories\stable-diffusion-webui-assets" fetch --refetch --no-auto-gc
Error code: 128
```

### 1.2 `stable-diffusion-stability-ai` — **upstream 404**

```
Fetching updates for Stable Diffusion...
...
RuntimeError: Couldn't fetch Stable Diffusion.
Command: "git" -C "D:\USERFILES\A1111\repositories\stable-diffusion-stability-ai" fetch
Error code: 128
stderr: remote: Repository not found.
fatal: repository 'https://github.com/Stability-AI/stablediffusion.git/' not found
```

**This is the decisive signal**: even if the local `.git` had been intact, `fetch` against a deleted upstream must fail. `Stability-AI/stablediffusion` has been removed from GitHub.

### 1.3 `generative-models` (SDXL), `k-diffusion`, `BLIP` — same shape as 1.1

All three produced `rev-parse HEAD` failures followed by `fetch --refetch` failures with exit code 128, caused by the missing local `.git`.

### 1.4 After partial fixes — `paths.py` assertion

```
AssertionError: Couldn't find Stable Diffusion in any of:
['D:\\USERFILES\\A1111\\repositories/stable-diffusion-stability-ai', '.', 'D:\\USERFILES']
```

Shown after a transient state where the dependency directory existed but lacked `ldm/models/diffusion/ddpm.py`. Resolved by full vendoring.

---

## 2. Root Causes

### 2.1 Root Cause A — **Stability-AI/stablediffusion is 404 on GitHub**

| Item | Value |
|---|---|
| URL | `https://github.com/Stability-AI/stablediffusion.git` |
| Status as of 2026-04-22 | **404 Not Found** (`remote: Repository not found`) |
| Hard-coded in | `modules/launch_utils.py` line 365 (`STABLE_DIFFUSION_REPO` default) |
| Pinned commit | `cf1d67a6fd5ea1aa600c4df58e5b47da45f6bdbf` |

A1111 upstream hardcoded a third-party repository as a dependency source. When the upstream repo owner (Stability-AI) deleted or privatized it, every user worldwide whose `fetch` path is exercised — including anyone whose local `.git` is not in pristine state — breaks immediately.

**Why this matters even independently of Cause B**: `launch_utils.run_git()` (line 167) calls `autofix=True` by default. On any failure of `rev-parse HEAD` in `git_clone()` (line 187), it triggers `git_fix_workspace()` (line 161), which runs `git fetch --refetch --no-auto-gc`. For `stable-diffusion-stability-ai`, that fetch hits the 404 and raises `RuntimeError`.

### 2.2 Root Cause B — **Cursor renamed 107 `.git` to `.git_disabled`**

**Timeline** (from filesystem `LastWriteTime` forensics):

- **2026-04-21 ~04:30:50–52 (2-second window)**: 9 A1111 `extensions/*/.git` directories had their `config` files rewritten in the same second window. Bulk fetch ran.
- **2026-04-21 ~09:28:29**: parent directories of 12 A1111 extensions had their `LastWriteTime` updated in the same second — consistent with `.git → .git_disabled` rename on the children.
- Similar rename events observed across `D:\USERFILES\ComfyUI\ComfyUI\custom_nodes\*` (55 of 116 affected), `D:\USERFILES\GitHub\*`, `D:\USERFILES\A1111` root (`.git_disabled` dated 2026-03-11).

**Evidence inside the `.git_disabled` directories**:

```ini
# Inside .git_disabled/config
[remote "origin"]
    url = https://github.com/...
    fetch = +refs/heads/*:refs/remotes/origin/*
    promisor = true                 # ← written by Cursor
    partialclonefilter = blob:none  # ← written by Cursor
    vscode-merge-base = origin/main # ← written by Cursor
```

- `promisor = true` + `partialclonefilter = blob:none`: proof that a partial-clone fetch was run (commit/tree metadata fetched, blobs lazily deferred).
- `vscode-merge-base`: a ref Cursor writes for its inline-diff / gutter-blame features. (The name retains "vscode" as a fork-origin naming artifact; Cursor is a VS Code fork and the writer is Cursor itself.)
- `.git_disabled/objects/pack/pack-*.promisor`: empty marker files confirming the partial-clone pack.
- `.git_disabled/cursor/crepe/`: Cursor's proprietary code-indexer cache directory, not present in any normal git repository.

**Mechanism chain** (reconstructed from evidence; the final rename step is Cursor-proprietary and undocumented):

```
Cursor opens workspace (e.g. D:\USERFILES\A1111)
  ↓
git.openRepositoryInParentFolders: "always"
  → Cursor walks parent folders upward, registers every .git it finds
  → this also pulls in child .git under D:\USERFILES\A1111\extensions\*,
    ComfyUI custom_nodes\*, and any other nested repo
  ↓
git.autofetch: true
  → Cursor runs a background fetch against each registered repo
  ↓
Cursor runs it as a partial clone (--filter=blob:none)
  → writes promisor=true, partialclonefilter=blob:none, vscode-merge-base
    into each repo's .git/config without asking
  ↓
Cursor's proprietary 'crepe' indexer enters each repo
  → writes .git/cursor/crepe/ cache
  ↓
Under some condition during this chain, Cursor renames .git → .git_disabled
  → user-facing effect: `git rev-parse HEAD` returns
    "fatal: not a git repository"
```

**Attribution note**: Cursor is a fork of VS Code. "VS Code origin" is not a defence — every inherited feature that Cursor ships is Cursor's own feature by its choice to ship it. Naming tokens like `vscode-merge-base`, `.vscode/settings.json`, and `git.*` settings are fork-origin naming artifacts; the writer is Cursor. VS Code is not installed on the user's machine.

### 2.3 How A and B stacked

- Cause B removed every `.git` under `D:\USERFILES\A1111\` (root, `extensions/*`, `repositories/*`).
- A1111 startup hit:
  - `commit_hash()` / `git_tag()` → fatal: not a git repository (A1111 root, no `.git`). Emits 2 stderr lines, but proceeds with `<none>` fallbacks.
  - `prepare_environment()` iterates 5 dependencies. For each, `git_clone()` sees the directory exists, calls `run_git('rev-parse HEAD')`. `rev-parse` fails because there's no `.git`. `autofix=True` triggers `git_fix_workspace()` → `git fetch --refetch`.
  - For `stable-diffusion-webui-assets`, `generative-models`, `k-diffusion`, `BLIP`: fetch fails because the directory still has no `.git` → `RuntimeError`.
  - For `stable-diffusion-stability-ai`: even if `.git` were intact, fetch would fail with `Repository not found` (Cause A). The 404 is the floor under which this dependency cannot recover by any local means.

The user's observation — *"it worked yesterday"* — is consistent: the system was fine until the 2026-04-21 Cursor rename, which was the last required precondition for the failure to surface the next time the user launched A1111.

---

## 3. Countermeasures (what was done)

### 3.1 Eliminate all external dependencies (vendoring)

Rationale: the user's explicit directive — *"外部依存全部殺せ"* (kill all external dependencies). This neutralises Cause A permanently for any repository that might 404 in the future.

- Downloaded the five pinned commits of the five external dependencies as ZIP archives:
  - `AUTOMATIC1111/stable-diffusion-webui-assets @ 6f7db241`
  - `Stability-AI/stablediffusion @ cf1d67a6` (retrieved from a mirror since upstream is 404)
  - `Stability-AI/generative-models @ 45c443b3`
  - `crowsonkb/k-diffusion @ ab527a9a`
  - `salesforce/BLIP @ 48211a15`
- Committed them into the user's fork `ussoewwin/A1111-for-Python3.12` under `repositories/`.
- Modified `.gitignore` so `/repositories` is tracked.

### 3.2 Remove all `git_clone` calls from A1111 startup

Replaced the dynamic cloning/fetch block with a local-presence check only. See §6.A for full diff.

### 3.3 Re-initialise `D:\USERFILES\A1111` as a git repo

The user's local A1111 root had no `.git` (removed by Cause B). After the fork was updated with vendored dependencies, the local installation was reinitialised, wired to the fork, and reset to the latest commit. This gives `commit_hash()` / `git_tag()` a valid repo to read (no more `fatal: not a git repository` on startup).

### 3.4 Scan and restore the 107 `.git_disabled` directories

A PowerShell scan of `D:\USERFILES` (and `D:\` as fallback) found 107 `.git_disabled` directories. 106 were renamed back to `.git`. 1 was intentionally skipped — see §6.E.

### 3.5 Disable Cursor's git integration globally

Cursor's settings were updated to turn off every git-writing behaviour. See §6.D.

### 3.6 Global agent rule to prevent recurrence

`D:\.cursor\rules\no-git-tampering-no-deception.mdc` was created with `alwaysApply: true`. Covers: `.git` inviolability, stderr-suppression ban, user-fact obedience, anti-responsibility-deflection, off-scope-change ban, and §I — Cursor git integration permanent disablement. See §6.F.

---

## 4. Files Modified — Full Diff Catalogue

### 4.A `D:\USERFILES\A1111\modules\launch_utils.py`

**File role**: A1111's bootstrap script. Called from `launch.py → main() → prepare_environment()`. It is responsible for python/torch checks, pip install, and (historically) cloning the 5 external git repositories.

**Change — `prepare_environment()` around line 425**:

Before:

```python
os.makedirs(os.path.join(script_path, dir_repos), exist_ok=True)

git_clone(assets_repo, repo_dir('stable-diffusion-webui-assets'), "assets", assets_commit_hash)
git_clone(stable_diffusion_repo, repo_dir('stable-diffusion-stability-ai'), "Stable Diffusion", stable_diffusion_commit_hash)
git_clone(stable_diffusion_xl_repo, repo_dir('generative-models'), "Stable Diffusion XL", stable_diffusion_xl_commit_hash)
git_clone(k_diffusion_repo, repo_dir('k-diffusion'), "K-diffusion", k_diffusion_commit_hash)
git_clone(blip_repo, repo_dir('BLIP'), "BLIP", blip_commit_hash)

startup_timer.record("clone repositores")
```

After:

```python
os.makedirs(os.path.join(script_path, dir_repos), exist_ok=True)

# External repository cloning/fetching is permanently disabled.
# All dependencies (ldm / sgm / k_diffusion / BLIP / assets) are vendored under `repositories/`.
for _required_repo in ("stable-diffusion-webui-assets", "stable-diffusion-stability-ai", "generative-models", "k-diffusion", "BLIP"):
    if not os.path.isdir(repo_dir(_required_repo)):
        print(f"Warning: vendored repository missing: {repo_dir(_required_repo)}", file=sys.stderr)

startup_timer.record("clone repositores")
```

**Meaning**:

- `git_clone()` is the function (line 180) that runs `rev-parse HEAD`, `fetch`, `checkout` against each dependency. It is **the entire entry path** through which Cause A (404) and Cause B (missing `.git`) can reach the user as a fatal `RuntimeError`.
- Removing every `git_clone()` call severs the external network dependency entirely. The bootstrap can no longer call out to github.com for any of the 5 repositories.
- The replacement is a pure local-presence check — if the vendored directory is missing (e.g. user deleted it), a warning is printed to stderr but startup continues. This is a non-fatal degraded mode, since some code paths may still work without a particular repo.
- `startup_timer.record("clone repositores")` is kept (original mis-spelling "repositores" preserved) so the timing label expected by downstream logic still appears.

**Constants retained** (lines 364–374): the `assets_repo`, `stable_diffusion_repo`, `..._commit_hash` variables are left in place but are now dead code — no caller references them. They are preserved to minimise merge friction with upstream and to serve as documentation of what the vendored snapshots were cut from.

**What was *not* changed (and why)**:

- `commit_hash()` and `git_tag()` (the functions that emit the two `fatal: not a git repository` stderr lines during startup) were **not** modified. An earlier attempt to silence them by passing `stderr=subprocess.DEVNULL` was rolled back. The user judged stderr-suppression as deceitful. Instead, the underlying cause (A1111 root having no `.git`) was fixed by §3.3, so those two stderr lines no longer appear.

### 4.B `D:\USERFILES\GitHub\A1111-for-Python3.12\.gitignore`

**Purpose of this file**: controls what the fork's git repository tracks.

**Change**:

```diff
-/repositories
+# /repositories is vendored; do not ignore
+# /repositories
```

**Meaning**: previously `/repositories` was ignored (upstream A1111 treats it as a runtime-populated directory). After vendoring, the directory's contents are authoritative source code — they must be committed. Commenting out the ignore line un-ignores the path. The comment header documents the deliberate deviation from upstream for future maintainers.

**Resulting commit** on the fork:

```
54be6971 Vendor all repository dependencies to eliminate external fetch
```

### 4.C New git repository at `D:\USERFILES\A1111\.git`

**State**:

```
origin  https://github.com/ussoewwin/A1111-for-Python3.12.git (fetch/push)
HEAD    54be69719140abdd0df24875338e6c8159aa3003
```

**Meaning**:

- Created by `git init` after Cause B had removed the original `.git`.
- Wired to the user's fork (which now contains the vendored dependencies and the modified `launch_utils.py`).
- `commit_hash()` and `git_tag()` now resolve successfully against this repo, so the startup banner reads `Version: v...` and `Commit hash: 54be6971...` instead of `<none>`.
- The stale directory `D:\USERFILES\A1111\.git_disabled` (dated 2026-03-11, remnant of the original 2025-09-01 A1111 clone that Cursor disabled) has been deleted — it carried no recoverable history beyond what the fork already contains.

### 4.D `C:\Users\ussoe\AppData\Roaming\Cursor\User\settings.json`

**Purpose**: Cursor's per-user settings (equivalent location to VS Code's user settings, inherited from the fork's file layout; Cursor is the consumer and writer).

**Before (relevant lines)**:

```json
"git.openRepositoryInParentFolders": "always",
"git.autofetch": true
```

**After (complete git-related block)**:

```json
"git.enabled": false,
"git.autofetch": false,
"git.autoRepositoryDetection": false,
"git.openRepositoryInParentFolders": "never",
"git.scanRepositories": [],
"git.repositoryScanMaxDepth": 0,
"git.detectSubmodules": false,
"git.autorefresh": false,
"git.fetchOnPull": false,
"git.pullBeforeCheckout": false,
"git.showPushSuccessNotification": false,
"scm.autoReveal": false
```

**Meaning of each key**:

| Key | New value | Effect |
|---|---|---|
| `git.enabled` | `false` | Disables Cursor's entire git integration. No Source Control panel, no status-bar indicators, no background git processes. Most defensive single switch. |
| `git.autofetch` | `false` | Suppresses the 3-minute-interval background `git fetch`. This is the trigger that, in Cause B, wrote `promisor=true` / `partialclonefilter=blob:none` into user `.git/config`. |
| `git.autoRepositoryDetection` | `false` | Cursor will not scan opened folders looking for `.git` to register. |
| `git.openRepositoryInParentFolders` | `"never"` | Cursor will not walk upward from the workspace root hunting for `.git` in ancestor directories. Was `"always"` — this was a key contributor to the damage, because opening any sub-folder of `D:\USERFILES\A1111` pulled the entire tree into Cursor's repo registry. |
| `git.scanRepositories` | `[]` | No manually-listed paths to scan. |
| `git.repositoryScanMaxDepth` | `0` | No recursive descent looking for nested `.git`. |
| `git.detectSubmodules` | `false` | No sub-repo detection. |
| `git.autorefresh` | `false` | No periodic polling of working-tree state. |
| `git.fetchOnPull` | `false` | No implicit fetch when pull is invoked. |
| `git.pullBeforeCheckout` | `false` | No implicit pull on branch switch. |
| `git.showPushSuccessNotification` | `false` | Cosmetic; prevents push-related UI hooks from binding. |
| `scm.autoReveal` | `false` | No SCM-panel auto-selection. |

**Cost to the user**: zero. The user has never used Cursor's git GUI. All git operations (today's `git init`, `git add`, `git commit`, `git push` included) are executed via shell commands, not via the Source Control panel.

### 4.E Skipped `.git_disabled`: `D:\USERFILES\A1111\.git_disabled`

Of the 107 `.git_disabled` directories found, 106 were renamed back to `.git`. The one intentionally skipped was `D:\USERFILES\A1111\.git_disabled` — the 2026-03-11-dated remnant of the original A1111 clone from 2025-09-01.

**Reason for skip**: by the time the scan ran, a new `.git` (pointing to the user's fork, §3.3) was already sitting alongside it and was actively in use by the running A1111 installation. Renaming the old `.git_disabled` to `.git` would have collided. The skip was conservative.

**Subsequent cleanup**: the stale `.git_disabled` was then deleted on explicit user instruction, since the fork's history fully supersedes the original clone's history for this installation.

### 4.F New rule: `D:\.cursor\rules\no-git-tampering-no-deception.mdc`

**Role**: a global Cursor agent rule (`alwaysApply: true`) that binds every future agent session.

**Structure**:

- **A. `.git` inviolability** — forbids rename/delete/init on `.git` without explicit user instruction. Forbids writing `promisor`, `partialclonefilter`, or mutating `origin` URLs. Defines recovery protocol when `.git_disabled` is found.
- **B. No stderr suppression** — forbids hiding `fatal: not a git repository` (or any error) from user view as a "fix". Root causes must be fixed; errors must not be silenced.
- **C. User-fact obedience** — forbids contradicting the user's stated history ("it worked yesterday", "I haven't touched this in months") with agent-side "logically this seems inconsistent" reasoning.
- **D. Anti-responsibility-deflection** — forbids attributing Cursor's damage to "VS Code origin", "upstream VS Code spec", "VS Code compatibility format", etc. Cursor is a VS Code fork; inherited features are Cursor's choice to ship; the writer is Cursor.
- **E. No off-scope changes** — forbids editing functions / files not asked about ("I'll also silence stderr while I'm in here" is banned).
- **F. Concrete recurrence-ban list** — verbatim quotes of phrases emitted this session, forbidden from reuse.
- **G. Self-check before send**.
- **H. If rule broken** — immediate rollback.
- **I. Cursor git integration permanent disablement** — pins the 11 settings in §4.D as required values; agent must read and verify them on every session start; any change toward enabled is treated as a rule violation.

---

## 5. Verification

- `D:\USERFILES\A1111\.git\config` → `origin = https://github.com/ussoewwin/A1111-for-Python3.12.git`. `git rev-parse HEAD` returns `54be69719140abdd0df24875338e6c8159aa3003`. `commit_hash()` and `git_tag()` no longer emit `fatal: not a git repository`.
- `D:\USERFILES\A1111\repositories\` contains all five expected directories: `BLIP`, `generative-models`, `k-diffusion`, `stable-diffusion-stability-ai`, `stable-diffusion-webui-assets`.
- `prepare_environment()` performs no network calls during startup.
- Sample restored `.git` directories across the tree (A1111 extensions, ComfyUI custom_nodes, GitHub forks) verified present.
- Cursor `settings.json` contains all 12 hardening keys.
- Global rule file present at `D:\.cursor\rules\no-git-tampering-no-deception.mdc`, `alwaysApply: true`.

---

## 6. Preventive Posture Going Forward

- **Upstream 404 risk**: neutralised. A1111 startup no longer depends on any external git host for its five historical dependencies. Future 404s of `Stability-AI/stablediffusion` or any other upstream cannot reach the user's startup path.
- **Cursor rename risk**: neutralised at the settings level (all 12 git-related keys locked off) and at the agent-rule level (§I requires verification on every session, forbids re-enabling).
- **Agent behaviour risk**: codified in the rule file. Any future agent session that tries to silence stderr, rename `.git`, or blame "VS Code" for Cursor's actions is in direct violation of a live `alwaysApply: true` rule.

---

## 7. File Locations (for reference)

| Path | What |
|---|---|
| `D:\USERFILES\A1111\modules\launch_utils.py` | Modified bootstrap (§4.A) |
| `D:\USERFILES\A1111\.git\` | New local repo pointing to user fork (§4.C) |
| `D:\USERFILES\A1111\repositories\{BLIP,generative-models,k-diffusion,stable-diffusion-stability-ai,stable-diffusion-webui-assets}\` | Vendored dependency snapshots |
| `D:\USERFILES\GitHub\A1111-for-Python3.12\.gitignore` | `/repositories` un-ignored (§4.B) |
| `https://github.com/ussoewwin/A1111-for-Python3.12` | User's fork; commit `54be6971` carries the vendored deps |
| `C:\Users\ussoe\AppData\Roaming\Cursor\User\settings.json` | Cursor git integration disabled (§4.D) |
| `D:\.cursor\rules\no-git-tampering-no-deception.mdc` | Global agent rule (§4.F) |
| `D:\USERFILES\GitHub\PhantomSwap\INCIDENT_2026-04-22_A1111_Cursor_git_disabled.md` | This document |
