# Python 3.12 互換性修正（1.03 時点・Windows/Linux/Mac 対応）

このバージョンの Stable Diffusion WebUI は Python 3.12 での動作をサポートし、Windows 環境でのビルドエラー回避、および Linux/Mac 環境での起動失敗回避を目的として修正されています。

本ドキュメントはリポジトリ `1.03` 時点（commit `b5b1dd8` 以降）のコード状態を反映しています。詳細な Linux/Mac 対応の経緯は [`LINUX_MAC_PY312_STARTUP_FIX.md`](LINUX_MAC_PY312_STARTUP_FIX.md) を、FA2 ダイレクトロード設計は [`FA2_direct_load_design.md`](FA2_direct_load_design.md) を参照してください。

## 修正内容

### 1. Python バージョンチェックの更新
**ファイル**: `modules/launch_utils.py` (`check_python_version()`)

- Windows / Linux / Mac いずれの環境でも Python 3.12 をサポート対象に追加
- 本家 A1111 では Python 3.12 が許可対象に含まれていなかったため、`check_python_version()` の許可リストを拡張
- `--skip-python-version-check` フラグで警告抑止可能

### 2. プラットフォーム別 requirements ファイル自動選択
**ファイル**: `modules/launch_utils.py`

本フォークは Python 3.12 専用改造版のため、起動時は以下の requirements を OS 別に自動選択します。

| 環境 | 要件ファイル |
|------|-------------|
| Windows | `requirements_versions_py312_windows.txt` |
| Linux / Mac | `requirements_versions_py312.txt` |

両ファイルは現状バイト同一（共通基盤）で、プラットフォーム固有処理は後述の launch_utils 側分岐で吸収しています。

### 3. scikit-image ビルド回避（Windows）
- `requirements_versions_py312_windows.txt` で `scikit-image>=0.22.0` を指定
- プリビルド wheel が利用可能な版を使用し、Visual Studio コンパイラ不要
- `numpy==1.26.4` とあわせて、後述の scipy wheel の dtype レイアウトと整合

### 4. torch / CUDA スタックの更新
**ファイル**: `modules/launch_utils.py`

- `torch==2.10.0` + `torchvision`
- `TORCH_INDEX_URL`: `https://download.pytorch.org/whl/cu130`（CUDA 13.0 系）
- 以前の CUDA 11.8 / torch 2.1.x 系から大幅昇格

### 5. Flash-Attention 2 のプラットフォーム別インストール
**ファイル**: `modules/launch_utils.py`

FA2 ソースは Windows / Linux / Mac で供給方法を分岐させています。

| 環境 | インストール方法 | 備考 |
|------|------------------|------|
| Windows | プリビルド wheel（HuggingFace `ussoewwin/Flash-Attention-2_for_Windows`, cu130 + torch2.10.0, cxx11abiTRUE, cp312) | 即時インストール |
| Linux | PyPI `flash-attn==2.8.3` をソースビルド（`--no-build-isolation`） | CUDA toolkit / nvcc 必須、〜30 分程度 |
| Mac | スキップ | FA2 は CUDA 前提のため MPS バックエンドでは使用不可 |

環境変数 `FLASH_ATTN_PACKAGE` で上書き可能です。詳細な設計意図は [`FA2_direct_load_design.md`](FA2_direct_load_design.md) を参照してください。

### 6. SciPy のプラットフォーム別インストール
**ファイル**: `modules/launch_utils.py`

`numpy==1.26.4` と dtype レイアウトを合わせるため、scipy のインストール元を OS ごとに切り替えます。

| 環境 | インストール方法 |
|------|------------------|
| Windows | HuggingFace `ussoewwin/scipy-1.16.1-cp312-cp312-win_amd64` の wheel を `--no-deps --no-index` で強制適用 |
| Linux/Mac | PyPI `scipy==1.16.1` の manylinux / macosx wheel |

環境変数 `SCIPY_WHEEL`（Windows 用）で上書き可能です。

### 7. clip.py 自動修正パスのプラットフォーム分岐
**ファイル**: `modules/launch_utils.py` (`fix_clip_packaging_import()`)

OpenAI CLIP の `pkg_resources` 依存を除去するパッチを仮想環境内の `clip.py` に適用します。対象パスを OS に合わせて切り替えます。

- Windows: `venv/Lib/site-packages/clip/clip.py`
- Linux/Mac: `venv/lib/pythonX.Y/site-packages/clip/clip.py`（実行中 Python のメジャー/マイナーを動的解決）

### 8. transformers 5.4+ への昇格・互換 shim 撤去
**ファイル**: `requirements_versions_py312_windows.txt`, `requirements_versions_py312.txt`, `modules/sd_disable_initialization.py`

- `transformers==5.4.0` をベースラインに引き上げ
- `sd_disable_initialization.py` に残っていた transformers 4.x 時代の互換ハックを撤去
- これによりクイックモデルロード時の `AttributeError` を解消

### 9. protobuf v7 互換 shim
**ファイル**: `requirements_versions_py312_windows.txt`, `requirements_versions_py312.txt`

- `protobuf==7.34.1` を採用（以前は 4.x 系、さらに以前の 3.20 系から大幅昇格）
- 本家 A1111 が Python 3.12 を想定していないため、上位バージョンへの追従が必要

### 10. OOM 対策と FA2 ダイレクトロード
- `configs/v1-inference.yaml` の `use_checkpoint: True`（gradient checkpointing 有効化）による VRAM 削減を恒久化
- FA2 を xformers 経由ではなくカーネル直接ロードで有効化する改造を適用（[`FA2_direct_load_design.md`](FA2_direct_load_design.md) 参照）
- xformers 自動インストールブロックは削除済み

## インストール方法

### Python 3.12 環境での使用
1. Python 3.12 がインストールされていることを確認
2. 本フォルダに移動
3. `webui.sh`（Linux/Mac）または `webui.bat`（Windows）を実行

### Windows 環境での注意事項
- Visual Studio のインストール不要
- FA2 / scipy はプリビルド wheel を自動取得
- scikit-image もプリビルド wheel を優先

### Linux 環境での注意事項
- FA2 のソースビルドには CUDA toolkit（`nvcc`）が必要で、30 分程度の時間を要します
- ビルドを行いたくない場合は `FLASH_ATTN_PACKAGE` 環境変数で別 wheel を指定可能

### Mac 環境での注意事項
- FA2 インストールは自動スキップされます（MPS バックエンドは CUDA 非対応のため）
- その他の Python 3.12 対応は Linux と共通

## トラブルシューティング

### scikit-image ビルドエラーが発生した場合（Windows）
手動でプリビルド wheel のみを許可:
```cmd
pip install "scikit-image>=0.22.0" --only-binary=all
```

### その他のパッケージエラー
仮想環境を削除して再生成:
```cmd
rmdir /s venv
```
その後 WebUI を再起動すると依存関係が再インストールされます。

### Python バージョンチェックを回避したい場合
`--skip-python-version-check` を起動引数に追加してください。

## ファイル構成
```
stable-diffusion-webui/
├── md/
│   ├── PYTHON312_COMPATIBILITY.md         # 本ドキュメント
│   ├── LINUX_MAC_PY312_STARTUP_FIX.md     # Linux/Mac 起動失敗修正の詳細（英語）
│   ├── FA2_direct_load_design.md          # FA2 ダイレクトロード設計（英語）
│   └── INCIDENT_2026-04-22_A1111_Cursor_git_disabled.md
├── requirements_versions_py312.txt         # Linux/Mac 用
├── requirements_versions_py312_windows.txt # Windows 用
├── modules/launch_utils.py                 # 本ドキュメントで解説している分岐ロジック
├── modules/sd_disable_initialization.py    # transformers 5.x 互換化
└── configs/v1-inference.yaml               # OOM 対策（use_checkpoint: True）
```

## 関連ドキュメント
- [`LINUX_MAC_PY312_STARTUP_FIX.md`](LINUX_MAC_PY312_STARTUP_FIX.md): Linux/Mac + Python 3.12 起動失敗の根本原因解析と対策（FA2 / scipy / clip.py 分岐）
- [`FA2_direct_load_design.md`](FA2_direct_load_design.md): Flash-Attention 2 を xformers 非経由で直接ロードする改造設計
- [`INCIDENT_2026-04-22_A1111_Cursor_git_disabled.md`](INCIDENT_2026-04-22_A1111_Cursor_git_disabled.md): `.git` → `.git_disabled` リネーム事象の記録

## 注意事項

- Python 3.12 は比較的新しいバージョンのため、一部拡張機能で互換性問題が発生する可能性があります
- Windows 環境では Visual Studio のインストールを避けるため、プリビルド wheel を優先使用しています
- 問題発生時は `--skip-python-version-check` でバージョンチェックをスキップ可能です

## 修正者情報
- 最終更新日: 2026 年 4 月 23 日
- 対象バージョン: Stable Diffusion WebUI master branch（ussoewwin/A1111-for-Python3.12 1.03 相当）
- Python 3.12 互換性対応（Windows / Linux / Mac 三環境）
