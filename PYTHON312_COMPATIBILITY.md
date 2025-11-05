# Python 3.12 互換性修正（Windows対応改善版）

このバージョンのStable Diffusion WebUIは、Python 3.12での動作をサポートし、特にWindows環境でのビルドエラーを解決するように修正されています。

## 修正内容

### 1. Pythonバージョンチェックの更新
**ファイル**: `modules/launch_utils.py`
- `check_python_version()`関数を修正
- Windows環境: Python 3.10, 3.11, 3.12をサポート
- Linux環境: Python 3.7-3.12をサポート

### 2. Windows環境でのscikit-imageビルドエラー対策
**新ファイル**: `requirements_versions_py312_windows.txt`
- Windows環境専用のPython 3.12対応requirements
- scikit-image 0.22.0に更新（プリビルドwheel利用可能）
- Visual Studioコンパイラ不要

### 3. プラットフォーム別要件ファイル自動選択
**ファイル**: `modules/launch_utils.py`
- Windows + Python 3.12: `requirements_versions_py312_windows.txt`
- Linux/Mac + Python 3.12: `requirements_versions_py312.txt`
- Python 3.11以下: `requirements_versions.txt`

### 4. フォールバック対応
**新ファイル**: `requirements_versions_py312_fallback.txt`
- scikit-imageのビルドに失敗した場合の代替requirements
- 手動でのフォールバック対応が可能

### 5. 依存関係の更新
- `scikit-image`: 0.21.0 → 0.22.0（Windows互換性改善）
- `gradio`: 3.41.2 → 4.15.0
- `protobuf`: 3.20.0 → 4.25.2
- `transformers`: 4.30.2 → 4.36.2
- その他のパッケージも最新の互換バージョンに更新

## インストール方法

### Python 3.12環境での使用
1. Python 3.12がインストールされていることを確認
2. このフォルダに移動
3. 通常通り`webui.sh`（Linux/Mac）または`webui.bat`（Windows）を実行

### Windows環境での特別な注意事項
- Visual Studioのインストールは不要
- プリビルドwheelを自動的に使用
- ビルドエラーが発生した場合は自動的にフォールバック

## トラブルシューティング

### scikit-imageビルドエラーが発生した場合
1. **自動フォールバック**: 環境変数を設定
   ```cmd
   set REQS_FILE=requirements_versions_py312_fallback.txt
   webui.bat
   ```

2. **手動インストール**: 
   ```cmd
   pip install scikit-image>=0.22.0 --only-binary=all
   ```

3. **代替手順**: scikit-imageなしで起動し、後で個別にインストール

### その他のパッケージエラー
- 仮想環境を削除: `rmdir /s venv`
- WebUIを再起動して依存関係を再インストール

## ファイル構成
```
stable-diffusion-webui-master/
├── PYTHON312_COMPATIBILITY.md                    # 修正内容の詳細
├── requirements_versions_py312.txt               # Linux/Mac用Python 3.12対応
├── requirements_versions_py312_windows.txt       # Windows用Python 3.12対応
├── requirements_versions_py312_fallback.txt      # フォールバック用
├── modules/launch_utils.py                       # 修正されたバージョンチェック
└── （その他の元ファイル）
```

## 注意事項

- Python 3.12は比較的新しいバージョンのため、一部の拡張機能で互換性問題が発生する可能性があります
- Windows環境では、Visual Studioのインストールを避けるため、プリビルドwheelを優先使用
- 問題が発生した場合は、`--skip-python-version-check`フラグを使用してバージョンチェックをスキップできます

## 修正者情報
- 修正日: 2025年8月31日
- 対象バージョン: Stable Diffusion WebUI master branch
- Python 3.12互換性対応（Windows環境改善版）

