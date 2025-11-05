import os
import shutil
import zipfile
from pathlib import Path
import tempfile
import hashlib
import sys

# 既存のscipyインストール場所
scipy_source = Path(r'D:\USERFILES\A1111\venv\Lib\site-packages\scipy')
scipy_libs_source = Path(r'D:\USERFILES\A1111\venv\Lib\site-packages\scipy.libs')
numpy_libs_source = Path(r'D:\USERFILES\A1111\venv\Lib\site-packages\numpy.libs')
output_dir = Path(r'D:\USERFILES\GitHub\A1111-Python3.12\build_scipy')

# scipyのバージョン情報を取得
try:
    sys.path.insert(0, str(scipy_source.parent))
    import scipy
    version = scipy.__version__
    name = "scipy"
    print(f"Found scipy version: {version}")
except Exception as e:
    print(f"Could not get version: {e}")
    version = "1.16.1"
    name = "scipy"

# whlファイル名
whl_name = f"{name}-{version}-cp312-cp312-win_amd64.whl"
whl_path = output_dir / whl_name

output_dir.mkdir(parents=True, exist_ok=True)

# 一時ディレクトリでwhlを作成
with tempfile.TemporaryDirectory() as tmpdir:
    tmp_path = Path(tmpdir)
    scipy_dir = tmp_path / "scipy"
    
    # scipyをコピー
    shutil.copytree(scipy_source, scipy_dir, dirs_exist_ok=True)
    
    # scipy.libsもコピー（DLLファイル）
    # numpy.libsは既にnumpyに含まれているので、scipyのwhlには含めない
    if scipy_libs_source.exists():
        scipy_libs_dir = tmp_path / "scipy.libs"
        shutil.copytree(scipy_libs_source, scipy_libs_dir, dirs_exist_ok=True)
    
    # METADATAファイルを作成
    metadata_dir = tmp_path / f"{name}-{version}.dist-info"
    metadata_dir.mkdir()
    
    metadata_content = f"""Metadata-Version: 2.1
Name: {name}
Version: {version}
"""
    (metadata_dir / "METADATA").write_text(metadata_content, encoding='utf-8')
    
    # WHEELファイルを作成
    wheel_content = f"""Wheel-Version: 1.0
Generator: manual
Root-Is-Purelib: false
Tag: cp312-cp312-win_amd64
"""
    (metadata_dir / "WHEEL").write_text(wheel_content, encoding='utf-8')
    
    # RECORDファイルを作成
    record_lines = []
    
    # whlファイルを作成
    with zipfile.ZipFile(whl_path, 'w', zipfile.ZIP_DEFLATED) as whl:
        # tmp_path配下のすべてのファイルを追加（scipyとscipy.libsの両方）
        for root, dirs, files in os.walk(tmp_path):
            # .dist-infoディレクトリは後で追加するのでスキップ
            if '.dist-info' in root:
                continue
            for file in files:
                file_path = Path(root) / file
                arcname = file_path.relative_to(tmp_path)
                whl.write(file_path, arcname)
                # RECORDに追加
                with open(file_path, 'rb') as f:
                    file_hash = hashlib.sha256(f.read()).hexdigest()
                record_lines.append(f"{arcname.as_posix()},sha256={file_hash},{file_path.stat().st_size}")
        
        # メタデータファイルを追加
        for meta_file in metadata_dir.rglob("*"):
            if meta_file.is_file():
                arcname = meta_file.relative_to(tmp_path)
                whl.write(meta_file, arcname)
                with open(meta_file, 'rb') as f:
                    file_hash = hashlib.sha256(f.read()).hexdigest()
                record_lines.append(f"{arcname.as_posix()},sha256={file_hash},{meta_file.stat().st_size}")
        
        # RECORDファイルを追加
        record_content = "\n".join(record_lines) + "\n"
        whl.writestr(f"{name}-{version}.dist-info/RECORD", record_content)
    
    print(f"Created whl file: {whl_path}")
    print(f"Size: {whl_path.stat().st_size / 1024 / 1024:.2f} MB")
