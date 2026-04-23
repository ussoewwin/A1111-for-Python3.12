from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from importlib.metadata import version  # python >= 3.8

from packaging.version import parse

import_name = {"py-cpuinfo": "cpuinfo", "protobuf": "google.protobuf"}

_ADETAILER_ROOT = os.path.dirname(os.path.abspath(__file__))
_BUILTIN_SUFFIX = os.path.join("extensions-builtin", "adetailer")
_IS_BUILTIN = os.path.normpath(_ADETAILER_ROOT).endswith(_BUILTIN_SUFFIX)

# mediapipe requirements:
#   mediapipe <=0.10.15 depends on protobuf<5
#   this project pins protobuf==7.34.1 for tensorflow>=2.20
# We install mediapipe with --no-deps so pip will not attempt to downgrade protobuf.
# Generated *_pb2 modules in mediapipe are runtime-compatible with protobuf 5+/7+.
_MEDIAPIPE_MIN = "0.10.13"
_MEDIAPIPE_MAX = "0.10.15"


def is_installed(
    package: str,
    min_version: str | None = None,
    max_version: str | None = None,
):
    name = import_name.get(package, package)
    try:
        spec = importlib.util.find_spec(name)
    except ModuleNotFoundError:
        return False

    if spec is None:
        return False

    if not min_version and not max_version:
        return True

    if not min_version:
        min_version = "0.0.0"
    if not max_version:
        max_version = "99999999.99999999.99999999"

    try:
        pkg_version = version(package)
        return parse(min_version) <= parse(pkg_version) <= parse(max_version)
    except Exception:
        return False


def run_pip(*args):
    subprocess.run([sys.executable, "-m", "pip", "install", *args], check=True)


def ensure_mediapipe():
    """Install mediapipe with --no-deps to avoid forcing protobuf<5.

    Runs for both external-extension and built-in modes because mediapipe is
    intentionally excluded from the main requirements to prevent the pip
    resolver from downgrading protobuf.
    """
    if is_installed("mediapipe", _MEDIAPIPE_MIN, _MEDIAPIPE_MAX):
        return
    spec = f"mediapipe>={_MEDIAPIPE_MIN},<={_MEDIAPIPE_MAX}"
    try:
        run_pip("--no-deps", "--prefer-binary", spec)
    except subprocess.CalledProcessError as e:
        print(f"[-] ADetailer: Failed to install {spec} with --no-deps: {e}")


def install():
    if _IS_BUILTIN:
        ensure_mediapipe()
        return

    deps = [
        # requirements
        ("ultralytics", "8.3.75", None),
        ("rich", "13.0.0", None),
    ]

    pkgs = []
    for pkg, low, high in deps:
        if not is_installed(pkg, low, high):
            if low and high:
                cmd = f"{pkg}>={low},<={high}"
            elif low:
                cmd = f"{pkg}>={low}"
            elif high:
                cmd = f"{pkg}<={high}"
            else:
                cmd = pkg
            pkgs.append(cmd)

    if pkgs:
        run_pip(*pkgs)

    ensure_mediapipe()


try:
    import launch

    skip_install = launch.args.skip_install
except Exception:
    skip_install = False

if not skip_install:
    install()
