"""HuggingFace Hub Windows compatibility patches.

`huggingface_hub` 1.17 detects symlink support via a probe and, when the probe
succeeds, calls ``os.symlink`` directly. On Windows without Developer Mode (or
admin rights), the actual symlink call raises bare ``OSError`` (WinError 1314 -
client does not hold the required privilege). The library only catches
``PermissionError`` in this code path, so the bare ``OSError`` is propagated and
all downloads fail.

Importing this module installs a wrapper around ``_create_symlink`` that falls
back to file copy/move whenever ``os.symlink`` raises any ``OSError``. The cost
is extra disk usage (no de-dup across snapshots), but for our single-dataset
use case that is acceptable.
"""
from __future__ import annotations

import os
import shutil
import sys


def install_symlink_fallback() -> None:
    if sys.platform != "win32":
        return
    try:
        import huggingface_hub.file_download as _fd
    except ImportError:
        return

    if getattr(_fd, "_pi0_lite_symlink_patched", False):
        return

    _orig = _fd._create_symlink

    def _patched(src: str, dst: str, new_blob: bool = False) -> None:
        try:
            _orig(src, dst, new_blob=new_blob)
            return
        except OSError:
            pass

        try:
            os.remove(dst)
        except OSError:
            pass
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        abs_src = os.path.abspath(os.path.expanduser(src))
        abs_dst = os.path.abspath(os.path.expanduser(dst))
        if new_blob:
            shutil.move(abs_src, abs_dst)
        else:
            shutil.copyfile(abs_src, abs_dst)

    _fd._create_symlink = _patched
    _fd._pi0_lite_symlink_patched = True


install_symlink_fallback()
