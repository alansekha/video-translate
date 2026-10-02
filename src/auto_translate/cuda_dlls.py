"""Make the NVIDIA DLLs from the pip wheels (cuBLAS, cuDNN) visible on Windows.

The `nvidia-cublas-cu12` / `nvidia-cudnn-cu12` wheels put their DLLs in
`site-packages/nvidia/<lib>/bin`, which Windows does not search by default.
Call `setup()` BEFORE importing faster_whisper / ctranslate2.
"""

import functools
import os
import sys
from pathlib import Path


@functools.cache  # run once; later calls return the cached result
def setup() -> list[Path]:
    """Add every `nvidia/*/bin` folder to the DLL search path. Returns the folders added."""
    if sys.platform != "win32":
        return []

    try:
        import nvidia  # namespace package installed by the nvidia-* wheels
    except ImportError:
        return []

    added: list[Path] = []
    # A namespace package can span several folders, hence the loop over __path__.
    for root in nvidia.__path__:
        for bin_dir in Path(root).glob("*/bin"):
            os.add_dll_directory(str(bin_dir))
            # Also prepend to PATH: some DLLs load their own dependencies via the
            # classic search order, which ignores add_dll_directory.
            os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")
            added.append(bin_dir)
    return added
