#!/usr/bin/env python3
"""
Step 1: Environment & CUDA Verification
========================================
Verifies all required dependencies are installed and that TensorFlow
can detect the NVIDIA GPU via CUDA.

DO NOT proceed to Step 2 until this script exits with ALL CHECKS PASSED.
The GPU detection check is critical — CREPE pitch extraction will silently
fall back to CPU if CUDA is not available, making analysis impractically slow.

Usage:
    python verify_cuda.py
"""

import sys
import subprocess

# ANSI colour codes for terminal output
_PASS = "\033[32mPASS\033[0m"
_FAIL = "\033[31mFAIL\033[0m"
_WARN = "\033[33mWARN\033[0m"
_HEAD = "\033[1;36m"
_RST  = "\033[0m"

results: list[bool] = []


def _check(label: str, fn) -> object:
    """Run fn(), print result, append pass/fail to results list."""
    try:
        value = fn()
        print(f"    [{_PASS}] {label}: {value}")
        results.append(True)
        return value
    except Exception as exc:
        print(f"    [{_FAIL}] {label}: {exc}")
        results.append(False)
        return None


def _section(title: str) -> None:
    print(f"\n{_HEAD}[{len(results) + 1}] {title}{_RST}")


# ---------------------------------------------------------------------------
# 1 — Python
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("  Nightcore Analyzer — Environment & CUDA Verification")
print("=" * 60)

_section("Python")
_check("Version (3.10+ required)", lambda: f"{sys.version.split()[0]}"
       if tuple(int(x) for x in sys.version.split()[0].split(".")) >= (3, 10)
       else (_ for _ in ()).throw(RuntimeError(f"Python {sys.version.split()[0]} is too old — need ≥3.10")))


# ---------------------------------------------------------------------------
# 2 — NumPy
# ---------------------------------------------------------------------------
_section("NumPy")
def _np():
    import numpy as np
    return np.__version__
_check("Import + version", _np)


# ---------------------------------------------------------------------------
# 3 — SciPy
# ---------------------------------------------------------------------------
_section("SciPy")
def _scipy():
    import scipy
    return scipy.__version__
_check("Import + version", _scipy)


# ---------------------------------------------------------------------------
# 4 — librosa
# ---------------------------------------------------------------------------
_section("librosa")
def _librosa():
    import librosa
    return librosa.__version__
_check("Import + version", _librosa)


# ---------------------------------------------------------------------------
# 5 — soundfile
# ---------------------------------------------------------------------------
_section("soundfile")
def _soundfile():
    import soundfile as sf
    return sf.__version__
_check("Import + version", _soundfile)


# ---------------------------------------------------------------------------
# 6 — pyrubberband + rubberband CLI
# ---------------------------------------------------------------------------
_section("pyrubberband")

def _pyrubberband():
    import pyrubberband
    return getattr(pyrubberband, "__version__", "installed (no __version__)")
_check("Import + version", _pyrubberband)

def _rubberband_cli():
    r = subprocess.run(
        ["rubberband", "--version"],
        capture_output=True, text=True
    )
    # rubberband --version exits 0 on some builds, 1 on others; both are fine
    # as long as it runs
    if r.returncode not in (0, 1):
        raise RuntimeError(
            "rubberband binary not found in PATH — "
            "install via package manager, e.g. 'sudo pacman -S rubberband'"
        )
    return (r.stdout or r.stderr).strip().splitlines()[0]
_check("rubberband binary in PATH", _rubberband_cli)


# ---------------------------------------------------------------------------
# 7 — PyQt6
# ---------------------------------------------------------------------------
_section("PyQt6")
def _pyqt6():
    from PyQt6 import QtCore
    return QtCore.PYQT_VERSION_STR
_check("Import + version", _pyqt6)


# ---------------------------------------------------------------------------
# 8 — matplotlib
# ---------------------------------------------------------------------------
_section("matplotlib")
def _mpl():
    import matplotlib
    return matplotlib.__version__
_check("Import + version", _mpl)


# ---------------------------------------------------------------------------
# 9 — pyqtgraph
# ---------------------------------------------------------------------------
_section("pyqtgraph")
def _pyqtgraph():
    import pyqtgraph
    return pyqtgraph.__version__
_check("Import + version", _pyqtgraph)


# ---------------------------------------------------------------------------
# 10 — TensorFlow (most critical — must see CUDA + GPU)
# ---------------------------------------------------------------------------
_section("TensorFlow — CRITICAL: GPU / CUDA checks")

def _tf_import():
    import tensorflow as tf  # noqa: F401 — side-effect import
    return tf.__version__
tf_version = _check("Import + version", _tf_import)

def _tf_cuda_build():
    import tensorflow as tf
    if not tf.test.is_built_with_cuda():
        raise RuntimeError(
            "TensorFlow is NOT built with CUDA support. "
            "Install the CUDA-enabled build: pip install 'tensorflow[and-cuda]'"
        )
    return "Built with CUDA support"
_check("Built with CUDA", _tf_cuda_build)

def _tf_gpu_detect():
    import tensorflow as tf
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        raise RuntimeError(
            "No GPU devices detected. Check:\n"
            "      • NVIDIA drivers: nvidia-smi\n"
            "      • CUDA libraries visible to TF: see README troubleshooting\n"
            "      • LD_LIBRARY_PATH or /etc/ld.so.conf includes CUDA lib path"
        )
    names = [g.name for g in gpus]
    return f"{len(gpus)} GPU(s): {names}"
gpu_ok = _check("GPU device detection (CRITICAL)", _tf_gpu_detect)

def _tf_gpu_compute():
    import tensorflow as tf
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        raise RuntimeError("No GPU available for compute test")
    with tf.device("/GPU:0"):
        a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        b = tf.constant([[5.0, 6.0], [7.0, 8.0]])
        c = tf.matmul(a, b)
    return f"Matrix multiply on GPU:0 — output shape {tuple(c.shape)}"
_check("GPU compute test (matmul on /GPU:0)", _tf_gpu_compute)


# ---------------------------------------------------------------------------
# 11 — CREPE
# ---------------------------------------------------------------------------
_section("CREPE (TensorFlow-based pitch detector)")
def _crepe():
    import crepe
    # CREPE does not expose __version__ in older releases; check model weights
    # directory as a proxy for a working install.
    v = getattr(crepe, "__version__", None)
    if v:
        return v
    # Attempt a lightweight import of the internal model builder
    from crepe import core  # noqa: F401
    return "installed (no __version__ attribute)"
_check("Import", _crepe)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
passed = sum(results)
total  = len(results)
width  = 60

print("\n" + "=" * width)
if passed == total:
    print(f"  ALL {total} CHECKS PASSED")
    print()
    print("  GPU and CUDA: VERIFIED")
    print("  Safe to proceed to Step 2 — Core Analysis Module.")
else:
    failed = total - passed
    print(f"  {passed}/{total} checks passed  |  {failed} FAILED")
    print()
    print("  DO NOT proceed to Step 2 until all checks pass.")
    print("  Consult README.md § Troubleshooting for guidance.")
print("=" * width + "\n")

sys.exit(0 if passed == total else 1)
