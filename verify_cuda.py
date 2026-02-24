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
    import importlib.metadata as _m
    ver = tf.__version__
    # Validate version is in the expected range [2.15, 2.17)
    parts = [int(x) for x in ver.split(".")[:2]]
    if parts < [2, 15]:
        raise RuntimeError(
            f"TensorFlow {ver} is too old — need >=2.15.0. "
            "Run: pip install --upgrade 'tensorflow[and-cuda]>=2.15.0,<2.17.0'"
        )
    if parts >= [2, 17]:
        raise RuntimeError(
            f"TensorFlow {ver} is newer than the tested range (<2.17). "
            "Downgrade with: pip install 'tensorflow[and-cuda]>=2.15.0,<2.17.0'"
        )
    return ver
tf_version = _check("Import + version (need >=2.15,<2.17)", _tf_import)

def _tf_cuda_build():
    import tensorflow as tf
    if not tf.test.is_built_with_cuda():
        raise RuntimeError(
            "TensorFlow is NOT built with CUDA support. "
            "Install the CUDA-enabled build: pip install 'tensorflow[and-cuda]'"
        )
    # Show the CUDA version TF was compiled against
    try:
        info = tf.sysconfig.get_build_info()
        cuda_ver  = info.get("cuda_version",  "?")
        cudnn_ver = info.get("cudnn_version", "?")
        return f"CUDA {cuda_ver} / cuDNN {cudnn_ver}"
    except Exception:
        return "Built with CUDA support"
_check("Built with CUDA", _tf_cuda_build)

def _nvidia_wheel_versions():
    """Report installed nvidia-* pip wheel versions (the bundled CUDA 12 libs)."""
    import importlib.metadata as m
    wanted = [
        "nvidia-cuda-runtime-cu12",
        "nvidia-cublas-cu12",
        "nvidia-cudnn-cu12",
        "nvidia-cufft-cu12",
    ]
    found = {}
    for pkg in wanted:
        try:
            found[pkg] = m.version(pkg)
        except m.PackageNotFoundError:
            found[pkg] = "NOT INSTALLED"
    missing = [k for k, v in found.items() if v == "NOT INSTALLED"]
    if missing:
        raise RuntimeError(
            "Missing nvidia pip wheels (should be installed by tensorflow[and-cuda]):\n"
            + "\n".join(f"          {p}" for p in missing)
        )
    return "  ".join(f"{k.split('-')[1]}={v}" for k, v in found.items())
_check("nvidia pip wheels (CUDA 12)", _nvidia_wheel_versions)

def _nvidia_smi():
    r = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(
            "nvidia-smi not found or failed — NVIDIA driver is not installed.\n"
            "      Arch/Manjaro : sudo pacman -S nvidia nvidia-utils\n"
            "      Ubuntu/Debian: sudo apt install nvidia-driver-535\n"
            "      After install: reboot, then re-run this script.\n"
            "      libcuda.so.1 (required by TensorFlow) ships with this driver."
        )
    for line in r.stdout.splitlines():
        if "Driver Version" in line:
            return line.strip()
    return r.stdout.strip().splitlines()[0]
_check("nvidia-smi (driver installed)", _nvidia_smi)

def _tf_gpu_detect():
    import tensorflow as tf
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        # --- diagnostics to pinpoint why libcuda.so.1 is missing ---
        import ctypes, os, subprocess as sp

        lines = ["No GPU devices detected — running diagnostics:\n"]

        # 1. Current LD_LIBRARY_PATH
        ldlp = os.environ.get("LD_LIBRARY_PATH", "<not set>")
        lines.append(f"      LD_LIBRARY_PATH = {ldlp}")

        # 2. Try dlopen("libcuda.so.1") with both lazy and eager binding.
        #    TF uses RTLD_NOW|RTLD_GLOBAL; ctypes default is RTLD_LAZY|RTLD_LOCAL.
        #    If lazy succeeds but eager fails, an LD_PRELOAD conflict is the cause.
        lazy_ok = False
        try:
            ctypes.CDLL("libcuda.so.1")
            lines.append("      ctypes.CDLL('libcuda.so.1') RTLD_LAZY  → OK")
            lazy_ok = True
        except OSError as e:
            lines.append(f"      ctypes.CDLL('libcuda.so.1') RTLD_LAZY  → FAILED: {e}")

        try:
            ctypes.CDLL("libcuda.so.1", mode=ctypes.RTLD_GLOBAL)
            lines.append("      ctypes.CDLL('libcuda.so.1') RTLD_GLOBAL → OK")
        except OSError as e:
            lines.append(f"      ctypes.CDLL('libcuda.so.1') RTLD_GLOBAL → FAILED: {e}")
            if lazy_ok:
                lines.append(
                    "      ↑ LAZY succeeded but GLOBAL failed — an LD_PRELOAD library\n"
                    "        is providing conflicting symbols (see LD_PRELOAD below)."
                )

        # 3. ldconfig cache
        try:
            r = sp.run(["ldconfig", "-p"], capture_output=True, text=True, timeout=5)
            hits = [l.strip() for l in r.stdout.splitlines() if "libcuda" in l]
            if hits:
                lines.append("      ldconfig -p | grep libcuda:")
                for h in hits:
                    lines.append(f"        {h}")
            else:
                lines.append(
                    "      ldconfig -p | grep libcuda → EMPTY  "
                    "(cache stale? run: sudo ldconfig)"
                )
        except Exception:
            pass

        # 4. Warn about LD_PRELOAD (e.g. NoMachine NX)
        ldp = os.environ.get("LD_PRELOAD", "")
        if ldp:
            is_nx = "nx" in ldp.lower() or "nomachine" in ldp.lower()
            nx_note = (
                "\n        NoMachine NX detected — its libnxegl.so intercepts EGL/GL\n"
                "        symbols; CUDA's eager linker (RTLD_NOW) hits a conflict.\n"
                "        Run without the preload to confirm:\n"
                "          env -u LD_PRELOAD python verify_cuda.py\n"
                "        Permanent fix — add to your analyzer launch script:\n"
                "          unset LD_PRELOAD   # bash/zsh\n"
                "          set -e LD_PRELOAD  # fish"
            ) if is_nx else ""
            lines.append(
                f"      LD_PRELOAD = {ldp}\n"
                f"        (preloaded library may supply conflicting EGL/GL symbols){nx_note}"
            )

        # 5. Actionable fix (detect shell)
        shell = os.environ.get("SHELL", "")
        if "fish" in shell:
            lines.append(
                "\n      Quick fix (fish shell):\n"
                "        set -gx LD_LIBRARY_PATH /usr/lib:$LD_LIBRARY_PATH\n"
                "        python verify_cuda.py\n"
                "\n      Permanent fix:\n"
                "        bash setup_conda_libcuda.sh   # writes fish + bash hooks\n"
                "        conda deactivate && conda activate nightcore-analyzer\n"
                "        python verify_cuda.py"
            )
        else:
            lines.append(
                "\n      Quick fix:\n"
                "        export LD_LIBRARY_PATH=/usr/lib:$LD_LIBRARY_PATH\n"
                "        python verify_cuda.py\n"
                "\n      Permanent fix:\n"
                "        bash setup_conda_libcuda.sh\n"
                "        conda deactivate && conda activate nightcore-analyzer\n"
                "        python verify_cuda.py"
            )

        raise RuntimeError("\n      ".join(lines))

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
