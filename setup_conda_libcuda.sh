#!/usr/bin/env bash
# setup_conda_libcuda.sh
# ============================================================
# Permanently fixes TensorFlow's "Could not find cuda drivers"
# error inside the nightcore-analyzer conda environment.
#
# Root cause: tensorflow[and-cuda] calls dlopen("libcuda.so.1")
# to reach the NVIDIA driver. On Arch/Garuda this file lives at
# /usr/lib/libcuda.so.1, but conda's LD_LIBRARY_PATH prepends
# its own lib dirs, pushing /usr/lib out of TF's search path.
#
# Fix: add /usr/lib to the FRONT of LD_LIBRARY_PATH via conda's
# activate.d / deactivate.d hook mechanism so it applies every
# time the environment is activated.
#
# Usage (run ONCE, with the env active):
#   conda activate nightcore-analyzer
#   bash setup_conda_libcuda.sh
#   conda deactivate && conda activate nightcore-analyzer
#   python verify_cuda.py
# ============================================================

set -euo pipefail

if [[ -z "${CONDA_PREFIX:-}" ]]; then
    echo "ERROR: No conda environment is active."
    echo "Run: conda activate nightcore-analyzer"
    exit 1
fi

ACTIVATE_DIR="$CONDA_PREFIX/etc/conda/activate.d"
DEACTIVATE_DIR="$CONDA_PREFIX/etc/conda/deactivate.d"

mkdir -p "$ACTIVATE_DIR" "$DEACTIVATE_DIR"

# --- activation hook ---
cat > "$ACTIVATE_DIR/nightcore_libcuda_path.sh" << 'HOOK'
# Added by setup_conda_libcuda.sh
# Ensures /usr/lib (which contains libcuda.so.1 from the NVIDIA
# driver package) is on LD_LIBRARY_PATH so TensorFlow can find it.
export LD_LIBRARY_PATH="/usr/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
HOOK

# --- deactivation hook ---
cat > "$DEACTIVATE_DIR/nightcore_libcuda_path.sh" << 'HOOK'
# Added by setup_conda_libcuda.sh
export LD_LIBRARY_PATH="$(printf '%s' "${LD_LIBRARY_PATH}" \
    | sed 's|^/usr/lib:||; s|:/usr/lib$||; s|^/usr/lib$||')"
HOOK

chmod +x "$ACTIVATE_DIR/nightcore_libcuda_path.sh" \
         "$DEACTIVATE_DIR/nightcore_libcuda_path.sh"

echo ""
echo "Activation hooks written to:"
echo "  $ACTIVATE_DIR/nightcore_libcuda_path.sh"
echo "  $DEACTIVATE_DIR/nightcore_libcuda_path.sh"
echo ""
echo "Now run:"
echo "  conda deactivate && conda activate nightcore-analyzer"
echo "  python verify_cuda.py"
