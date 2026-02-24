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
# Shell support:
#   Both bash/zsh (.sh hooks) and fish (.fish hooks) are written
#   so the fix works regardless of which shell you use.
#
# Usage (run ONCE, with the env active):
#   conda activate nightcore-analyzer
#   bash setup_conda_libcuda.sh
#   conda deactivate && conda activate nightcore-analyzer
#   python verify_cuda.py
#
# Fish users — one-off manual test (fish syntax):
#   set -gx LD_LIBRARY_PATH /usr/lib:$LD_LIBRARY_PATH
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

# ── bash/zsh activation hook ──────────────────────────────────────────────────
cat > "$ACTIVATE_DIR/nightcore_libcuda_path.sh" << 'HOOK'
# Added by setup_conda_libcuda.sh (bash/zsh)
# Ensures /usr/lib (which contains libcuda.so.1 from the NVIDIA
# driver package) is on LD_LIBRARY_PATH so TensorFlow can find it.
export LD_LIBRARY_PATH="/usr/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
HOOK

# ── bash/zsh deactivation hook ────────────────────────────────────────────────
cat > "$DEACTIVATE_DIR/nightcore_libcuda_path.sh" << 'HOOK'
# Added by setup_conda_libcuda.sh (bash/zsh)
export LD_LIBRARY_PATH="$(printf '%s' "${LD_LIBRARY_PATH}" \
    | sed 's|^/usr/lib:||; s|:/usr/lib$||; s|^/usr/lib$||')"
HOOK

# ── fish activation hook ──────────────────────────────────────────────────────
# Conda sources .fish scripts directly when the user's shell is fish,
# so this is the reliable path for fish users.
cat > "$ACTIVATE_DIR/nightcore_libcuda_path.fish" << 'HOOK'
# Added by setup_conda_libcuda.sh (fish)
# Ensures /usr/lib is at the front of LD_LIBRARY_PATH.
set -gx LD_LIBRARY_PATH /usr/lib:$LD_LIBRARY_PATH
HOOK

# ── fish deactivation hook ────────────────────────────────────────────────────
cat > "$DEACTIVATE_DIR/nightcore_libcuda_path.fish" << 'HOOK'
# Added by setup_conda_libcuda.sh (fish)
# Remove the /usr/lib: prefix that was added on activation.
if set -q LD_LIBRARY_PATH
    set -gx LD_LIBRARY_PATH (string replace --regex '^/usr/lib:|:/usr/lib$|^/usr/lib$' '' "$LD_LIBRARY_PATH")
    # If nothing left after stripping, unexport the variable entirely.
    if test -z "$LD_LIBRARY_PATH"
        set -ge LD_LIBRARY_PATH
    end
end
HOOK

chmod +x \
    "$ACTIVATE_DIR/nightcore_libcuda_path.sh" \
    "$DEACTIVATE_DIR/nightcore_libcuda_path.sh"
# .fish files are not executed directly, no chmod needed

echo ""
echo "Activation hooks written:"
echo "  bash/zsh: $ACTIVATE_DIR/nightcore_libcuda_path.sh"
echo "  fish:     $ACTIVATE_DIR/nightcore_libcuda_path.fish"
echo "  bash/zsh: $DEACTIVATE_DIR/nightcore_libcuda_path.sh"
echo "  fish:     $DEACTIVATE_DIR/nightcore_libcuda_path.fish"
echo ""
echo "Now run:"
echo "  conda deactivate && conda activate nightcore-analyzer"
echo "  python verify_cuda.py"
echo ""
echo "If you use fish shell and the reactivation still doesn't work, test with:"
echo "  set -gx LD_LIBRARY_PATH /usr/lib:\$LD_LIBRARY_PATH"
echo "  python verify_cuda.py"
