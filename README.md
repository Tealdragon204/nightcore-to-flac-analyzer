# Nightcore Audio Analysis Tool

Algorithmic extraction of the precise **tempo ratio** and **pitch ratio** between a nightcore track and its FLAC source, with sub-decimal accuracy, to enable high-quality reconstruction via Rubber Band.

Nightcore tracks (2010–2014 era YouTube rips, low quality) are speed-shifted versions of FLAC originals. Traditional nightcore pitches up as a byproduct of tempo increase — both shift by the same factor. Some tracks may additionally have independent pitch shifting on top. This tool detects and disambiguates both cases.

---

## Table of Contents

1. [How It Works](#how-it-works)
2. [Prerequisites](#prerequisites)
3. [Clone the Repository](#clone-the-repository)
4. [Environment Setup](#environment-setup)
   - [Option A — pip + venv (recommended for most users)](#option-a--pip--venv)
   - [Option B — Conda](#option-b--conda)
5. [Step 1: Verify CUDA and Dependencies](#step-1-verify-cuda-and-dependencies)
6. [Troubleshooting CUDA](#troubleshooting-cuda)
7. [Project Structure](#project-structure)
8. [Development Roadmap](#development-roadmap)
9. [Usage](#usage)
   - [Interactive Workflow (recommended)](#interactive-workflow-recommended)
   - [GUI mode](#gui-mode)
   - [CLI mode](#cli-mode)
10. [Uninstall and Delete Cloned Files](#uninstall-and-delete-cloned-files)

---

## How It Works

Input files are dirty by design: leading/trailing silence, localised artifacts (loud scratches, broken audio glitches at intros/outros), poor mastering. The algorithm is robust to this without requiring pre-cleaned files.

**Windowed consensus pipeline:**

1. Both files (nightcore + FLAC source) are sliced into overlapping windows internally.
2. Each window independently produces a pitch estimate (CREPE via TensorFlow/CUDA) and a tempo estimate (librosa onset detection).
3. Energy-gated filtering discards silent or low-confidence windows.
4. RANSAC / median consensus over all windows produces the final ratios, naturally rejecting localised artifact windows as outliers.

**Output:**
- Tempo ratio (nightcore tempo ÷ source tempo)
- Pitch ratio (nightcore pitch ÷ source pitch)
- Confidence intervals for both
- Alignment classification: *pure nightcore* (co-shifted) vs. *independently pitch-shifted on top* vs. *time-stretched without pitch shift*
- Exact `pyrubberband` parameters for reconstruction

---

## Prerequisites

### System requirements

| Requirement | Notes |
|---|---|
| **NVIDIA GPU** | Any CUDA-capable card; a 4090 is the target. |
| **NVIDIA driver** | 525+ recommended for CUDA 12.x. Check: `nvidia-smi` |
| **CUDA Toolkit** | 11.8 or 12.x. `tensorflow[and-cuda]` bundles its own CUDA wheels so a system install is not strictly required for TF, but the driver must be present. |
| **Python 3.10–3.12** | 3.11 recommended. |
| **rubberband binary** | Required by pyrubberband at runtime. |
| **sox** | Required by the interactive workflow to create the sped-up HQNC file. |

### Install the rubberband binary

```bash
# Arch / Manjaro
sudo pacman -S rubberband

# Ubuntu / Debian
sudo apt install rubberband-cli

# Fedora
sudo dnf install rubberband

# macOS (Homebrew)
brew install rubberband
```

Verify it is on PATH:

```bash
rubberband --version
```

### Install sox

sox is used by the interactive workflow to speed up the HQ source file and
create the HQNC (high-quality nightcore) file.

```bash
# Arch / Manjaro
sudo pacman -S sox

# Ubuntu / Debian
sudo apt install sox

# Fedora
sudo dnf install sox

# macOS (Homebrew)
brew install sox
```

Verify it is on PATH:

```bash
sox --version
```

> **Note:** sox is only needed if you use the interactive workflow (`python -m nightcore_analyzer.workflow`) and choose to create an HQNC file.  The CLI and GUI analysis modes do not require it.

---

## Clone the Repository

```bash
git clone https://github.com/Tealdragon204/nightcore-to-flac-analyzer.git
cd nightcore-to-flac-analyzer
```

If you want a specific branch (e.g., the active development branch):

```bash
git clone -b claude/nightcore-audio-analysis-R1wZ2 \
    https://github.com/Tealdragon204/nightcore-to-flac-analyzer.git
cd nightcore-to-flac-analyzer
```

---

## Environment Setup

Choose **one** of the two approaches below. Option A (pip + venv) is simpler. Option B (Conda) is more robust on Arch-derivative systems where CUDA library paths can be fragile.

---

### Option A — pip + venv

```bash
# 1. Create a virtual environment inside the project directory
python3.11 -m venv .venv

# 2. Activate it
source .venv/bin/activate          # Linux / macOS
# .venv\Scripts\activate           # Windows

# 3. Upgrade pip
pip install --upgrade pip

# 4. Install all dependencies
pip install -r requirements.txt
```

> **Note:** `tensorflow[and-cuda]` (included in `requirements.txt`) ships its own cuDNN and CUDA runtime as Python wheels. You do **not** need a system-level CUDA Toolkit install for TensorFlow itself, but the NVIDIA driver must be installed and `nvidia-smi` must work.

---

### Option B — Conda

```bash
# 1. Create the environment from the spec file
conda env create -f environment.yml

# 2. Activate it
conda activate nightcore-analyzer

# 3. Install CREPE — must be done separately.
#    setuptools 81+ removed pkg_resources entirely; CREPE's 2020-era setup.py
#    imports it. Pin setuptools<81, then install CREPE without build isolation:
pip install "setuptools<81"
pip install --no-build-isolation crepe

# 4. (Optional) If TensorFlow still cannot see the GPU after conda setup,
#    install the CUDA toolkit inside the conda env:
conda install -c nvidia cuda-toolkit cudnn
```

---

## Step 1: Verify CUDA and Dependencies

**This step is mandatory before any analysis.** Run the verification script:

```bash
python verify_cuda.py
```

Expected output on a working system (abbreviated):

```
============================================================
  Nightcore Analyzer — Environment & CUDA Verification
============================================================

[1] Python
    [PASS] Version (3.10+ required): 3.11.x

[2] NumPy
    [PASS] Import + version: 1.26.x
...
[10] TensorFlow — CRITICAL: GPU / CUDA checks
    [PASS] Import + version: 2.15.x
    [PASS] Built with CUDA
    [PASS] GPU device detection (CRITICAL): 1 GPU(s): ['/physical_device:GPU:0']
    [PASS] GPU compute test (matmul on /GPU:0): Matrix multiply on GPU:0 — output shape (2, 2)

[11] CREPE (TensorFlow-based pitch detector)
    [PASS] Import

============================================================
  ALL 15 CHECKS PASSED

  GPU and CUDA: VERIFIED
  Safe to proceed to Step 2 — Core Analysis Module.
============================================================
```

**Do not proceed to Step 2 (or beyond) until all checks pass, particularly the GPU detection check.** CREPE pitch extraction running on CPU is impractically slow for the windowed consensus pipeline.

---

## Troubleshooting CUDA

### `nvidia-smi` not found — NVIDIA driver not installed

This is the most common cause of the GPU CRITICAL failure. TensorFlow (even
`tensorflow[and-cuda]`, which bundles its own CUDA toolkit) still requires the
NVIDIA driver's user-space library `libcuda.so.1` to be present on the system.
That library ships with the driver package, not with any conda or pip install.

**Check first:**

```bash
nvidia-smi
```

If you get `command not found`, install the driver:

```bash
# Arch / Manjaro
sudo pacman -S nvidia nvidia-utils
# Then reboot — the kernel module must be loaded

# Ubuntu / Debian (pick the version matching your kernel)
sudo apt install nvidia-driver-535
# Then reboot

# Fedora
sudo dnf install akmod-nvidia
# Then reboot
```

After rebooting, confirm the driver is loaded:

```bash
nvidia-smi          # should show your GPU, driver version, and CUDA version
lsmod | grep nvidia # should list nvidia, nvidia_modeset, etc.
```

Then re-run `python verify_cuda.py`.

---

### `nvidia-smi` works but TensorFlow cannot see the GPU

`tensorflow[and-cuda]` bundles cuDNN, cuFFT, cuBLAS etc. as pip wheels, but
it still calls `dlopen("libcuda.so.1")` at import time to reach the NVIDIA
driver. That file comes from the system driver package, not from any pip or
conda install. When using a **conda environment**, conda prepends its own
`$CONDA_PREFIX/lib` to `LD_LIBRARY_PATH`, which can push `/usr/lib` (where
`libcuda.so.1` lives on Arch/Garuda) out of TF's search window.

**Permanent fix — use the included activation hook script (recommended):**

```bash
conda activate nightcore-analyzer
bash setup_conda_libcuda.sh
conda deactivate && conda activate nightcore-analyzer
python verify_cuda.py
```

This writes a conda activation script that prepends `/usr/lib` to
`LD_LIBRARY_PATH` every time the environment is activated. It is automatically
undone on `conda deactivate`.

**Quick one-off test (no permanent change):**

```bash
# bash / zsh
export LD_LIBRARY_PATH=/usr/lib:$LD_LIBRARY_PATH
python verify_cuda.py
```

```fish
# fish shell — export/= syntax does NOT work; use set -gx
set -gx LD_LIBRARY_PATH /usr/lib:$LD_LIBRARY_PATH
python verify_cuda.py
```

**Verify `libcuda.so.1` is present first:**

```bash
find /usr -name "libcuda.so.1" 2>/dev/null
# Expected on Arch/Garuda: /usr/lib/libcuda.so.1
# Expected on Ubuntu/Debian: /usr/lib/x86_64-linux-gnu/libcuda.so.1
```

Adjust the path in `setup_conda_libcuda.sh` if your distro puts the library
somewhere else.

**Also check for a conda cuda-toolkit conflict:**

If you previously ran `conda install cuda-toolkit` or
`conda install -c nvidia cudnn`, you have both conda-managed CUDA 13 and
pip-managed CUDA 12 in the same environment — a known source of subtle
failures. The cleanest fix is to recreate the environment from scratch:

```bash
conda deactivate
conda env remove -n nightcore-analyzer
conda env create -f environment.yml
conda activate nightcore-analyzer
pip install "setuptools<81"
pip install --no-build-isolation crepe
bash setup_conda_libcuda.sh
conda deactivate && conda activate nightcore-analyzer
python verify_cuda.py
```

Do **not** run `conda install cuda-toolkit` or `conda install cudnn` into
this environment — `tensorflow[and-cuda]` supplies its own CUDA 12 wheels.

### TensorFlow version mismatch

The bundled CUDA wheel approach requires TF ≥ 2.13. If you have an older TF installed:

```bash
pip install --upgrade "tensorflow[and-cuda]>=2.15.0,<2.17.0"
```

### `libcudnn.so.8` not found

```bash
# Check what cuDNN is available in the TF wheel
python -c "import tensorflow as tf; print(tf.sysconfig.get_lib())"
ls $(python -c "import tensorflow as tf; print(tf.sysconfig.get_lib())")

# If missing, install cuDNN separately via conda:
conda install -c nvidia cudnn
```

### Multiple GPUs / wrong device selected

TensorFlow uses GPU 0 by default. To pin to a specific device:

```bash
export CUDA_VISIBLE_DEVICES=0    # use only GPU 0 (the 4090)
python verify_cuda.py
```

### CREPE model weights

CREPE downloads model weights on first use (~84 MB) from the internet. Ensure network access is available during first analysis run. Weights are cached in `~/.crepe/` (Linux/macOS).

---

## Project Structure

```
nightcore-to-flac-analyzer/
├── verify_cuda.py              # Step 1: environment + CUDA verification
├── setup_conda_libcuda.sh      # one-time conda LD_LIBRARY_PATH fix (fish + bash)
├── requirements.txt            # pip dependency specification
├── environment.yml             # Conda environment specification
├── .gitignore
├── LICENSE
├── README.md
│
├── nightcore_analyzer/
│   ├── __init__.py             # package root; exposes run(), AnalysisResult, export, session
│   ├── __main__.py             # python -m nightcore_analyzer → launches PyQt6 GUI
│   ├── cli.py                  # python -m nightcore_analyzer.cli  (fully working)
│   ├── workflow.py             # python -m nightcore_analyzer.workflow  (interactive guided mode)
│   ├── pipeline.py             # top-level orchestrator
│   ├── io.py                   # audio loading, resampling, windowing, energy gating
│   ├── pitch.py                # CREPE per-window F0 estimation
│   ├── tempo.py                # librosa per-window BPM estimation
│   ├── consensus.py            # median/bootstrap ratio, CI, classification, RB params
│   ├── spectral.py             # spectral comparison (brightness, EQ, compression, reverb)
│   ├── session.py              # session persistence (last directory, parameters)
│   ├── export.py               # JSON / CSV result export
│   └── gui/
│       ├── __init__.py         # exposes MainWindow
│       ├── main_window.py      # PyQt6 main window + file pickers + menu
│       ├── worker.py           # QThread worker wrapping the pipeline
│       ├── histogram_widget.py # per-window pitch/tempo histograms (matplotlib)
│       └── results_panel.py    # ratios, CIs, classification, Rubber Band params
```

---

## Development Roadmap

| Step | Status | Description |
|------|--------|-------------|
| 1 | ✅ Complete | Environment setup, CUDA verification |
| 2 | ✅ Complete | Core analysis module — CLI-testable windowed pipeline |
| 3 | ✅ Complete | PyQt6 GUI shell — file pickers, run button, worker thread |
| 4 | ✅ Complete | Results visualisation — per-window histograms |
| 5 | ✅ Complete | Output panel — ratios, CIs, classification, Rubber Band params |
| 6 | ✅ Complete | QoL — session persistence, JSON/CSV export |
| 7 | ✅ Complete | Interactive workflow — speed comparison, HQNC creation via sox, spectral analysis |

---

## Usage

### Interactive Workflow (recommended)

The interactive workflow guides you through the full process end-to-end:
comparing speeds, creating a high-quality nightcore FLAC, verifying the
result, and running a spectral analysis to surface any remaining differences.

```bash
python -m nightcore_analyzer.workflow [HQ_FILE] [NCOG_FILE]
```

Both file arguments are optional — the tool prompts for any missing paths.
You can type a path, paste one, or **drag-and-drop a file from your file
manager directly into the terminal** — surrounding quotes are stripped
automatically, so all three forms work:

```
Path to HQ source: /home/user/Song.flac
Path to HQ source: '/home/user/Song.flac'
Path to HQ source: "/home/user/Song.flac"
```

At startup you choose one of three modes:

```
  [f]  Full suite  (speed comparison → create HQNC → verification → spectral)
  [s]  Speed comparison  (+ optional HQNC creation + optional spectral)
  [a]  Spectral analysis  (standalone two-file comparison)
  [e]  Exit
```

**Three file roles:**

| Role | Description |
|---|---|
| **HQ** | Original high-quality source (FLAC/WAV) |
| **NCOG** | The nightcore edit you want to match (typically a lossy MP3 rip) |
| **HQNC** | HQ sped up to match NCOG — created by the workflow via `sox` |

---

#### Mode `[f]` — Full suite

The recommended end-to-end flow:

1. **Step 1/3** — Analyses HQ vs NCOG to determine the speed factor and pitch ratio.
   Three estimators are reported:

   | Estimator | Precision | Notes |
   |-----------|-----------|-------|
   | **BPM ratio** | ~0.1–0.3% | windowed median, always available |
   | **IBI ratio** | ~0.01% | beat timestamps at hop=64; 10–30× more precise |

   The **IBI ratio** is used as the speed factor for the `sox` command (BPM shown as
   fallback).  Also prints the *inverse ratio* (speed needed if files are accidentally
   swapped) so you can sanity-check without re-running.

2. **Create HQNC?** — Prompts `[Y/n/e]`.  If yes, runs:
   ```
   sox 'Song.flac' 'Song [Nightcore].flac' speed 1.XXXXXX
   ```
   The output file is placed alongside HQ with `[Nightcore]` appended to the stem.

3. **Step 2/3 — Verification** (nightcore ↔ nightcore) — Analyses HQNC vs
   NCOG and interprets the result:
   - Tempo and pitch both ≈ 1:1 → "Files are essentially identical ✓"
   - Tempo matches but pitch differs → "Additional pitch shift detected in NCOG"
   - Tempo still differs → **retry loop**: the tool offers to re-run `sox`
     with a corrected cumulative speed factor.

   Three estimators are shown for each verification attempt:

   | Estimator | Tolerance | Notes |
   |-----------|-----------|-------|
   | **BPM ratio** | ±2% | windowed median |
   | **IBI ratio** | ±0.5% | beat timestamps, tighter pass/fail |
   | **Xcorr ratio** | — | waveform cross-correlation, quality score 0–1 |

   **Retry loop details:** if the IBI ratio (or BPM ratio when IBI is unavailable)
   is outside tolerance the tool shows the residual error and a corrected factor, then
   prompts:
   ```
   Speed is still off by +4.55%.
   Corrected factor (IBI): 1.250000 × 1.045455 = 1.306819×
   Would create: Song [Nightcore] UPD1.flac
   Re-run sox with corrected factor? [Y/n/e]:
   ```
   Answering **Y** runs sox again, names the output `UPD1` (then `UPD2`,
   `UPD3`, …), and re-verifies.  The loop continues until the result is
   within tolerance or you skip to Step 3.  The latest `UPDn` file is
   automatically used as the reference for spectral analysis.

   Also prints a quality note: HQNC (lossless FLAC) vs NCOG (lossy MP3).

4. **Step 3/3** — Optional spectral analysis on HQNC vs NCOG, reporting:
   - Brightness / low-pass filtering
   - High-frequency rolloff (treble cut — common in MP3 128k rips)
   - Dynamic range / compression
   - Frequency-band breakdown (sub-bass → brilliance)
   - Reverb tail difference
   - Duration mismatch (fade-in/out, different edits)
   - Format quality note (lossless vs lossy) with a reverse-order warning

---

#### Mode `[s]` — Speed comparison

Runs the HQ vs NCOG analysis, prints the speed factor and pitch ratio, then
optionally creates HQNC and optionally runs spectral analysis.  Useful when
you already have HQNC and want a quick check, or just want the sox command.

---

#### Mode `[a]` — Spectral analysis

Prompts for any two audio files and runs the spectral comparison report
standalone — no pipeline analysis involved.  Useful for comparing any two
audio files directly.

---

### GUI mode

```bash
python -m nightcore_analyzer
```

Opens the PyQt6 interface.  Select the nightcore and source files with the
file pickers, adjust analysis parameters if needed, then click **Run Analysis**.
Results appear in the **Results** tab (ratios, CIs, classification, Rubber Band
command) and the **Histograms** tab (per-window distributions).  Use
**File → Save results as JSON/CSV** to export.

> **Requirements:** PyQt6 must be installed (`pip install PyQt6`).  On headless
> servers the GUI is not available; use the workflow CLI or CLI mode instead.

---

### CLI mode

For scripting or headless use:

```bash
# Always quote paths — spaces and parentheses (common in music filenames) break
# unquoted shell arguments, especially in fish where (...) is command substitution.
python -m nightcore_analyzer.cli \
    --nightcore "/path/to/nightcore track.mp3" \
    --source    "/path/to/original (Radio Edit).flac" \
    --output    results.json
```

Run `python -m nightcore_analyzer.cli --help` for all options.

---

## Uninstall and Delete Cloned Files

Follow these steps to completely remove the tool and all associated files from your system.

### 1. Deactivate the environment (if active)

```bash
# pip / venv
deactivate

# Conda
conda deactivate
```

### 2. Remove the Python environment

```bash
# Option A — pip + venv
# The .venv directory lives inside the project folder and is removed with it in step 4.
# Nothing else to do here unless you installed globally (not recommended).

# Option B — Conda
conda env remove -n nightcore-analyzer
# Verify removal:
conda env list
```

### 3. Remove CREPE model weights

CREPE downloads ~84 MB of model weights on first use and caches them here:

```bash
# Linux / macOS
rm -rf ~/.crepe

# Windows
rmdir /s /q "%USERPROFILE%\.crepe"
```

### 4. Delete the cloned repository

```bash
# Navigate to the parent directory of the cloned folder
cd ..

# Remove the project directory and everything inside it
rm -rf nightcore-to-flac-analyzer
```

This deletes:
- All source code
- The `.venv/` virtual environment (if you used Option A)
- Any cached results, logs, or audio files stored inside the project

### 5. Verify removal

```bash
# Confirm the directory is gone
ls nightcore-to-flac-analyzer   # should return: No such file or directory

# Confirm the conda environment is gone (if applicable)
conda env list                  # nightcore-analyzer should not appear

# Confirm CREPE weights are gone
ls ~/.crepe                     # should return: No such file or directory
```

After these steps, no files from this project remain on your system.
