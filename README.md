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
5. [Step 1: Verify Dependencies](#step-1-verify-dependencies)
6. [Project Structure](#project-structure)
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

1. Both files (nightcore + FLAC source) are loaded as mono float32 at 22 050 Hz. Leading and trailing silence is trimmed automatically before any analysis or duration measurement (configurable threshold; can be disabled).
2. Both files are sliced into overlapping fixed-length windows (default: 10 s windows, 5 s hop = 50% overlap).
3. Both audio files produce a pitch-shift estimate via CQT chromagram cross-correlation (librosa). Optionally refined by MELODIA (essentia, if installed). Each window independently produces a tempo estimate (librosa onset detection).
4. Energy-gated filtering discards silent or low-energy windows whose RMS energy is more than 40 dB below the peak window.
5. Median/bootstrap consensus over all windows produces the final ratios, rejecting localised artifact windows as outliers. 2000 bootstrap resamples generate 95% confidence intervals for both tempo and pitch.
6. Inter-beat-interval (IBI) ratio — full-signal beat tracking at fine hop resolution (hop=64, ≈1.45 ms) produces a secondary speed ratio at ~0.01% precision, 10–30× finer than the windowed BPM ratio. This is the recommended speed factor for `sox`.
7. Waveform cross-correlation provides an independent speed estimate with a quality score 0–1; used in verification to flag gross mismatches. Results below quality 0.30 are discarded automatically.

**Output:**
- Tempo ratio (nightcore ÷ source) — windowed BPM median
- IBI ratio — beat-timestamp-based, ~0.01% precision (primary recommendation when available)
- Pitch ratio (nightcore ÷ source)
- 95% confidence intervals for both tempo and pitch
- Raw file durations in seconds (ms precision) after silence trim, plus duration ratio and inverse ratio (independent of beat detection — computed directly from sample counts)
- Alignment classification: *pure nightcore* (tempo and pitch co-shifted) | *independently pitch-shifted* (extra pitch on top of speed-up) | *time-stretched* (tempo shifted, pitch unchanged) | *ambiguous* (CIs overlap)
- Rubber Band CLI parameters for reconstruction (`rubberband --time X --pitch Y nightcore.flac reconstructed.flac`)
- Cross-correlation ratio + quality score (verification step only)

---

## Prerequisites

### System requirements

| Requirement | Notes |
|---|---|
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

Choose **one** of the two approaches below. Option A (pip + venv) is simpler. Option B (Conda) is an alternative for users who prefer conda-managed environments.

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

---

### Option B — Conda

```bash
# 1. Create the environment from the spec file
conda env create -f environment.yml

# 2. Activate it
conda activate nightcore-analyzer
```

---

## Step 1: Verify Dependencies

Run the verification script to confirm all required packages are importable:

```bash
python verify_cuda.py
```

Expected output (abbreviated):

```
============================================================
  Nightcore Analyzer — Environment Verification
============================================================

[1] Python
    [PASS] Version (3.10+ required): 3.11.x

[2] NumPy
    [PASS] Import + version: 1.26.x
...
============================================================
  ALL CHECKS PASSED

  Safe to proceed to Step 2 — Core Analysis Module.
============================================================
```

---

## Project Structure

```
nightcore-to-flac-analyzer/
├── verify_cuda.py              # Step 1: environment verification
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
│   ├── pitch.py                # chromagram xcorr pitch-shift estimation (+ optional MELODIA)
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
| 1 | ✅ Complete | Environment setup, dependency verification |
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
result, running a spectral analysis to surface any remaining differences,
and optionally fixing any digital clipping to prevent crackling on playback.

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

At startup you choose one of four modes:

```
  [f]  Full suite  (speed comparison → create HQNC → verification → spectral)
  [s]  Speed comparison  (+ optional HQNC creation + optional spectral)
  [a]  Spectral analysis  (standalone two-file comparison)
  [l]  Loudness adjustment  (clipping detection + true peak limiter / gain)
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

4. **Step 3/4** — Optional spectral analysis on HQNC vs NCOG, reporting:
   - Brightness / low-pass filtering
   - High-frequency rolloff (treble cut — common in MP3 128k rips)
   - Dynamic range / compression
   - Frequency-band breakdown (sub-bass → brilliance)
   - Reverb tail difference
   - Duration mismatch (fade-in/out, different edits)
   - Format quality note (lossless vs lossy) with a reverse-order warning

5. **Step 4/4** — Optional loudness adjustment (see Mode `[l]` below).
   Targets the HQNC if one was created; falls back to the HQ source.

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

#### Mode `[l]` — Loudness adjustment

Detects whether a file has digital clipping (peak ≥ 0 dBFS) that can cause
crackling on playback, and offers to fix it with **minimal impact on quality
and dynamic range**.

Two correction methods are offered each run:

| Method | How it works | Quality impact |
|--------|-------------|----------------|
| **`[l]` True Peak Limiter** *(recommended)* | ffmpeg `alimiter` — only attenuates samples that exceed the ceiling; everything below is left completely unchanged | Surgical: only the clipped peaks are touched; dynamic range is fully preserved |
| **`[g]` Gain Reduction** | sox/ffmpeg uniform level shift by N dB | Simple but lowers the entire signal — quiet passages get quieter even though they were fine |

The limiter is preferred: a file that is 0.3 dBFS over does not need to be
pulled down 0.3 dB across the board — the limiter just shaves those specific
peaks off.

**Usage flow:**

1. The tool reports the peak level of the file: e.g. `Peak: +0.23 dBFS  !! CLIPPING`
2. You choose a method (`[l]` or `[g]`) and confirm the parameters (ceiling or dB amount).
3. The adjusted file is written as `ADJ1` next to the source:
   ```
   Song [Nightcore].flac  →  Song [Nightcore] ADJ1.flac
   ```
4. The tool re-measures the peak and reports the result.
5. If still clipping (or you want another pass), answer `[y]` to continue:
   ```
   Song [Nightcore] ADJ1.flac  →  Song [Nightcore] ADJ2.flac
   ```

The standalone `[l]` mode prompts for a single file.  The same loop is also
offered as **Step 4/4** at the end of the Full Suite, targeting the HQNC.

**Requirements:** `ffmpeg` (for the limiter) and/or `sox` (for gain reduction)
must be on `PATH`.

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

**All available flags:**

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `--nightcore` | `-n` | *(required)* | Nightcore audio file (FLAC, MP3, WAV, …) |
| `--source` | `-s` | *(required)* | Source audio file (original, ideally FLAC) |
| `--output` | `-o` | stdout | Write JSON results to this file instead of printing |
| `--window` | — | `10.0` s | Analysis window length in seconds |
| `--hop` | — | `5.0` s | Hop between consecutive windows; less than `--window` gives overlap |
| `--energy-gate` | — | `-40.0` dB | Discard windows whose RMS is more than this many dB below the peak window |
| `--silence-strip-db` | — | `60.0` dB | Top-dB threshold for librosa silence trimming; frames more than this dB below the peak are considered silent |
| `--no-silence-strip` | — | off | Disable leading/trailing silence removal entirely |
| `--src-trim-sec` | — | `0.0` | Manually trim this many seconds from the start of the source file (takes priority over `--auto-align`) |
| `--auto-align` | — | off | Attempt automatic intro-offset detection via RMS envelope cross-correlation. Unreliable on repetitive or dense music; `--src-trim-sec` is preferred when the offset is known |
| `--quiet` | `-q` | off | Suppress progress output; errors still go to stderr |

Output is a JSON object containing all `AnalysisResult` fields: ratios, confidence intervals, IBI ratio, duration data, classification, and Rubber Band parameters.

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

### 3. Remove essentia data (if installed)

If you installed essentia for MELODIA pitch refinement, no model weights are
downloaded automatically — essentia ships its algorithms in the package itself.
No additional cleanup is needed.

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


```

After these steps, no files from this project remain on your system.
