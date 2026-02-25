"""
Interactive three-mode workflow for the Nightcore Analyzer.

Usage
-----
    python -m nightcore_analyzer.workflow [NCOG_FILE] [HQ_FILE]

Both positional arguments are optional; the tool prompts for any missing
paths interactively.  NCOG (the nightcore edit) is always requested first,
then HQ (the original high-quality source).

Modes
-----
[f]  Full suite
     NCOG vs HQ  →  create HQNC via sox  →  HQNC vs NCOG  →  spectral analysis

[s]  Speed comparison
     NCOG vs HQ  →  optional create HQNC  →  optional spectral analysis

[a]  Spectral analysis (standalone)
     Compare any two audio files spectrally.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

from . import pipeline
from . import spectral as spec


# ── I/O helpers ───────────────────────────────────────────────────────────────

def _prompt_choice(question: str, options: str = "yne", default: str = "") -> str:
    """
    Ask *question* and loop until the user enters one char from *options*.
    'e' always exits immediately.  Returns the character lower-cased.

    *default* (if given) is accepted on a bare Enter and shown uppercase in
    the prompt; all other options are shown lowercase — e.g. "Y/n/e" means
    Y is the default.
    """
    parts = []
    for c in options.lower():
        parts.append(c.upper() if c == default.lower() else c.lower())
    opts_display = "/".join(parts)

    while True:
        raw = input(f"{question} [{opts_display}]: ").strip().lower()
        if raw == "e":
            print("Exiting.")
            sys.exit(0)
        if not raw and default and default.lower() in options.lower():
            return default.lower()
        if raw in options.lower():
            return raw
        print(f"  Please type one of: {', '.join(c.upper() for c in options)}")


def _prompt_file(label: str, existing: Optional[str] = None) -> Path:
    """Return a validated Path for *label*, re-using *existing* if supplied."""
    if existing:
        p = Path(existing)
        if p.is_file():
            return p
        print(f"  File not found: {existing}")

    while True:
        raw = input(f"Path to {label}: ").strip()
        if not raw:
            continue
        p = Path(raw)
        if p.is_file():
            return p
        print(f"  File not found: {p}")


def _hr(char: str = "─", width: int = 57) -> None:
    print(char * width)


# ── sox helper ────────────────────────────────────────────────────────────────

def _make_hqnc_path(hq: Path) -> Path:
    """Return e.g. /path/Song [Nightcore].flac from /path/Song.flac"""
    return hq.with_name(hq.stem + " [Nightcore]" + hq.suffix)


def _make_hqnc_path_v(hq: Path, version: int) -> Path:
    """
    Return a versioned HQNC path.

    version 0  →  ``Song [Nightcore].flac``
    version 1  →  ``Song [Nightcore] UPD1.flac``
    version 2  →  ``Song [Nightcore] UPD2.flac``
    """
    if version == 0:
        return hq.with_name(hq.stem + " [Nightcore]" + hq.suffix)
    return hq.with_name(hq.stem + f" [Nightcore] UPD{version}" + hq.suffix)


def _run_sox(src: Path, dst: Path, speed: float) -> None:
    if not shutil.which("sox"):
        print(
            "\n  ERROR: sox not found on PATH.\n"
            "  Install it:  sudo apt install sox   (Debian/Ubuntu)\n"
            "               brew install sox        (macOS)\n"
        )
        raise SystemExit(1)
    print(f"\n  Running: sox '{src}' '{dst}' speed {speed:.6f}")
    subprocess.run(["sox", str(src), str(dst), "speed", f"{speed:.6f}"], check=True)
    print(f"  Created: {dst}")


# ── pipeline wrapper / result display ─────────────────────────────────────────

_NEAR_UNITY = 0.02   # |ratio − 1| below this → "essentially the same"
_PITCH_TEMPO_TOLERANCE = 0.02  # extra pitch shift threshold beyond tempo ratio


def _run_pipeline(nightcore: Path, source: Path, step_label: str) -> "pipeline.AnalysisResult":
    print()
    _hr("─")
    print(f"  {step_label}")
    _hr("─")
    print(f"  Nightcore : {nightcore.name}")
    print(f"  Source    : {source.name}")
    print()

    result = pipeline.run(str(nightcore), str(source), log=lambda m: print(f"  {m}"))
    return result


def _print_speed_result(result: "pipeline.AnalysisResult", hq: Path, ncog: Path) -> None:
    """Print the speed/pitch summary and the recommended sox command."""
    tr = result.tempo_ratio
    pr = result.pitch_ratio

    print()
    _hr("═")
    print("  SPEED COMPARISON RESULTS")
    _hr("═")
    print(f"  Speed factor  : {tr:.6f}×")
    print(f"  Pitch ratio   : {pr:.6f}")
    print(f"  Classification: {result.classification}")

    # CIs
    lo, hi = result.tempo_ci
    print(f"  Tempo 95% CI  : [{lo:.4f}, {hi:.4f}]")
    lo_p, hi_p = result.pitch_ci
    print(f"  Pitch 95% CI  : [{lo_p:.4f}, {hi_p:.4f}]")

    # BPMs
    if result.nc_median_bpm and result.src_median_bpm:
        print(
            f"  Median BPMs   : NCOG {result.nc_median_bpm:.1f} BPM  |"
            f"  HQ {result.src_median_bpm:.1f} BPM"
        )

    # Pitch vs tempo note
    pt_diff = abs(pr - tr) / tr if tr > 0 else 0
    if pt_diff > _PITCH_TEMPO_TOLERANCE:
        st_extra = -12 * __import__("math").log2(pr / tr)
        print(
            f"\n  Note: Pitch ratio ({pr:.4f}) differs from tempo ratio ({tr:.4f}) "
            f"by {pt_diff * 100:.1f}%.\n"
            f"  This suggests an additional pitch shift of ~{st_extra:+.2f} semitones\n"
            f"  was applied to NCOG on top of the speed-up."
        )
    else:
        print("\n  Pitch and tempo ratios agree — consistent with a pure speed-up.")

    # Warnings
    if result.warnings:
        print()
        for w in result.warnings:
            # wrap long warnings at 80 chars
            print(f"  Warning: {w[:200]}")

    # ── Sanity check: inverse direction (if files are swapped) ────────────────
    # Show this BEFORE the sox command so the reader evaluates it first.
    print()
    if tr > 0:
        inv = 1.0 / tr
        if abs(tr - 1.0) < _NEAR_UNITY:
            print("  If files are swapped: speed would also be ~1.000× (no difference).")
        elif inv < 1.0:
            print(
                f"  If files are swapped: speed = 1 / {tr:.4f} = {inv:.6f}×  "
                f"(would SLOW DOWN HQ — files appear to be in the correct order)"
            )
        else:
            print(
                f"  If files are swapped: speed = 1 / {tr:.4f} = {inv:.6f}×  "
                f"(would speed up HQ — double-check which file is the nightcore)"
            )

    # ── Recommended sox command ────────────────────────────────────────────────
    hqnc_path = _make_hqnc_path(hq)
    print()
    print("  Recommended sox command:")
    print(f"    sox '{hq}' '{hqnc_path}' speed {tr:.6f}")


def _print_verification_result(
    result: "pipeline.AnalysisResult",
    hqnc: Path,
    ncog: Path,
) -> bool:
    """
    Interpret the HQNC-vs-NCOG comparison.

    Returns True if tempo (and pitch) are within tolerance so the
    caller knows whether a re-run is worth offering.
    """
    tr = result.tempo_ratio
    pr = result.pitch_ratio

    print()
    _hr("═")
    print("  VERIFICATION  (HQNC vs NCOG — nightcore \u2194 nightcore)")
    _hr("═")
    print(f"  Comparing : {hqnc.name}")
    print(f"       vs   : {ncog.name}")
    print(f"  Tempo ratio : {tr:.6f}×")
    print(f"  Pitch ratio : {pr:.6f}")

    tempo_ok = abs(tr - 1.0) < _NEAR_UNITY
    pitch_ok = abs(pr - 1.0) < _NEAR_UNITY

    if tempo_ok and pitch_ok:
        print()
        print("  Files are essentially identical in tempo and pitch.")
        print("  HQNC is a faithful high-quality recreation of NCOG.")
    elif tempo_ok and not pitch_ok:
        st = -12 * __import__("math").log2(pr)
        print()
        print(f"  Tempos match, but pitch differs by ~{st:+.2f} semitones.")
        print("  NCOG appears to have an additional pitch shift on top of the speed-up.")
        print("  Add a '--pitch' flag to rubberband if you want to undo it.")
    else:
        pct_off = (tr - 1.0) * 100
        print()
        print(f"  Speed still differs by {pct_off:+.2f}%.")

    # Format / quality
    print()
    _print_format_quality(hqnc, ncog)

    return tempo_ok


def _print_format_quality(hqnc: Path, ncog: Path) -> None:
    """
    Print a brief container-format note.  Full quality analysis (including
    lossy-transcode detection via effective bandwidth) is done by the spectral
    analysis step; this is just a quick header note.
    """
    lossless  = {"flac", "wav", "aiff", "aif", "pcm"}
    ext_hqnc  = hqnc.suffix.lstrip(".").lower()
    ext_ncog  = ncog.suffix.lstrip(".").lower()
    note_hqnc = "lossless container" if ext_hqnc in lossless else "lossy"
    note_ncog = "lossless container" if ext_ncog in lossless else "lossy"
    print(f"  Format: HQNC = {ext_hqnc.upper()} ({note_hqnc})  |  NCOG = {ext_ncog.upper()} ({note_ncog})")
    print("  Run spectral analysis for a full quality assessment (including transcode detection).")


# ── spectral analysis ─────────────────────────────────────────────────────────

def run_spectral_analysis(
    path_a: Optional[Path] = None,
    path_b: Optional[Path] = None,
    label_a: str = "FILE A",
    label_b: str = "FILE B",
) -> None:
    print()
    _hr("═")
    print("  SPECTRAL ANALYSIS")
    _hr("═")

    if path_a is None:
        path_a = _prompt_file("File A (reference)")
        label_a = path_a.name
    if path_b is None:
        path_b = _prompt_file("File B (other)")
        label_b = path_b.name

    print()
    stats_a = spec.analyze(str(path_a), label=label_a)
    stats_b = spec.analyze(str(path_b), label=label_b)

    spec.compare_and_print(
        stats_a, stats_b,
        label_ref=label_a, label_other=label_b,
        ref_path=str(path_a), other_path=str(path_b),
    )


# ── mode: full suite ──────────────────────────────────────────────────────────

def run_full_suite(hq: Path, ncog: Path) -> None:
    print()
    _hr("═")
    print("  FULL SUITE")
    _hr("═")

    # Step 1 — speed comparison
    print("\n  Step 1/3 — Speed comparison  (HQ vs NCOG)")
    result1 = _run_pipeline(nightcore=ncog, source=hq, step_label="Analysing HQ vs NCOG…")
    _print_speed_result(result1, hq, ncog)

    tr = result1.tempo_ratio

    # Ask to create HQNC — prompt wording and default depend on the speed factor
    print()
    if abs(tr - 1.0) < _NEAR_UNITY:
        hqnc_name = _make_hqnc_path(hq).name
        print(
            f"  ! Speed factor is ~1.000× — no meaningful speed change would be applied.\n"
            f"    Output would be: {hqnc_name}\n"
            f"    If HQ is already a nightcore, this produces a pointless copy.\n"
            f"    Check that the correct files were provided (NCOG first, then HQ)."
        )
        ans = _prompt_choice("  Create HQNC anyway?", options="yne", default="n")
    elif tr < 1.0:
        print(
            f"  !! Speed factor is {tr:.6f}× — LESS THAN 1.\n"
            f"     This would create a SLOWER version of HQ, not a faster one.\n"
            f"     Check that files are in the correct order (NCOG first, then HQ)."
        )
        ans = _prompt_choice("  Create this slower file anyway?", options="yne", default="n")
    else:
        ans = _prompt_choice(
            "  Create HQNC (speed up HQ by the detected factor)?",
            options="yne",
            default="y",
        )

    hqnc: Optional[Path] = None
    current_speed = tr
    upd_version = 0
    if ans == "y":
        hqnc = _make_hqnc_path_v(hq, upd_version)
        _run_sox(hq, hqnc, current_speed)

    # Step 2 — verification loop (retry with corrected factor until OK or user bails)
    if hqnc and hqnc.is_file():
        attempt = 0
        while True:
            attempt += 1
            step_label = (
                "Step 2/3 — Verification  (HQNC vs NCOG)"
                if attempt == 1
                else f"Step 2/3 — Re-verification  (attempt {attempt})"
            )
            print(f"\n  {step_label}")
            result2 = _run_pipeline(
                nightcore=ncog, source=hqnc,
                step_label="Analysing HQNC vs NCOG…",
            )
            tempo_ok = _print_verification_result(result2, hqnc, ncog)

            if tempo_ok:
                break  # good enough — proceed to spectral

            # ── Offer corrected re-run ────────────────────────────────────────
            corrected_speed = current_speed * result2.tempo_ratio
            upd_version += 1
            next_hqnc = _make_hqnc_path_v(hq, upd_version)
            pct_off = (result2.tempo_ratio - 1.0) * 100

            print()
            print(f"  Speed is still off by {pct_off:+.2f}%.")
            print(
                f"  Corrected factor: {current_speed:.6f} × {result2.tempo_ratio:.6f}"
                f" = {corrected_speed:.6f}×"
            )
            print(f"  Would create: {next_hqnc.name}")
            ans_retry = _prompt_choice(
                "  Re-run sox with corrected factor?",
                options="yne",
                default="y",
            )
            if ans_retry != "y":
                break

            _run_sox(hq, next_hqnc, corrected_speed)
            hqnc = next_hqnc
            current_speed = corrected_speed
    else:
        print("\n  Step 2/3 — Skipped (no HQNC created).")

    # Step 3 — spectral analysis
    print()
    ans2 = _prompt_choice("  Run spectral analysis?", options="yn")
    if ans2 == "y":
        if hqnc and hqnc.is_file():
            run_spectral_analysis(
                path_a=hqnc, path_b=ncog,
                label_a=f"HQNC ({hqnc.name})",
                label_b=f"NCOG ({ncog.name})",
            )
        else:
            # No HQNC — compare HQ vs NCOG spectrally
            run_spectral_analysis(
                path_a=hq, path_b=ncog,
                label_a=f"HQ ({hq.name})",
                label_b=f"NCOG ({ncog.name})",
            )


# ── mode: speed comparison ────────────────────────────────────────────────────

def run_speed_comparison(hq: Path, ncog: Path) -> None:
    print()
    _hr("═")
    print("  SPEED COMPARISON")
    _hr("═")

    result = _run_pipeline(nightcore=ncog, source=hq, step_label="Analysing HQ vs NCOG…")
    _print_speed_result(result, hq, ncog)

    tr = result.tempo_ratio
    tempo_same = abs(tr - 1.0) < _NEAR_UNITY
    pr = result.pitch_ratio
    pitch_same = abs(pr - 1.0) < _NEAR_UNITY

    hqnc: Optional[Path] = None

    if tempo_same and pitch_same:
        print("\n  Files appear to be at the same speed and pitch — possibly the same file.")
    else:
        if not tempo_same:
            print()
            if abs(tr - 1.0) < _NEAR_UNITY:
                hqnc_name = _make_hqnc_path(hq).name
                print(
                    f"  ! Speed factor is ~1.000× — output would be: {hqnc_name}\n"
                    f"    Check that the correct files were provided (NCOG first, then HQ)."
                )
                ans = _prompt_choice("  Create HQNC anyway?", options="yne", default="n")
            elif tr < 1.0:
                print(
                    f"  !! Speed factor is {tr:.6f}× — LESS THAN 1.\n"
                    f"     This would create a SLOWER file. Check file order (NCOG first, then HQ)."
                )
                ans = _prompt_choice("  Create this slower file anyway?", options="yne", default="n")
            else:
                ans = _prompt_choice(
                    "  Create HQNC (speed up HQ by the detected factor)?",
                    options="yne",
                    default="y",
                )
            if ans == "y":
                hqnc = _make_hqnc_path(hq)
                _run_sox(hq, hqnc, tr)

    # Spectral
    print()
    ans2 = _prompt_choice("  Run spectral analysis?", options="yn")
    if ans2 == "y":
        if hqnc and hqnc.is_file():
            run_spectral_analysis(
                path_a=hqnc, path_b=ncog,
                label_a=f"HQNC ({hqnc.name})",
                label_b=f"NCOG ({ncog.name})",
            )
        else:
            run_spectral_analysis(
                path_a=hq, path_b=ncog,
                label_a=f"HQ ({hq.name})",
                label_b=f"NCOG ({ncog.name})",
            )


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    args = sys.argv[1:]
    ncog_arg = args[0] if len(args) > 0 else None
    hq_arg   = args[1] if len(args) > 1 else None

    print()
    _hr("═")
    print("  NIGHTCORE ANALYZER — WORKFLOW")
    _hr("═")
    print("  [f]  Full suite  (speed comparison → create HQNC → verification → spectral)")
    print("  [s]  Speed comparison  (+ optional HQNC creation + optional spectral)")
    print("  [a]  Spectral analysis  (standalone two-file comparison)")
    print("  [e]  Exit")
    print()

    mode = _prompt_choice("Choose mode", options="fsae")

    if mode == "a":
        run_spectral_analysis()
        return

    # Modes f and s need both files — NCOG first, then HQ
    print()
    ncog = _prompt_file("NCOG (nightcore edit)", ncog_arg)
    hq   = _prompt_file("HQ source (original high-quality)", hq_arg)

    if mode == "f":
        run_full_suite(hq, ncog)
    else:
        run_speed_comparison(hq, ncog)


if __name__ == "__main__":
    main()
