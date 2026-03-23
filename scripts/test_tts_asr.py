"""TTS -> ASR round-trip quality test.

Generates several audio files via modeling_qwen3_tts.exe, then transcribes
each one with modeling_qwen3_asr.exe, and scores accuracy using Character
Error Rate (CER).

Usage:
  python test_tts_asr.py
  python test_tts_asr.py --models-root D:/data/models
  python test_tts_asr.py --models-root D:/data/models --exe-dir ../Release
  python test_tts_asr.py --device CPU
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_MODELS_ROOT = r"D:\data\models"
TTS_MODEL_REL = Path("Huggingface") / "Qwen3-TTS-12Hz-1.7B-Base"
ASR_MODEL_REL = Path("Huggingface") / "Qwen3-ASR-0.6B"

SCRIPT_DIR = Path(__file__).resolve().parent
print(SCRIPT_DIR)
DEFAULT_EXE_DIR = SCRIPT_DIR / "Release"
DEFAULT_PROMPTS_FILE = SCRIPT_DIR / "prompt.json"

DEFAULT_DEVICE = "GPU"
ASR_MAX_NEW_TOKENS = "500"


def load_test_cases(prompts_file: Path) -> List[Tuple[str, str]]:
    with prompts_file.open(encoding="utf-8") as f:
        data = json.load(f)
    return [(entry["label"], entry["text"]) for entry in data]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def edit_distance(a: str, b: str) -> int:
    """Levenshtein distance between two strings."""
    m, n = len(a), len(b)
    # Use two-row DP to save memory
    prev = list(range(n + 1))
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        curr[0] = i
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1]
            else:
                curr[j] = 1 + min(prev[j], curr[j - 1], prev[j - 1])
        prev, curr = curr, prev
    return prev[n]


def cer(reference: str, hypothesis: str) -> float:
    """Character Error Rate in percent. Returns 100.0 if reference is empty."""
    ref = reference.replace(" ", "")
    hyp = hypothesis.replace(" ", "")
    if not ref:
        return 100.0
    return edit_distance(ref, hyp) / len(ref) * 100.0


def accuracy_score(cer_value: float) -> float:
    """Convert CER to 0-100 accuracy score (clamped)."""
    return max(0.0, 100.0 - cer_value)


def extract_asr_text(output: str) -> str:
    """Pull the transcription from ASR stdout."""
    import re
    # Match "  text: <content>" line
    for line in output.splitlines():
        m = re.match(r"\s*text:\s*(.+)", line)
        if m:
            text = m.group(1).strip()
            # Strip leading artifacts like "language ChineseBitFields"
            text = re.sub(r"^(language\s+\w+\s*)+(BitFields)?", "", text).strip()
            # Strip trailing artifacts like "language Chinese</think>"
            text = re.sub(r"\s*language\s+\w+(<\/think>)?$", "", text).strip()
            return text
    # Fallback: "Generated text:" marker
    marker = "Generated text:"
    idx = output.find(marker)
    if idx != -1:
        text = output[idx + len(marker):].lstrip("\r\n").strip()
        return text if text else ""
    return ""


def run_tts(
    exe: Path,
    model: Path,
    text: str,
    output_wav: Path,
    device: str,
    env: dict,
    work_dir: Path,
) -> Tuple[int, str]:
    cmd = [str(exe), str(model), text, str(output_wav), device]
    #print(f"  CMD: {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        cwd=str(work_dir),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return result.returncode, result.stdout or ""


def run_asr(
    exe: Path,
    model: Path,
    wav_path: Path,
    device: str,
    env: dict,
    work_dir: Path,
) -> Tuple[int, str]:
    cmd = [
        str(exe),
        str(model),
        "--cache-model",
        "--wav", str(wav_path),
        "--device", device,
        "--max_new_tokens", ASR_MAX_NEW_TOKENS,
    ]
    #print(f"  CMD: {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        cwd=str(work_dir),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return result.returncode, result.stdout or ""


def build_env(exe_dir: Path) -> dict:
    env = os.environ.copy()
    original_path = env.get("PATH", "")
    exe_dir_str = str(exe_dir)
    env["PATH"] = f"{exe_dir_str};{original_path}" if original_path else exe_dir_str
    env.setdefault("OV_GENAI_USE_MODELING_API", "1")
    return env


def format_duration(delta: _dt.timedelta) -> str:
    s = delta.total_seconds()
    if s < 60:
        return f"{s:.2f}s"
    m = int(s // 60)
    return f"{m}m{s % 60:05.2f}s"


def bar(score: float, width: int = 30) -> str:
    filled = int(round(score / 100.0 * width))
    return "[" + "#" * filled + "-" * (width - filled) + f"] {score:.1f}%"


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TTS -> ASR round-trip quality test",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python test_tts_asr.py\n"
            "  python test_tts_asr.py --models-root D:/data/models\n"
            "  python test_tts_asr.py --device CPU\n"
        ),
    )
    parser.add_argument(
        "--models-root",
        default=DEFAULT_MODELS_ROOT,
        help=f"Root folder for model files (default: {DEFAULT_MODELS_ROOT})",
    )
    parser.add_argument(
        "--exe-dir",
        default=str(DEFAULT_EXE_DIR),
        help=f"Directory containing the .exe files (default: {DEFAULT_EXE_DIR})",
    )
    parser.add_argument(
        "--tts-model",
        default=None,
        help="Override TTS model path (absolute).",
    )
    parser.add_argument(
        "--asr-model",
        default=None,
        help="Override ASR model path (absolute).",
    )
    parser.add_argument(
        "--device",
        default=DEFAULT_DEVICE,
        help=f"Inference device (default: {DEFAULT_DEVICE})",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for generated .wav files (default: <script-dir>/tts_asr_test_<timestamp>)",
    )
    parser.add_argument(
        "--prompts",
        default=str(DEFAULT_PROMPTS_FILE),
        help=f"JSON file with test cases (default: {DEFAULT_PROMPTS_FILE})",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    exe_dir = Path(args.exe_dir).resolve()
    models_root = Path(args.models_root)
    device = args.device

    tts_exe = exe_dir / "modeling_qwen3_tts.exe"
    asr_exe = exe_dir / "modeling_qwen3_asr.exe"
    tts_model = Path(args.tts_model) if args.tts_model else models_root / TTS_MODEL_REL
    asr_model = Path(args.asr_model) if args.asr_model else models_root / ASR_MODEL_REL

    # Validate
    errors: List[str] = []
    for label, path in [
        ("TTS exe", tts_exe),
        ("ASR exe", asr_exe),
        ("TTS model", tts_model),
        ("ASR model", asr_model),
    ]:
        if not path.exists():
            errors.append(f"  {label} not found: {path}")
    if errors:
        print("ERROR: Missing required paths:", file=sys.stderr)
        for e in errors:
            print(e, file=sys.stderr)
        return 2

    timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = SCRIPT_DIR / "output" / f"tts_asr_test_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    env = build_env(exe_dir)
    work_dir = exe_dir

    print("=" * 70)
    print("TTS -> ASR Round-Trip Quality Test")
    print("=" * 70)
    print(f"TTS model : {tts_model}")
    print(f"ASR model : {asr_model}")
    print(f"Device    : {device}")
    print(f"Output dir: {output_dir}")
    print()

    test_cases = load_test_cases(Path(args.prompts))
    print(f"Prompts   : {args.prompts} ({len(test_cases)} cases)")
    print()

    results: List[dict] = []
    total_start = _dt.datetime.now()

    for idx, (label, ref_text) in enumerate(test_cases):
        wav_name = f"tts_{label}_{timestamp}.wav"
        wav_path = output_dir / wav_name

        print(f"[{idx + 1}/{len(test_cases)}] Case: {label}")
        print(f"  TTS input : {ref_text}")

        # --- TTS ---
        tts_start = _dt.datetime.now()
        tts_rc, tts_out = run_tts(tts_exe, tts_model, ref_text, wav_path, device, env, work_dir)
        tts_dur = _dt.datetime.now() - tts_start

        if tts_rc != 0:
            print(f"  TTS FAILED (exit {tts_rc})")
            print("  --- TTS output ---")
            print(tts_out[:2000])
            results.append(
                {
                    "label": label,
                    "ref": ref_text,
                    "asr": "",
                    "cer": 100.0,
                    "score": 0.0,
                    "tts_ok": False,
                    "asr_ok": False,
                    "wav": str(wav_path),
                    "tts_dur": tts_dur,
                    "asr_dur": _dt.timedelta(0),
                    "tts_raw": tts_out,
                    "asr_raw": "",
                }
            )
            print()
            continue

        print(f"  TTS done  : {format_duration(tts_dur)} -> {wav_path.name}")

        if not wav_path.is_file():
            print("  ERROR: WAV file not created.")
            results.append(
                {
                    "label": label,
                    "ref": ref_text,
                    "asr": "",
                    "cer": 100.0,
                    "score": 0.0,
                    "tts_ok": False,
                    "asr_ok": False,
                    "wav": str(wav_path),
                    "tts_dur": tts_dur,
                    "asr_dur": _dt.timedelta(0),
                    "tts_raw": tts_out,
                    "asr_raw": "",
                }
            )
            print()
            continue

        # --- ASR ---
        asr_start = _dt.datetime.now()
        asr_rc, asr_out = run_asr(asr_exe, asr_model, wav_path, device, env, work_dir)
        asr_dur = _dt.datetime.now() - asr_start

        asr_text = extract_asr_text(asr_out)
        asr_ok = asr_rc == 0

        if not asr_ok:
            print(f"  ASR FAILED (exit {asr_rc})")
            print("  --- ASR output ---")
            print(asr_out[:2000])

        print(f"  ASR done  : {format_duration(asr_dur)}")
        print(f"  ASR output: {asr_text}")
        print(f"  ASR raw   :\n{asr_out}")

        cer_val = cer(ref_text, asr_text)
        score = accuracy_score(cer_val)

        print(f"  CER       : {cer_val:.1f}%")
        print(f"  Score     : {bar(score)}")

        results.append(
            {
                "label": label,
                "ref": ref_text,
                "asr": asr_text,
                "cer": cer_val,
                "score": score,
                "tts_ok": True,
                "asr_ok": asr_ok,
                "wav": str(wav_path),
                "tts_dur": tts_dur,
                "asr_dur": asr_dur,
                "tts_raw": tts_out,
                "asr_raw": asr_out,
            }
        )
        print()

    total_dur = _dt.datetime.now() - total_start

    # ---------------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------------
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    header = f"{'Case':<12} {'TTS':>4} {'ASR':>4} {'CER':>8} {'Score':>8}  Reference / Transcription"
    print(header)
    print("-" * 70)

    valid_scores = [r["score"] for r in results if r["tts_ok"] and r["asr_ok"]]
    for r in results:
        tts_flag = "OK" if r["tts_ok"] else "FAIL"
        asr_flag = "OK" if r["asr_ok"] else "FAIL"
        cer_str = f"{r['cer']:.1f}%" if r["tts_ok"] else "N/A"
        score_str = f"{r['score']:.1f}" if r["tts_ok"] else "N/A"
        print(f"  {r['label']:<10} {tts_flag:>4} {asr_flag:>4} {cer_str:>8} {score_str:>8}")
        print(f"    REF: {r['ref']}")
        print(f"    HYP: {r['asr']}")

    print("-" * 70)
    if valid_scores:
        avg_score = sum(valid_scores) / len(valid_scores)
        avg_cer = sum(r["cer"] for r in results if r["tts_ok"] and r["asr_ok"]) / len(valid_scores)
        print(f"  Average CER  : {avg_cer:.1f}%")
        print(f"  Overall Score: {bar(avg_score)}")
    else:
        avg_score = 0.0
        print("  No valid results to score.")

    print(f"  Total time   : {format_duration(total_dur)}")
    print("=" * 70)

    # ---------------------------------------------------------------------------
    # Save report
    # ---------------------------------------------------------------------------
    report_path = output_dir / f"report_{timestamp}.txt"
    with report_path.open("w", encoding="utf-8") as f:
        f.write(f"TTS -> ASR Round-Trip Quality Test\n")
        f.write(f"Generated: {_dt.datetime.now().isoformat()}\n")
        f.write(f"Device: {device}\n")
        f.write(f"TTS model: {tts_model}\n")
        f.write(f"ASR model: {asr_model}\n\n")
        for r in results:
            f.write(f"[{r['label']}]\n")
            f.write(f"  WAV      : {r['wav']}\n")
            f.write(f"  TTS time : {format_duration(r['tts_dur'])}\n")
            f.write(f"  ASR time : {format_duration(r['asr_dur'])}\n")
            f.write(f"  Reference: {r['ref']}\n")
            f.write(f"  ASR text : {r['asr']}\n")
            f.write(f"  CER      : {r['cer']:.2f}%\n")
            f.write(f"  Score    : {r['score']:.2f}/100\n")
            f.write(f"  TTS raw  :\n{r.get('tts_raw', '')}\n")
            f.write(f"  ASR raw  :\n{r.get('asr_raw', '')}\n\n")
        if valid_scores:
            f.write(f"Average CER  : {avg_cer:.2f}%\n")
            f.write(f"Overall Score: {avg_score:.2f}/100\n")
        f.write(f"Total time   : {format_duration(total_dur)}\n")

    print(f"Report saved: {report_path}")

    # ---------------------------------------------------------------------------
    # Save CSV
    # ---------------------------------------------------------------------------
    import csv
    csv_path = output_dir / f"summary_{timestamp}.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "label", "tts_ok", "asr_ok", "cer", "score",
            "tts_dur_s", "asr_dur_s", "wav", "ref", "asr"
        ])
        writer.writeheader()
        for r in results:
            writer.writerow({
                "label": r["label"],
                "tts_ok": r["tts_ok"],
                "asr_ok": r["asr_ok"],
                "cer": f"{r['cer']:.2f}",
                "score": f"{r['score']:.2f}",
                "tts_dur_s": f"{r['tts_dur'].total_seconds():.2f}",
                "asr_dur_s": f"{r['asr_dur'].total_seconds():.2f}",
                "wav": r["wav"],
                "ref": r["ref"],
                "asr": r["asr"],
            })
    print(f"CSV saved   : {csv_path}")

    return 0 if avg_score > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
