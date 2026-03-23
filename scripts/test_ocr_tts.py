"""OCR -> TTS pipeline test.

Recognizes text from images via modeling_glm_ocr.exe, then synthesizes
the recognized text to audio via modeling_qwen3_tts.exe.

Usage:
  python test_ocr_tts.py
  python test_ocr_tts.py --models-root D:/data/models
  python test_ocr_tts.py --models-root D:/data/models --exe-dir ./Release
  python test_ocr_tts.py --device CPU
  python test_ocr_tts.py --images path/to/a.png path/to/b.png
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_MODELS_ROOT = r"D:\data\models"
OCR_MODEL_REL = Path("Huggingface") / "GLM-OCR"
TTS_MODEL_REL = Path("Huggingface") / "Qwen3-TTS-12Hz-1.7B-Base"

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_EXE_DIR = SCRIPT_DIR / "Release"

DEFAULT_DEVICE = "GPU"
OCR_MAX_NEW_TOKENS = "1000"
OCR_PROMPT = "Text Recognition"

# Default test images bundled alongside this script
DEFAULT_IMAGES: List[Path] = [
    SCRIPT_DIR / "OCR-CN-Short.png",
    SCRIPT_DIR / "OCR-CN-medium.png",
    SCRIPT_DIR / "OCR-CN-long.png",
    SCRIPT_DIR / "OCR-EN-Short.png",
    SCRIPT_DIR / "OCR-EN-medium.png",
    SCRIPT_DIR / "OCR-EN-long.png",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def format_duration(delta: _dt.timedelta) -> str:
    s = delta.total_seconds()
    if s < 60:
        return f"{s:.2f}s"
    m = int(s // 60)
    return f"{m}m{s % 60:05.2f}s"


def extract_ocr_text(output: str) -> str:
    """Extract the recognized text from OCR exe stdout.

    The OCR exe prints performance stats first, then dumps the recognized
    text (possibly multi-line) at the end with no prefix marker.
    We collect every line after the last performance/stats line.
    """
    import re
    # Try "  text: <content>" line (same format as ASR)
    for line in output.splitlines():
        m = re.match(r"\s*text:\s*(.+)", line)
        if m:
            return m.group(1).strip()

    # Perf/stats lines: start with "[", "->", or match "Key: value" patterns
    # like "Throughput: 130.88 tokens/s", "TTFT: ...", "Image shape: ..."
    stats_pattern = re.compile(
        r"^\s*(\[|->|"
        r"Throughput:|TTFT:|TPOT:|Decode|Preprocess|Vision|Prompt|Output|"
        r"Repetition|pixel_values|grid_thw|Image shape|"
        r"Total weights|Quantized|Quantization|Timing|Fetch|Quant|Graph|"
        r"perf\.)"
    )

    lines = output.splitlines()
    # Find the last stats line, collect everything after it
    last_stats_idx = -1
    for i, line in enumerate(lines):
        if stats_pattern.match(line):
            last_stats_idx = i

    if last_stats_idx != -1:
        text_lines = [l.rstrip('"') for l in lines[last_stats_idx + 1:] if l.strip()]
        if text_lines:
            return "".join(text_lines)

    return ""


# ---------------------------------------------------------------------------
# Runner functions
# ---------------------------------------------------------------------------

def run_ocr(
    exe: Path,
    model: Path,
    image: Path,
    prompt: str,
    device: str,
    max_new_tokens: str,
    env: dict,
    work_dir: Path,
) -> Tuple[int, str]:
    cmd = [str(exe), str(model), str(image), prompt, device, max_new_tokens]
    print(f"  CMD: {' '.join(cmd)}")
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
    print(f"  CMD: {' '.join(cmd)}")
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


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OCR -> TTS pipeline test",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python test_ocr_tts.py\n"
            "  python test_ocr_tts.py --models-root D:/data/models\n"
            "  python test_ocr_tts.py --device CPU\n"
            "  python test_ocr_tts.py --images a.png b.png\n"
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
        "--ocr-model",
        default=None,
        help="Override OCR model path (absolute).",
    )
    parser.add_argument(
        "--tts-model",
        default=None,
        help="Override TTS model path (absolute).",
    )
    parser.add_argument(
        "--device",
        default=DEFAULT_DEVICE,
        help=f"Inference device (default: {DEFAULT_DEVICE})",
    )
    parser.add_argument(
        "--images",
        nargs="+",
        default=None,
        help="Image files to test (default: built-in OCR-CN-*.png test images)",
    )
    parser.add_argument(
        "--ocr-prompt",
        default=OCR_PROMPT,
        help=f'OCR prompt string (default: "{OCR_PROMPT}")',
    )
    parser.add_argument(
        "--ocr-max-tokens",
        default=OCR_MAX_NEW_TOKENS,
        help=f"Max new tokens for OCR (default: {OCR_MAX_NEW_TOKENS})",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for wav files and reports (default: <script-dir>/ocr_tts_test_<timestamp>)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()

    exe_dir = Path(args.exe_dir).resolve()
    models_root = Path(args.models_root)

    ocr_model = Path(args.ocr_model) if args.ocr_model else models_root / OCR_MODEL_REL
    tts_model = Path(args.tts_model) if args.tts_model else models_root / TTS_MODEL_REL

    ocr_exe = exe_dir / "modeling_glm_ocr.exe"
    tts_exe = exe_dir / "modeling_qwen3_tts.exe"

    device = args.device

    # Resolve image list
    if args.images:
        images = [Path(p).resolve() for p in args.images]
    else:
        images = [p for p in DEFAULT_IMAGES if p.exists()]
        if not images:
            print("ERROR: No default test images found. Place OCR-CN-*.png next to this script or use --images.")
            return 1

    # Validate required paths
    missing = []
    for label, path in [("OCR exe", ocr_exe), ("TTS exe", tts_exe),
                         ("OCR model", ocr_model), ("TTS model", tts_model)]:
        if not path.exists():
            missing.append(f"  {label} not found: {path}")
    for img in images:
        if not img.exists():
            missing.append(f"  Image not found: {img}")
    if missing:
        print("ERROR: Missing required paths:")
        print("\n".join(missing))
        return 1

    timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir).resolve() if args.output_dir else SCRIPT_DIR / "output" / f"ocr_tts_test_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    env = build_env(exe_dir)
    work_dir = exe_dir

    print("=" * 70)
    print("OCR -> TTS Pipeline Test")
    print("=" * 70)
    print(f"OCR model  : {ocr_model}")
    print(f"TTS model  : {tts_model}")
    print(f"Device     : {device}")
    print(f"Output dir : {output_dir}")
    print()

    total_start = _dt.datetime.now()
    results = []

    for idx, image_path in enumerate(images, 1):
        label = image_path.stem
        print(f"[{idx}/{len(images)}] Image: {image_path.name}")

        # --- OCR ---
        ocr_start = _dt.datetime.now()
        ocr_rc, ocr_out = run_ocr(
            ocr_exe, ocr_model, image_path,
            args.ocr_prompt, device, args.ocr_max_tokens,
            env, work_dir,
        )
        ocr_dur = _dt.datetime.now() - ocr_start

        ocr_text = extract_ocr_text(ocr_out)
        ocr_ok = ocr_rc == 0

        print(f"  OCR done  : {format_duration(ocr_dur)}")
        print(f"  OCR raw   :\n{ocr_out}")
        print(f"  OCR text  : {ocr_text}")

        if not ocr_ok or not ocr_text:
            print(f"  OCR FAILED (exit {ocr_rc})" if not ocr_ok else "  OCR returned empty text, skipping TTS.")
            results.append({
                "label": label, "image": str(image_path),
                "ocr_ok": ocr_ok, "tts_ok": False,
                "ocr_text": ocr_text, "wav": "",
                "ocr_dur": ocr_dur, "tts_dur": _dt.timedelta(0),
                "ocr_raw": ocr_out, "tts_raw": "",
            })
            print()
            continue

        # --- TTS ---
        wav_name = f"tts_{label}_{timestamp}.wav"
        wav_path = output_dir / wav_name

        tts_start = _dt.datetime.now()
        tts_rc, tts_out = run_tts(tts_exe, tts_model, ocr_text, wav_path, device, env, work_dir)
        tts_dur = _dt.datetime.now() - tts_start

        tts_ok = tts_rc == 0 and wav_path.exists()

        if not tts_ok:
            print(f"  TTS FAILED (exit {tts_rc})")
            print("  --- TTS output ---")
            print(tts_out[:2000])
        else:
            print(f"  TTS done  : {format_duration(tts_dur)} -> {wav_name}")

        results.append({
            "label": label, "image": str(image_path),
            "ocr_ok": ocr_ok, "tts_ok": tts_ok,
            "ocr_text": ocr_text, "wav": str(wav_path) if tts_ok else "",
            "ocr_dur": ocr_dur, "tts_dur": tts_dur,
            "ocr_raw": ocr_out, "tts_raw": tts_out,
        })
        print()

    total_dur = _dt.datetime.now() - total_start

    # ---------------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------------
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  {'Image':<20} {'OCR':>4} {'TTS':>4}  {'OCR time':>8}  {'TTS time':>8}  Text preview")
    print("-" * 70)
    for r in results:
        ocr_flag = "OK" if r["ocr_ok"] else "FAIL"
        tts_flag = "OK" if r["tts_ok"] else ("SKIP" if r["ocr_ok"] else "FAIL")
        preview = r["ocr_text"][:40] + ("…" if len(r["ocr_text"]) > 40 else "")
        ocr_t = format_duration(r["ocr_dur"])
        tts_t = format_duration(r["tts_dur"]) if r["tts_ok"] else "-"
        print(f"  {r['label']:<20} {ocr_flag:>4} {tts_flag:>4}  {ocr_t:>8}  {tts_t:>8}  {preview}")
    print("-" * 70)
    ok_count = sum(1 for r in results if r["tts_ok"])
    print(f"  Passed: {ok_count}/{len(results)}")
    print(f"  Total time: {format_duration(total_dur)}")
    print("=" * 70)

    # ---------------------------------------------------------------------------
    # Save report
    # ---------------------------------------------------------------------------
    report_path = output_dir / f"report_{timestamp}.txt"
    with report_path.open("w", encoding="utf-8") as f:
        f.write("OCR -> TTS Pipeline Test\n")
        f.write(f"Generated : {_dt.datetime.now().isoformat()}\n")
        f.write(f"Device    : {device}\n")
        f.write(f"OCR model : {ocr_model}\n")
        f.write(f"TTS model : {tts_model}\n\n")
        for r in results:
            f.write(f"[{r['label']}]\n")
            f.write(f"  Image    : {r['image']}\n")
            f.write(f"  OCR time : {format_duration(r['ocr_dur'])}\n")
            f.write(f"  TTS time : {format_duration(r['tts_dur'])}\n")
            f.write(f"  OCR text : {r['ocr_text']}\n")
            f.write(f"  WAV      : {r['wav']}\n")
            f.write(f"  OCR raw  :\n{r.get('ocr_raw', '')}\n")
            f.write(f"  TTS raw  :\n{r.get('tts_raw', '')}\n\n")
        f.write(f"Passed     : {ok_count}/{len(results)}\n")
        f.write(f"Total time : {format_duration(total_dur)}\n")
    print(f"Report saved: {report_path}")

    # ---------------------------------------------------------------------------
    # Save CSV
    # ---------------------------------------------------------------------------
    csv_path = output_dir / f"summary_{timestamp}.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "label", "image", "ocr_ok", "tts_ok",
            "ocr_dur_s", "tts_dur_s", "wav", "ocr_text"
        ])
        writer.writeheader()
        for r in results:
            writer.writerow({
                "label": r["label"],
                "image": r["image"],
                "ocr_ok": r["ocr_ok"],
                "tts_ok": r["tts_ok"],
                "ocr_dur_s": f"{r['ocr_dur'].total_seconds():.2f}",
                "tts_dur_s": f"{r['tts_dur'].total_seconds():.2f}",
                "wav": r["wav"],
                "ocr_text": r["ocr_text"],
            })
    print(f"CSV saved   : {csv_path}")

    return 0 if ok_count == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
