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

DEFAULT_DEVICE = "GPU"
ASR_MAX_NEW_TOKENS = "500"

# ---------------------------------------------------------------------------
# Test cases: (label, text_to_synthesize)
# Texts are sized to produce approximately 30s / 60s / 120s audio at typical
# Chinese TTS speed (~4 chars/sec).  Actual duration depends on the model.
# ---------------------------------------------------------------------------
TEST_CASES: List[Tuple[str, str]] = [
    (
        # ~30 seconds (~120 Chinese characters)
        "short_30s",
        (
            "人工智能是计算机科学的一个重要分支，致力于研究和开发能够模拟人类智能的系统。"
            "近年来，随着深度学习技术的快速发展，人工智能在图像识别、语音处理和自然语言"
            "理解等领域取得了突破性进展。未来，它将在医疗、教育和交通等领域发挥更大作用。"
        ),
    ),
    (
        # ~60 seconds (~240 Chinese characters)
        "medium_60s",
        (
            "OpenVINO 是英特尔推出的开源深度学习推理框架，专为在各种英特尔硬件上部署人工智能"
            "推理任务而设计，支持 CPU、集成显卡、独立显卡以及 VPU 等多种设备。"
            "它通过模型优化和硬件加速，显著降低了推理延迟并提升了吞吐量，使开发者能够在边缘"
            "设备和云端服务器上高效运行图像分类、目标检测、语音识别和自然语言处理等各类 AI 任务。"
            "借助 OpenVINO，工程师可以将训练好的模型从主流深度学习框架无缝转换并进行优化部署。"
        ),
    ),
    (
        # ~120 seconds (~480 Chinese characters)
        "long_120s",
        (
            "大语言模型是近年来人工智能领域最引人瞩目的技术突破之一。这类模型通过在海量文本数据上"
            "进行预训练，学习到了语言的语法结构、语义关系以及大量的世界知识，从而具备了理解和生成"
            "自然语言的强大能力。以 GPT 系列和通义千问为代表的大语言模型，不仅能够完成问答、摘要、"
            "翻译等传统自然语言处理任务，还能进行代码编写、数学推理和创意写作等复杂任务。"
            "然而，大语言模型的训练和推理都需要消耗大量的计算资源，这给实际部署带来了巨大挑战。"
            "为了解决这一问题，研究人员提出了模型量化、知识蒸馏和稀疏化等多种模型压缩技术，"
            "旨在在保持模型性能的同时大幅降低其计算和存储开销。英特尔的 OpenVINO 工具包正是在这一"
            "背景下发挥了重要作用，它能够将大语言模型高效地部署到各种英特尔硬件平台上，让更多"
            "用户和场景都能够受益于先进的人工智能技术，推动 AI 在各行各业的广泛普及与应用。"
        ),
    ),
    (
        # ~30 seconds (~70 English words)
        "en_short_30s",
        (
            "Artificial intelligence is one of the most transformative technologies of our time. "
            "By enabling machines to learn from data and make decisions, AI is reshaping industries "
            "ranging from healthcare and finance to transportation and entertainment. "
            "As the technology continues to advance, its impact on society will only grow deeper."
        ),
    ),
    (
        # ~60 seconds (~140 English words)
        "en_medium_60s",
        (
            "OpenVINO is an open-source toolkit developed by Intel for optimizing and deploying "
            "deep learning inference across a wide range of Intel hardware, including CPUs, "
            "integrated and discrete GPUs, and VPUs. "
            "It enables developers to convert models trained in popular frameworks such as "
            "PyTorch and TensorFlow into an optimized intermediate representation, "
            "then run them efficiently on the target device. "
            "By applying techniques like quantization and layer fusion, OpenVINO can dramatically "
            "reduce latency and increase throughput, making it well suited for real-time applications "
            "such as object detection, speech recognition, and natural language processing "
            "in both edge and cloud environments."
        ),
    ),
    (
        # ~120 seconds (~280 English words)
        "en_long_120s",
        (
            "Large language models represent one of the most significant breakthroughs in the history "
            "of artificial intelligence. These models are trained on vast corpora of text data, "
            "allowing them to learn grammar, semantics, and an enormous breadth of world knowledge. "
            "As a result, they can understand and generate human language with remarkable fluency, "
            "handling tasks as diverse as question answering, summarization, translation, "
            "code generation, mathematical reasoning, and creative writing. "
            "Models such as GPT-4 and Qwen have demonstrated that scale, when combined with "
            "careful training techniques, leads to emergent capabilities that were not explicitly "
            "programmed but arise naturally from the learning process. "
            "However, training and running these models requires immense computational resources, "
            "posing significant challenges for practical deployment. "
            "Researchers have responded with a range of model compression strategies, "
            "including quantization, knowledge distillation, and structured pruning, "
            "all aimed at preserving model quality while reducing memory footprint and inference cost. "
            "Intel's OpenVINO toolkit plays an important role in this ecosystem by providing "
            "hardware-aware optimizations that allow large language models to run efficiently "
            "on Intel CPUs and GPUs, bringing the benefits of advanced AI to a much broader "
            "set of users and deployment scenarios across industry and everyday life."
        ),
    ),
]


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
        help="Directory for generated .wav files (default: <exe-dir>/tts_asr_test_<timestamp>)",
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
        output_dir = exe_dir / f"tts_asr_test_{timestamp}"
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

    results: List[dict] = []
    total_start = _dt.datetime.now()

    for idx, (label, ref_text) in enumerate(TEST_CASES):
        wav_name = f"tts_{label}_{timestamp}.wav"
        wav_path = output_dir / wav_name

        print(f"[{idx + 1}/{len(TEST_CASES)}] Case: {label}")
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
            f.write(f"  Score    : {r['score']:.2f}/100\n\n")
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
