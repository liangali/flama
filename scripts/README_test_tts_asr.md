# test_tts_asr.py — TTS → ASR Round-Trip Quality Test

端到端测试脚本：用 `modeling_qwen3_tts.exe` 生成音频，再用 `modeling_qwen3_asr.exe` 转录，最后用字符错误率（CER）评估质量。

## 依赖

- Python 3.8+，无需额外安装第三方包
- 编译好的可执行文件：
  - `modeling_qwen3_tts.exe`
  - `modeling_qwen3_asr.exe`
- 模型文件：
  - `<models-root>/Huggingface/Qwen3-TTS-12Hz-1.7B-Base`
  - `<models-root>/Huggingface/Qwen3-ASR-0.6B`

## 目录结构

```
scripts/
├── test_tts_asr.py
└── Release/                  ← 默认 exe 目录，需要包含dll
    ├── modeling_qwen3_tts.exe
    └── modeling_qwen3_asr.exe
```

## 用法

```bash
# 使用默认路径（exe 在 scripts/Release/，模型在 D:/data/models）
python test_tts_asr.py

# 指定模型根目录
python test_tts_asr.py --models-root D:/data/models

# 指定 exe 目录
python test_tts_asr.py --models-root D:/data/models --exe-dir ./Release

# 使用 CPU 推理
python test_tts_asr.py --device CPU

# 完整参数示例
python test_tts_asr.py \
  --models-root D:/data/models \
  --exe-dir ./Release \
  --device GPU \
  --output-dir ./my_output
```

## 参数说明

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--models-root` | `D:\data\models` | 模型文件根目录 |
| `--exe-dir` | `scripts/Release` | exe 文件所在目录 |
| `--tts-model` | 自动推导 | 覆盖 TTS 模型路径（绝对路径） |
| `--asr-model` | 自动推导 | 覆盖 ASR 模型路径（绝对路径） |
| `--device` | `GPU` | 推理设备，可选 `GPU` / `CPU` |
| `--output-dir` | `<exe-dir>/tts_asr_test_<时间戳>` | 音频及报告输出目录 |

## 输出文件

每次运行在 `--output-dir` 下生成：

| 文件 | 说明 |
|---|---|
| `tts_<label>_<timestamp>.wav` | TTS 生成的音频文件 |
| `report_<timestamp>.txt` | 详细文本报告（含每条用例的 REF/HYP/CER） |
| `summary_<timestamp>.csv` | 汇总 CSV，便于后续分析 |

CSV 字段：`label, tts_ok, asr_ok, cer, score, tts_dur_s, asr_dur_s, wav, ref, asr`

## 指标说明

- **CER**（Character Error Rate）：字符错误率，越低越好，0% 为完美
- **Score**：准确率 = `max(0, 100 - CER)`，越高越好
- 控制台进度条 `[██████████]` 直观显示单条 Score

## 测试用例

脚本内置 6 条中文测试用例，覆盖不同音频时长：

| 标签 | 预期时长 |
|---|---|
| `short_30s` | ~30 秒 |
| `medium_60s` | ~60 秒 |
| `long_120s` | ~120 秒 |
| 等… | … |
