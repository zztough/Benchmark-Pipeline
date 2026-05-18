# MeViS Event-Driven QA Pipeline

This directory contains a modular Python pipeline for:

1. Loading MeViS metadata and mask annotations.
2. Calling SiliconFlow-hosted API models for name extraction and temporal grounding.
3. Generating event-driven QA pairs and computing geometric ground truth from masks.

## Quick Start

```bash
conda create -n bench-pipeline python=3.10 -y
conda activate bench-pipeline
pip install -r requirements.txt

export SILICONFLOW_API_KEY="YOUR_API_KEY"

python main.py \
  --data-root /home/zhaobing/MeViS/dataset/MeViSv2/valid_u \
  --output-dir /home/zhaobing/bench-pipeline/outputs/mevis_demo \
  --max-videos 5 \
  --seed 42 \
  --use-depth-anything \
  --depth-model da3-large \
  --depth-device cuda
```

Outputs are written to:

- `structured_logs.jsonl`
- `qa_samples.jsonl`

## Notes

- `DeepSeekExtractionInterface` uses `deepseek-ai/DeepSeek-V4-Flash` by default.
- `QwenVTGInterface` uses `Qwen/Qwen3.6-35B-A3B` by default.
- Both clients talk to SiliconFlow through the OpenAI-compatible `/v1` endpoint.
- The pipeline only keeps expressions with exactly one `anno_id` and skips duplicate `anno_id` entries within the same video.
- When a video has only one valid entity, the QA generator only emits the single-entity template.
