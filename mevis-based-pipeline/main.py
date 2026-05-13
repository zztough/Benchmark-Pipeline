from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from api_clients import DeepSeekExtractionInterface, QwenVTGInterface
from data_loader import MeViSDataLoader
from extractor import StructuredLogBuilder
from qa_generator import QAGenerator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an event-driven spatio-temporal QA pipeline for MeViS.")
    parser.add_argument(
        "--data-root",
        type=str,
        default="/home/zhaobing/MeViS/dataset/MeViSv2/valid_u",
        help="Path to the MeViS split directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/home/zhaobing/bench-pipeline/outputs/mevis_demo",
        help="Directory where logs and QA samples will be written.",
    )
    parser.add_argument("--max-videos", type=int, default=10, help="Limit the number of videos for a quick demo.")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for QA sampling.",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.getenv("SILICONFLOW_API_KEY"),
        help="SiliconFlow API key used by the DeepSeek and Qwen clients.",
    )
    parser.add_argument(
        "--api-base-url",
        type=str,
        default=os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1"),
        help="OpenAI-compatible base URL for SiliconFlow.",
    )
    parser.add_argument(
        "--extract-model",
        type=str,
        default="deepseek-ai/DeepSeek-V4-Flash",
        help="Model name for extraction.",
    )
    parser.add_argument(
        "--vtg-model",
        type=str,
        default="Qwen/Qwen3.6-35B-A3B",
        help="Model name for temporal grounding.",
    )
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def main() -> None:
    setup_logging()
    args = parse_args()
    logger = logging.getLogger("bench_pipeline")

    data_loader = MeViSDataLoader(args.data_root)
    videos = data_loader.load()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    structured_log_path = output_dir / "structured_logs.jsonl"
    qa_path = output_dir / "qa_samples.jsonl"

    llm = DeepSeekExtractionInterface(api_key=args.api_key, base_url=args.api_base_url, model=args.extract_model)
    vtg = QwenVTGInterface(api_key=args.api_key, base_url=args.api_base_url, model=args.vtg_model)
    log_builder = StructuredLogBuilder(data_loader=data_loader, llm=llm, vtg=vtg)
    qa_generator = QAGenerator(log_builder=log_builder, seed=args.seed)

    structured_logs = []
    qa_samples = []
    sample_printed = False

    video_items = list(videos.items())[: max(args.max_videos, 0)] if args.max_videos else list(videos.items())
    logger.info("Processing %d videos", len(video_items))

    for index, (video_key, video) in enumerate(video_items, start=1):
        logger.info("[%d/%d] Processing video %s", index, len(video_items), video_key)
        try:
            structured_log = log_builder.build_for_video(video)
            if structured_log is None:
                logger.warning("No structured entities found for video=%s", video_key)
                continue

            structured_logs.append(structured_log.to_json_dict())
            qa_examples = qa_generator.generate_for_video(video, structured_log)
            qa_samples.extend([example.to_json_dict() for example in qa_examples])

            if qa_examples and not sample_printed:
                sample = qa_examples[0]
                logger.info("Sample QA question: %s", sample.question)
                logger.info("Sample QA answer: %s", sample.answer)
                sample_printed = True
        except Exception as exc:
            logger.warning("Failed to process video=%s err=%s", video_key, exc)
            continue

    with structured_log_path.open("w", encoding="utf-8") as f:
        for item in structured_logs:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with qa_path.open("w", encoding="utf-8") as f:
        for item in qa_samples:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    logger.info("Saved %d structured logs to %s", len(structured_logs), structured_log_path)
    logger.info("Saved %d QA samples to %s", len(qa_samples), qa_path)

    if qa_samples:
        first = qa_samples[0]
        print("\n=== Sample QA ===")
        print(f"Video: {first['video_id']}")
        print(f"Q: {first['question']}")
        print(f"A: {first['answer']}")
    else:
        print("No QA samples were generated. Check warnings in the log output.")


if __name__ == "__main__":
    main()
