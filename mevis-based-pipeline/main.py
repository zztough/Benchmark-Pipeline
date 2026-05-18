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
        default="/home/zhaobing/bench-pipeline/mevis-based-pipeline/outputs/mevis_demo",
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
    parser.add_argument(
        "--use-da",
        action="store_true",
        help="Use Depth Anything 3 depth maps for forward/backward motion instead of mask area proxy.",
    )
    parser.add_argument(
        "--depth-model",
        type=str,
        default="depth-anything/DA3-GIANT-1.1",
        help="Hugging Face repo id or local directory for Depth Anything 3 weights.",
    )
    parser.add_argument(
        "--depth-device",
        type=str,
        default="cuda",
        help="Device for Depth Anything 3 inference.",
    )
    parser.add_argument(
        "--depth-process-res",
        type=int,
        default=504,
        help="Depth Anything 3 processing resolution.",
    )
    parser.add_argument(
        "--depth-debug-dir",
        type=str,
        default="/home/zhaobing/bench-pipeline/mevis-based-pipeline/outputs/depth_debug",
        help="Directory for depth-map visualization PNGs. Defaults to <output-dir>/depth_debug.",
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
    depth_estimator = None
    if args.use_da:
        from depth_estimator import DepthAnything3Estimator

        depth_estimator = DepthAnything3Estimator(
            model_name=args.depth_model,
            device=args.depth_device,
            process_res=args.depth_process_res,
            debug_dir=args.depth_debug_dir or str(output_dir / "depth_debug"),
        )
    log_builder = StructuredLogBuilder(data_loader=data_loader, llm=llm, vtg=vtg, depth_estimator=depth_estimator)
    qa_generator = QAGenerator(log_builder=log_builder, seed=args.seed)

    structured_log_count = 0
    qa_sample_count = 0
    first_sample = None
    sample_printed = False

    video_items = list(videos.items())[: max(args.max_videos, 0)] if args.max_videos else list(videos.items())
    logger.info("Processing %d videos", len(video_items))

    with structured_log_path.open("w", encoding="utf-8", buffering=1) as structured_log_file, qa_path.open(
        "w", encoding="utf-8", buffering=1
    ) as qa_file:
        for index, (video_key, video) in enumerate(video_items, start=1):
            logger.info("[%d/%d] Processing video %s", index, len(video_items), video_key)
            try:
                structured_log = log_builder.build_for_video(video)
                if structured_log is None:
                    logger.warning("No structured entities found for video=%s", video_key)
                    continue

                structured_log_file.write(json.dumps(structured_log.to_json_dict(), ensure_ascii=False) + "\n")
                structured_log_file.flush()
                structured_log_count += 1

                qa_examples = qa_generator.generate_for_video(video, structured_log)
                for example in qa_examples:
                    qa_item = example.to_json_dict()
                    qa_file.write(json.dumps(qa_item, ensure_ascii=False) + "\n")
                    qa_file.flush()
                    qa_sample_count += 1
                    if first_sample is None:
                        first_sample = qa_item

                if qa_examples and not sample_printed:
                    sample = qa_examples[0]
                    logger.info("Sample QA question: %s", sample.question)
                    logger.info("Sample QA answer: %s", sample.answer)
                    sample_printed = True
            except Exception as exc:
                logger.warning("Failed to process video=%s err=%s", video_key, exc)
                continue

    logger.info("Saved %d structured logs to %s", structured_log_count, structured_log_path)
    logger.info("Saved %d QA samples to %s", qa_sample_count, qa_path)

    if first_sample:
        first = first_sample
        print("\n=== Sample QA ===")
        print(f"Video: {first['video_id']}")
        print(f"Q: {first['question']}")
        print(f"A: {first['answer']}")
    else:
        print("No QA samples were generated. Check warnings in the log output.")


if __name__ == "__main__":
    main()
