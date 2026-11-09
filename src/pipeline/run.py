"""Command-line interface for the HengSeng Sentiment Price Prediction Agent."""

import argparse
import sys
from loguru import logger
from src.pipeline.orchestrator import PipelineOrchestrator


def main():
    parser = argparse.ArgumentParser(
        description="HengSeng Sentiment Price Prediction Agent - Master Pipeline Runner"
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["ingest", "process", "train", "all"],
        help="Pipeline execution stage (default: all)",
    )

    args = parser.parse_args()
    logger.info(f"Starting pipeline runner with stage: {args.stage}")

    orchestrator = PipelineOrchestrator()

    try:
        if args.stage == "ingest":
            result = orchestrator.run_ingestion_stage()
        elif args.stage == "process":
            result = orchestrator.run_processing_stage()
        elif args.stage == "train":
            result = orchestrator.run_training_stage()
        elif args.stage == "all":
            result = orchestrator.run_all()
        else:
            logger.error(f"Unknown stage: {args.stage}")
            sys.exit(1)

        logger.info(f"Pipeline executed successfully. Results summary: {result}")
    except Exception as e:
        logger.exception(f"Pipeline execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
