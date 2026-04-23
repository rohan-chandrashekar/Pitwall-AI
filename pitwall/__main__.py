"""CLI entry point for PitWall-AI.

Usage:
    python -m pitwall --race bahrain-2024 --driver VER --speed 1000
    python -m pitwall --race live --driver VER
    python -m pitwall --train --seasons 2023 2024 2025
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys

from dotenv import load_dotenv
import numpy as np
import torch


def setup_logging():
    """Configure logging: INFO for pitwall, WARNING for everything else."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # Suppress noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("influxdb_client").setLevel(logging.ERROR)
    logging.getLogger("urllib3").setLevel(logging.ERROR)
    logging.getLogger("fastf1").setLevel(logging.WARNING)


def main():
    # Load .env FIRST so GROQ_API_KEY is available everywhere
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

    parser = argparse.ArgumentParser(
        description="PitWall-AI: F1 Race Strategy Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m pitwall --race bahrain-2024 --driver VER --speed 1000
  python -m pitwall --race monaco-2024 --driver HAM
  python -m pitwall --race live --driver VER
  python -m pitwall --train --seasons 2023 2024 2025

Groq LLM briefings:
  Set GROQ_API_KEY in .env or pass --groq-key to enable AI radio briefings.
  Free key: https://console.groq.com/keys

Default speed is 1000 (fast replay). Use --speed 1 for real-time.
        """,
    )

    parser.add_argument("--race", type=str, help="Race name (e.g., bahrain-2024) or 'live'")
    parser.add_argument("--driver", type=str, help="Driver code (e.g., VER, HAM, LEC)")
    parser.add_argument("--speed", type=float, default=1000.0,
                        help="Replay speed multiplier (default: 1000, fast replay)")
    parser.add_argument("--train", action="store_true",
                        help="Train tire model from OpenF1 data (no weather features)")
    parser.add_argument("--train-fastf1", action="store_true",
                        help="Train tire model from FastF1 data (with weather features)")
    parser.add_argument("--seasons", type=int, nargs="+", default=[2023, 2024, 2025],
                        help="Seasons to train on (default: 2023 2024 2025)")
    parser.add_argument("--groq-key", type=str, default=None,
                        help="Groq API key (alternative to GROQ_API_KEY env var)")

    args = parser.parse_args()
    setup_logging()
    logger = logging.getLogger("pitwall")

    # Set Groq key from CLI flag if provided (overrides .env)
    if args.groq_key:
        os.environ["GROQ_API_KEY"] = args.groq_key

    # Startup banner
    groq_status = "ON" if os.environ.get("GROQ_API_KEY") else "OFF (set GROQ_API_KEY or --groq-key)"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print()
    print("=" * 60)
    print("  PitWall-AI  |  F1 Race Strategy Engine")
    print("=" * 60)
    logger.info(f"PyTorch device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    logger.info(f"LLM briefings: {groq_status}")

    if args.train_fastf1:
        train_model_fastf1(args.seasons)
    elif args.train:
        asyncio.run(train_model(args.seasons))
    elif args.race and args.driver:
        live = args.race.lower() == "live"
        race_name = args.race if not live else "live"

        from pitwall.graph import RaceRunner
        runner = RaceRunner(
            race_name=race_name,
            driver_code=args.driver,
            speed_multiplier=args.speed,
            live=live,
        )
        asyncio.run(runner.run())
    else:
        parser.print_help()
        sys.exit(1)


async def train_model(seasons: list[int]):
    """Train the tire degradation model on historical data."""
    logger = logging.getLogger("pitwall.train")

    from pitwall.data.openf1_client import OpenF1Client
    from pitwall.models.tire_model import (
        TireModel,
        prepare_training_data_from_openf1,
        generate_training_data_from_profiles,
    )

    client = OpenF1Client()

    logger.info(f"Fetching training data for seasons: {seasons}")
    data = await prepare_training_data_from_openf1(client, seasons)
    await client.close()

    if len(data) < 1000:
        logger.warning(
            f"Only {len(data)} real samples found. "
            f"Augmenting with extra synthetic data from compound profiles."
        )

    # ALWAYS mix in synthetic data to anchor compound physics.
    # Real data is noisy: SOFT is used at race start (heavy fuel → slow laps),
    # HARD mid-race (light fuel → fast laps). Without synthetic anchoring the
    # network learns "SOFT = slow" from fuel correlation, not tire physics.
    # ~30% synthetic preserves the compound ordering signal without drowning
    # out real-world patterns the profiles can't capture.
    n_synthetic = max(5000, len(data) // 3)
    synthetic = generate_training_data_from_profiles(n_samples=n_synthetic)
    data.extend(synthetic)

    logger.info(f"Training samples: {len(data) - len(synthetic)} real + {len(synthetic)} synthetic = {len(data)} total")

    model = TireModel()
    metrics = model.train_on_data(data, epochs=80, lr=5e-4, batch_size=256)

    logger.info(f"Training complete: {metrics}")

    # Validate compound differentiation
    diff = metrics.get("compound_differentiation", {})
    if diff:
        soft_hard = abs(diff.get("HARD", 0) - diff.get("SOFT", 0))
        if soft_hard < 0.5:
            logger.error(
                f"COMPOUND DIFFERENTIATION FAILED: HARD-SOFT = {soft_hard:.3f}s. "
                f"The model cannot tell compounds apart. Strategy output will be meaningless."
            )
        else:
            logger.info(f"Compound differentiation OK: HARD-SOFT = {soft_hard:.3f}s")

    model.save()
    logger.info("Model saved to models/tire_deg.pt")


def train_model_fastf1(seasons: list[int]):
    """Train the tire model on FastF1 data with weather features (V2 model)."""
    logger = logging.getLogger("pitwall.train")

    from pitwall.data.fastf1_client import prepare_training_data_from_fastf1
    from pitwall.models.tire_model import (
        TireModel,
        FEATURE_DIM_V2,
        generate_training_data_from_profiles,
    )

    logger.info(f"FastF1 training: fetching data for seasons {seasons}")
    data = prepare_training_data_from_fastf1(seasons)

    if len(data) < 1000:
        logger.warning(
            f"Only {len(data)} FastF1 samples. "
            f"Try adding more seasons (data available from 2018+)."
        )

    # Mix synthetic data (with weather fields set to dry defaults)
    n_synthetic = max(5000, len(data) // 3)
    synthetic = generate_training_data_from_profiles(n_samples=n_synthetic)
    # Add weather fields to synthetic data (dry conditions)
    for s in synthetic:
        s["rainfall"] = 0.0
        s["humidity"] = 0.4 + (0.2 * np.random.random())  # 40-60%
        s["track_wetness"] = 0.0
    data.extend(synthetic)

    logger.info(
        f"Training samples: {len(data) - len(synthetic)} real (FastF1) + "
        f"{len(synthetic)} synthetic = {len(data)} total"
    )

    # Create V2 model with weather features
    model = TireModel(feature_dim=FEATURE_DIM_V2)
    metrics = model.train_on_data(data, epochs=80, lr=5e-4, batch_size=256)

    logger.info(f"Training complete: {metrics}")

    diff = metrics.get("compound_differentiation", {})
    if diff:
        soft_hard = abs(diff.get("HARD", 0) - diff.get("SOFT", 0))
        if soft_hard < 0.5:
            logger.error(
                f"COMPOUND DIFFERENTIATION FAILED: HARD-SOFT = {soft_hard:.3f}s. "
                f"The model cannot tell compounds apart."
            )
        else:
            logger.info(f"Compound differentiation OK: HARD-SOFT = {soft_hard:.3f}s")

    model.save()
    logger.info("V2 model saved to models/tire_deg.pt")


if __name__ == "__main__":
    main()
