import argparse
import sys
from pathlib import Path

import yaml

from utils.batch_processing import build_runtime_config, run_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run batch LIF processing from a YAML config file."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing input .lif containers.",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML configuration file.",
    )
    return parser.parse_args()


def load_yaml_config(config_path: Path) -> dict:
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("Top-level YAML config must be a mapping/dictionary.")
    return data


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    config_path = Path(args.config)

    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {input_dir}")

    user_config = load_yaml_config(config_path)
    runtime_config = build_runtime_config(user_config, raw_data_directory=input_dir)
    result = run_batch(runtime_config)

    print(
        "Batch complete | "
        f"containers={result['containers']} | "
        f"processed={result['processed']} | "
        f"success={result['success']} | "
        f"skipped={result['skipped']} | "
        f"failed={result['failed']} | "
        f"elapsed_min={result['elapsed_min']:.2f}"
    )

    return 1 if result["failed"] > 0 else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
