from __future__ import annotations

import argparse
from pathlib import Path

from draftrl.pipeline import run


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a DRAFT-RL profile")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    artifact_root = run(args.config.resolve())
    print(f"DRAFT_RL_COMPLETE={artifact_root}")


if __name__ == "__main__":
    main()
