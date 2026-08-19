from __future__ import annotations

import argparse
from pathlib import Path

from draftrl.smoke import run


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a DRAFT-RL method-smoke profile")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    artifact_root = run(args.config.resolve())
    print(f"METHOD_SMOKE_COMPLETE={artifact_root}")


if __name__ == "__main__":
    main()
