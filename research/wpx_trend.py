"""Run the isolated causal-trajectory screen on corrected consumed history."""
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.wpx_objective import main, parse_args


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    main(parse_args(default_family="trend"))
