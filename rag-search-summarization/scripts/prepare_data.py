"""Step 1: download and pre-process the corpus.

    .venv/bin/python scripts/prepare_data.py                         # validation + test splits (~25k articles)
    .venv/bin/python scripts/prepare_data.py --splits train validation test   # full corpus (~310k articles)
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ragsearch.config import Settings  # noqa: E402
from ragsearch.data import prepare  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--splits", nargs="+", default=["validation", "test"], choices=["train", "validation", "test"])
args = parser.parse_args()

stats = prepare(Settings.from_env(), args.splits)
print(json.dumps(stats.__dict__, indent=2))
