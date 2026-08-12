"""Regenerate the frozen 50-problem MATH-500 evaluation subset.

Scaffold script (Claude-written; data prep, not pipeline code). Same
recipe as the study lab: HuggingFaceH4/MATH-500 test split, 50 indices
via random.Random(42), sorted, original fields plus source_index.
Byte-identical output to the lab's frozen subset; committed once and
never changed mid-experiment.
"""
import json
import random
from pathlib import Path

from datasets import load_dataset

OUT = Path(__file__).resolve().parent / "math500_subset50.jsonl"
SEED = 42
K = 50


def main():
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    idx = sorted(random.Random(SEED).sample(range(len(ds)), K))
    with open(OUT, "w") as f:
        for i in idx:
            rec = dict(ds[i])
            rec["source_index"] = i
            f.write(json.dumps(rec) + "\n")
    print(f"wrote {OUT} ({K} problems, seed {SEED})")


if __name__ == "__main__":
    main()
