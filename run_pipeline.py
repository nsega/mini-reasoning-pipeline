"""Single-command entry point: the capstone's done-criterion.

Scaffold (Claude-written wiring; stages are Naoki's modules). Each
stage raises NotImplementedError until its module lands, so the CLI
shape is testable before the pipeline is.
"""
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Qwen3-0.6B reasoning pipeline: baseline eval -> "
                    "self-consistency -> GRPO -> before/after eval")
    parser.add_argument("--subset", default="data/math500_subset50.jsonl")
    parser.add_argument("--steps", type=int, default=40,
                        help="GRPO training steps")
    parser.add_argument("--n-samples", type=int, default=7,
                        help="samples per problem for self-consistency")
    parser.add_argument("--skip-training", action="store_true",
                        help="run the inference/eval stages only")
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()

    from mini_reasoning import consistency, evaluate, grpo, scorers, verifier  # noqa: F401

    raise NotImplementedError(
        "pipeline stages land as mini_reasoning/ modules are implemented; "
        f"config: {vars(args)}")


if __name__ == "__main__":
    main()
