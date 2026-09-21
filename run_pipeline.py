"""Single-command entry point: the capstone's done-criterion.

Wiring only. Every stage is one of the modules in mini_reasoning/, and
the decisions behind them are recorded in docs/design-notes.md. What
this file owns is the order, the arguments each stage is given, and
where the records land.

    baseline eval (role 1)  -> self-consistency over those records
    -> GRPO training (role 2) -> validation eval (role 3)

Both evals grade with the same lenient extraction, which is what makes
before and after comparable; the reward role's boxed-only strictness is
the trainer's, passed there and not here. The self-consistency stage
re-reads the records the baseline already stored rather than generating
again, which is what storing the bundle bought.
"""
import argparse
import json
from pathlib import Path

from mini_reasoning import (
    consistency, evaluate as evaluation, scorers, trainer,
)

EVAL_FALLBACK = "number"


def build_parser() -> argparse.ArgumentParser:
    """The CLI. --n-samples sets the candidate set every stage sees."""
    parser = argparse.ArgumentParser(
        description="Qwen3-0.6B reasoning pipeline: baseline eval -> "
                    "self-consistency -> GRPO -> before/after eval")
    parser.add_argument("--subset", default="data/math500_subset50.jsonl")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    parser.add_argument("--steps", type=int, default=40,
                        help="GRPO training steps")
    parser.add_argument("--n-samples", type=int, default=7,
                        help="samples per problem: the candidate set both "
                             "self-consistency and the selection scorer see")
    parser.add_argument("--lr", type=float, default=1e-6,
                        help="Adam learning rate for GRPO")
    parser.add_argument("--seed", type=int, default=42,
                        help="seeds sampling, so a run repeats")
    parser.add_argument("--skip-training", action="store_true",
                        help="run the inference/eval stages only")
    parser.add_argument("--out-dir", default="results")
    return parser


def load_problems(path) -> list[dict]:
    """Reads the frozen subset, one problem per line.

    Args:
        path: Path to a jsonl file of MATH-500 problems.

    Returns:
        The problems in file order.

    Raises:
        FileNotFoundError: if the file is absent, naming the command
            that rebuilds it.
        ValueError: if it holds no problems.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"no subset at {path}. It is generated once and committed: "
            "run `uv run --group data python data/make_subset.py` to "
            "rebuild the seed-42 50-problem subset. Without --group data "
            "the script cannot import datasets, which is not a default "
            "dependency.")
    problems = [json.loads(line) for line in path.read_text().splitlines()
                if line.strip()]
    if not problems:
        raise ValueError(f"the subset at {path} holds no problems")
    return problems


def build_scorer(fallback: str | None = EVAL_FALLBACK) -> scorers.Composite:
    """The lab's composition verdict: gate, then rank, then agreement.

    Both extracting stages get the same fallback the evals grade with. A
    gate stricter than the grader after it would discard candidates that
    grader could have scored.
    """
    return scorers.Composite((
        scorers.ParseabilityGate(fallback=fallback),
        scorers.LogprobRank(),
        scorers.VoteAgreement(fallback=fallback),
    ))


def vote_accuracy(records) -> float:
    """Accuracy of self-consistency voting over already-stored records.

    Voting needs no model, so it runs over the candidates the baseline
    eval extracted rather than sampling again. A record nothing could be
    extracted from makes the vote raise by module policy, and counts as
    wrong here: there is no answer to be right with.
    """
    correct = 0
    for record in records:
        try:
            winner = consistency.self_consistency_vote(record["candidates"])
        except ValueError:
            continue
        correct += evaluation.verifier.verify_answer(winner, record["answer"])
    return correct / len(records)


def write_results(out_dir, name: str, results: dict) -> Path:
    """Writes one stage's results, records and all, under out_dir."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.json"
    path.write_text(json.dumps(results, indent=2) + "\n")
    return path


def _load_model(name: str):
    """Loads the policy in float32. The one step no test reaches.

    The dtype is passed rather than left to the library. transformers
    defaults to the checkpoint's own dtype, and Qwen3 ships bfloat16,
    whose representable step at these weights is an order of magnitude
    wider than Adam's 1e-6 update: the optimizer steps, reports a
    gradient norm, and rounds back to the same weights. float32 is what
    the trainer's 8.6 GB memory note and the README's timings measured.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    return (AutoModelForCausalLM.from_pretrained(name,
                                                 dtype=torch.float32),
            AutoTokenizer.from_pretrained(name))


def run(args, *, load_model=_load_model,
        sampler=evaluation.sample_solutions,
        rollout=trainer.sample_rollouts) -> dict:
    """Runs the stages the flags ask for and writes each one out.

    Args:
        args: Parsed arguments from build_parser.
        load_model: Returns (model, tokenizer) for a checkpoint name.
        sampler: Passed through to evaluate.
        rollout: Passed through to the trainer.

    Returns:
        The stage results: "baseline" and "vote_accuracy" always, plus
        "training" and "validation" when training ran.
    """
    problems = load_problems(args.subset)
    model, tokenizer = load_model(args.model)
    scorer = build_scorer()
    eval_args = dict(n_samples=args.n_samples, fallback=EVAL_FALLBACK,
                     sampler=sampler, seed=args.seed)

    baseline = evaluation.evaluate(model, tokenizer, problems, scorer,
                                   **eval_args)
    write_results(args.out_dir, "baseline", baseline)
    results = {"baseline": baseline,
               "vote_accuracy": vote_accuracy(baseline["records"])}

    if args.skip_training:
        return results

    history = trainer.train(model, tokenizer, problems, steps=args.steps,
                            rollout=rollout, group_size=args.n_samples,
                            lr=args.lr)
    write_results(args.out_dir, "training", history)
    results["training"] = history

    validation = evaluation.evaluate(model, tokenizer, problems, scorer,
                                     **eval_args)
    write_results(args.out_dir, "validation", validation)
    results["validation"] = validation
    return results


def main():
    args = build_parser().parse_args()
    results = run(args)
    print(f"baseline accuracy      {results['baseline']['accuracy']:.3f}")
    print("level-reweighted       "
          f"{results['baseline']['level_reweighted_accuracy']:.3f}")
    print(f"self-consistency vote  {results['vote_accuracy']:.3f}")
    if "validation" in results:
        stepped = sum(s["stepped"] for s in results["training"]["steps"])
        print(f"training steps taken   {stepped}"
              f" of {len(results['training']['steps'])}")
        print("validation accuracy    "
              f"{results['validation']['accuracy']:.3f}")
        print("level-reweighted       "
              f"{results['validation']['level_reweighted_accuracy']:.3f}")


if __name__ == "__main__":
    main()
