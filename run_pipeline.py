"""Single-command entry point: the capstone's done-criterion.

Wiring only. Every stage is one of the modules in mini_reasoning/, and
the decisions behind them are recorded in docs/design-notes.md. What
this file owns is the order, the arguments each stage is given, and
where the records land.

    baseline eval (role 1)  -> self-consistency over those records
    -> GRPO training (role 2) -> validation eval (role 3)
    -> the before/after, split by what training did to each problem

Both evals grade with the same lenient extraction, which is what makes
before and after comparable; the reward role's boxed-only strictness is
the trainer's, passed there and not here. The self-consistency stage
re-reads the records the baseline already stored rather than generating
again, which is what storing the bundle bought. --reuse-baseline
extends that bargain to a whole stage: the baseline eval is
deterministic in its seed, so a run whose training died reproduces the
records it already wrote, at the same hour they cost the first time.

The held-out split is also the entry point's. The last --held-out
problems are kept from training and from nothing else, so both evals
still grade the whole subset, and the split is checked before the model
loads, since a bad one found after the baseline eval costs an hour.
"""
import argparse
import json
from pathlib import Path

from mini_reasoning import (
    consistency, evaluate as evaluation, report, scorers, trainer,
)

EVAL_FALLBACK = "number"


def positive_int(text: str) -> int:
    """An argparse type for counts that must be at least one.

    Checked at parse time because the stage that would otherwise refuse
    the value runs after the model load and the hour-long baseline eval.

    Raises:
        argparse.ArgumentTypeError: if the value is below one.
    """
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {value}")
    return value


def build_parser() -> argparse.ArgumentParser:
    """The CLI. --n-samples sets the candidate set every stage sees."""
    parser = argparse.ArgumentParser(
        description="Qwen3-0.6B reasoning pipeline: baseline eval -> "
                    "self-consistency -> GRPO -> before/after eval")
    parser.add_argument("--subset", default="data/math500_subset50.jsonl")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    parser.add_argument("--steps", type=positive_int, default=40,
                        help="GRPO optimizer updates. A flat group takes "
                             "no update and retires its problem, so "
                             "training draws until this many updates, "
                             "--max-draws, or no live problem is left")
    parser.add_argument("--max-draws", type=positive_int, default=None,
                        help="cap on training draws, a runtime backstop; "
                             "default 3 x --steps")
    parser.add_argument("--held-out", type=int, default=10,
                        help="reserve the last N problems of the subset "
                             "from training; both evals still cover "
                             "every problem")
    parser.add_argument("--n-samples", type=int, default=7,
                        help="samples per problem: the candidate set both "
                             "self-consistency and the selection scorer see")
    parser.add_argument("--lr", type=float, default=1e-6,
                        help="Adam learning rate for GRPO")
    parser.add_argument("--seed", type=int, default=42,
                        help="seeds sampling, so a run repeats")
    parser.add_argument("--skip-training", action="store_true",
                        help="run the inference/eval stages only")
    parser.add_argument("--reuse-baseline", action="store_true",
                        help="read the baseline stored under --out-dir "
                             "instead of running the baseline eval. The "
                             "stored provenance is checked against this "
                             "run, so a baseline from another model, "
                             "dtype or seed is refused")
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


COMPARED = ("model", "dtype", "seed", "n_samples", "fallback")


def split_held_out(problems: list[dict],
                   n: int) -> tuple[list[dict], list[dict]]:
    """Reserves the last n problems from training.

    "Last" is stable because the subset's bytes, order included, are
    pinned by its test. The default of ten reserves exactly the problems
    the first float32 run never drew, which is what makes that run this
    one's control: same pool, same held-out set, same eval.

    Args:
        problems: The subset, in file order.
        n: How many to reserve; 0 reserves nothing.

    Returns:
        (pool, held_out): what training may draw, and what it may not.

    Raises:
        ValueError: if n is negative, or leaves nothing to train on.
    """
    if n < 0:
        raise ValueError(f"--held-out must be >= 0, got {n}")
    if n >= len(problems):
        raise ValueError(f"--held-out {n} reserves all {len(problems)} "
                         "problems, leaving nothing to train on")
    cut = len(problems) - n
    return problems[:cut], problems[cut:]


def load_baseline(out_dir, problems, expected: dict) -> dict:
    """Reads a stored baseline back, checking it fits this run.

    Two kinds of check, in that order. The records are read first, for
    the problems they graded and the width they drew, because those are
    evidence. The provenance stamp is read second, because it is a
    claim, and a file whose claim disagrees with its records fails on
    the records.

    A stamp is required, not merely compared. Records written before
    evaluate stamped them cannot be checked at all, and this repo's
    rule is that a missing argument fails toward strictness: refusing
    them costs an hour, reusing one silently costs a before half drawn
    from another model, which is the failure that put the stamp here.

    Args:
        out_dir: Where write_results put it.
        problems: The problems this run is about to grade.
        expected: This run's stamp, from evaluate.provenance. The
            fields in COMPARED must match; the library versions are
            recorded on both sides and compared on neither.

    Returns:
        The stored baseline results, shaped as evaluate returned them.

    Raises:
        FileNotFoundError: if no baseline is stored there.
        ValueError: if it graded other problems, drew another width,
            carries no stamp, or carries one that disagrees.
    """
    path = Path(out_dir) / "baseline.json"
    if not path.exists():
        raise FileNotFoundError(
            f"no baseline at {path} to reuse. Run once without "
            "--reuse-baseline to write one.")
    baseline = json.loads(path.read_text())
    records = baseline["records"]
    if [r["unique_id"] for r in records] != [p["unique_id"]
                                             for p in problems]:
        raise ValueError(
            f"the baseline at {path} graded different problems than the "
            "subset holds, so it cannot be this run's before half")
    widths = {len(r["samples"]) for r in records}
    if widths != {expected["n_samples"]}:
        raise ValueError(
            f"the baseline at {path} drew {sorted(widths)} samples per "
            f"problem, not the {expected['n_samples']} this run asks for")
    stamp = baseline.get("provenance")
    if stamp is None:
        raise ValueError(
            f"the baseline at {path} carries no provenance, so the "
            "model and dtype behind it cannot be checked. It predates "
            "the stamp: run once without --reuse-baseline to write one "
            "that can be.")
    for field in COMPARED:
        if stamp.get(field) != expected.get(field):
            raise ValueError(
                f"the baseline at {path} was drawn with "
                f"{field}={stamp.get(field)!r} and this run has "
                f"{field}={expected.get(field)!r}, so it is a before "
                "half from another run")
    return baseline


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
    the README's measured timings and memory figure assume.
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
        "training", "validation" and "report" when training ran.
    """
    problems = load_problems(args.subset)
    # The split only concerns training, so an eval-only run skips it; a
    # default of ten would otherwise refuse any smaller smoke subset.
    pool, held_out = split_held_out(
        problems, 0 if args.skip_training else args.held_out)
    model, tokenizer = load_model(args.model)
    scorer = build_scorer()
    eval_args = dict(n_samples=args.n_samples, fallback=EVAL_FALLBACK,
                     sampler=sampler, seed=args.seed)

    if args.reuse_baseline:
        baseline = load_baseline(args.out_dir, problems,
                                 evaluation.provenance(
                                     model, seed=args.seed,
                                     n_samples=args.n_samples,
                                     fallback=EVAL_FALLBACK))
    else:
        baseline = evaluation.evaluate(model, tokenizer, problems, scorer,
                                       **eval_args)
        write_results(args.out_dir, "baseline", baseline)
    results = {"baseline": baseline,
               "vote_accuracy": vote_accuracy(baseline["records"])}

    if args.skip_training:
        return results

    # The evals seed problem i with seed + i. Training's stream starts past
    # the last of those: sharing one would replay the baseline's own
    # samples, and first-pass retirement would be decided on that draw.
    training_seed = args.seed + len(problems)
    history = trainer.train(model, tokenizer, pool, steps=args.steps,
                            rollout=rollout, group_size=args.n_samples,
                            lr=args.lr, max_draws=args.max_draws,
                            seed=training_seed)
    history["held_out"] = [p["unique_id"] for p in held_out]
    history["seed"] = training_seed
    write_results(args.out_dir, "training", history)
    results["training"] = history

    validation = evaluation.evaluate(model, tokenizer, problems, scorer,
                                     **eval_args)
    write_results(args.out_dir, "validation", validation)
    results["validation"] = validation

    breakdown = report.report(baseline, validation, history)
    write_results(args.out_dir, "report", breakdown)
    results["report"] = breakdown
    return results


def main():
    args = build_parser().parse_args()
    results = run(args)
    print(f"baseline accuracy      {results['baseline']['accuracy']:.3f}")
    print("level-reweighted       "
          f"{results['baseline']['level_reweighted_accuracy']:.3f}")
    print(f"self-consistency vote  {results['vote_accuracy']:.3f}")
    if "validation" in results:
        training = results["training"]
        print(f"training updates       {training['updates']} over "
              f"{training['draws']} draws (stopped: {training['stopped']})")
        print("validation accuracy    "
              f"{results['validation']['accuracy']:.3f}")
        print("level-reweighted       "
              f"{results['validation']['level_reweighted_accuracy']:.3f}")
        print()
        print(report.format_table(results["report"]))


if __name__ == "__main__":
    main()
