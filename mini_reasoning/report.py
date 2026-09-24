"""Before against after, split by what training did to each problem.

The pipeline trains on the subset it evaluates, so a single before/after
number blends three populations that answer different questions: the
problems training stepped on, the ones it drew and skipped as flat, and
the ones it never drew. The first says whether the reward loop moved
the policy where it trained; the last is the only held-out signal there
is. The first full float32 run showed why they must stay apart: stepped
problems rose, untouched ones fell, and the blend read as a decline.

- Scores are per sample, not per selected answer. Every eval stores all
  of its samples, so the same run yields seven times the draws, and the
  selected answer adds the scorer's own variance on top of the model's.
- Two graders, one per role. "eval" grades each sample as the eval
  graded it, from the stored candidates, so it inherits whatever
  fallback that eval used. "reward" re-extracts boxed-only, the reward's
  strictness, which is what training optimised. Where they disagree is
  the bare-number rescue the eval grants and the reward does not.
- Groups are derived from the training history, not reserved. Untouched
  is whatever training happened not to reach, and at enough steps that
  is nothing: the report then says so rather than inventing a held-out
  set. A reserved split belongs with any change that draws beyond the
  first pass, such as resampling flat groups, which would reach it.
- Uncertainty is a paired sign-flip test on the per-problem changes,
  seeded, so the same records always give the same p.

Everything here is a pure function of stored material, so any run on
disk can be reported without a model: python -m mini_reasoning.report
<run-dir>.
"""
import argparse
import json
import random
from collections.abc import Mapping, Sequence
from pathlib import Path

from mini_reasoning import verifier

GROUPS = ("stepped", "flat", "untouched")
EMPTY = {
    "all": "none: there were no problems",
    "stepped": "none: no drawn problem gave a gradient",
    "flat": "none: every drawn problem stepped",
    "untouched": "none: training drew every problem, so nothing here is "
                 "held out",
}
METRICS = ("eval", "reward")
PERMUTATIONS = 20000
SEED = 0


def training_groups(problem_ids: Sequence[str],
                    steps: Sequence[Mapping]) -> dict[str, list[str]]:
    """Which problems training stepped on, skipped as flat, or never drew.

    Steps cycle the subset, so past its length a problem is drawn more
    than once. It counts as stepped if any of its draws stepped: the
    group asks whether the weights ever moved on it.

    Args:
        problem_ids: The subset's unique ids, in order.
        steps: The training history's per-step reports.

    Returns:
        Each group's ids, in subset order.
    """
    drawn = {s["unique_id"] for s in steps}
    stepped = {s["unique_id"] for s in steps if s["stepped"]}
    return {
        "stepped": [i for i in problem_ids if i in stepped],
        "flat": [i for i in problem_ids if i in drawn and i not in stepped],
        "untouched": [i for i in problem_ids if i not in drawn],
    }


def sample_scores(records: Sequence[Mapping]) -> dict[str, dict[str, float]]:
    """Per problem, the share of its samples each grader accepts.

    Args:
        records: Eval records, each holding its samples' texts and the
            candidates the eval extracted from them.

    Returns:
        By unique id, the "eval" and "reward" shares.
    """
    scores = {}
    for r in records:
        n = len(r["samples"])
        graded = sum(verifier.verify_answer(c, r["answer"])
                     for c in r["candidates"])
        rewarded = sum(
            verifier.verify_answer(
                verifier.extract_final_candidate(s["text"], fallback=None),
                r["answer"])
            for s in r["samples"])
        scores[r["unique_id"]] = {"eval": graded / n, "reward": rewarded / n}
    return scores


def signflip_p(differences: Sequence[float]) -> float:
    """Two-sided paired sign-flip p for a mean difference.

    Under no effect, each problem's change is as likely to have come out
    with the opposite sign, so the observed mean is ranked against means
    with the signs flipped at random. The count includes the observed
    arrangement itself, so p is never zero from a finite draw.
    """
    rng = random.Random(SEED)
    n = len(differences)
    observed = abs(sum(differences)) / n
    hits = sum(
        abs(sum(d if rng.random() < 0.5 else -d for d in differences)) / n
        >= observed - 1e-12
        for _ in range(PERMUTATIONS))
    return (hits + 1) / (PERMUTATIONS + 1)


def _compare(before, after, ids: Sequence[str]) -> dict:
    """One group's before, after, mean change and p, per grader."""
    result = {"n": len(ids)}
    for metric in METRICS:
        if not ids:
            result[metric] = None
            continue
        b = [before[i][metric] for i in ids]
        a = [after[i][metric] for i in ids]
        changes = [y - x for x, y in zip(b, a)]
        result[metric] = {"before": sum(b) / len(b),
                          "after": sum(a) / len(a),
                          "delta": sum(changes) / len(changes),
                          "p": signflip_p(changes)}
    return result


def report(baseline: Mapping, validation: Mapping, history: Mapping) -> dict:
    """The before/after comparison, whole and split by training's reach.

    Args:
        baseline: The baseline eval's results, records and all.
        validation: The validation eval's results over the same problems.
        history: The training history, {"steps": [...]}.

    Returns:
        "groups", each group's ids, and "comparisons", keyed by "all"
        and each group: its size, and per grader the before and after
        shares, their mean paired change and its p. An empty group
        carries None per grader, since a zero would read as a measured
        null.

    Raises:
        ValueError: if the two evals graded different problems.
    """
    ids = [r["unique_id"] for r in baseline["records"]]
    if [r["unique_id"] for r in validation["records"]] != ids:
        raise ValueError("before and after must grade the same problems in "
                         "the same order, or the pairs are not pairs")
    before = sample_scores(baseline["records"])
    after = sample_scores(validation["records"])
    groups = training_groups(ids, history["steps"])
    comparisons = {"all": _compare(before, after, ids)}
    for name in GROUPS:
        comparisons[name] = _compare(before, after, groups[name])
    return {"groups": groups, "comparisons": comparisons}


def format_table(result: Mapping) -> str:
    """The report as the lines a run ends with."""
    lines = ["per sample, before -> after (change, p)",
             f"{'group':10s} {'n':>3s}  {'eval':34s}  reward"]
    for name in ("all", *GROUPS):
        row = result["comparisons"][name]
        if row["n"] == 0:
            lines.append(f"{name:10s} {0:3d}  {EMPTY[name]}")
            continue
        cells = [f"{m['before']:.3f} -> {m['after']:.3f} "
                 f"({m['delta']:+.3f}, p {m['p']:.3f})"
                 for m in (row[k] for k in METRICS)]
        lines.append(f"{name:10s} {row['n']:3d}  {cells[0]:34s}  {cells[1]}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> None:
    """Reports a stored run and writes report.json beside it."""
    parser = argparse.ArgumentParser(
        description="report a stored run by what training did to each "
                    "problem")
    parser.add_argument("run_dir",
                        help="holds baseline.json, validation.json and "
                             "training.json")
    run_dir = Path(parser.parse_args(argv).run_dir)
    loaded = {}
    for name in ("baseline", "validation", "training"):
        path = run_dir / f"{name}.json"
        if not path.exists():
            raise FileNotFoundError(
                f"no {name}.json in {run_dir}. Only a run that trained "
                "writes all three.")
        loaded[name] = json.loads(path.read_text())
    result = report(loaded["baseline"], loaded["validation"],
                    loaded["training"])
    (run_dir / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(format_table(result))


if __name__ == "__main__":
    main()
