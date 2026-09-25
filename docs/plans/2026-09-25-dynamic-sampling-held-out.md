# Dynamic Sampling with a Reserved Held-Out Split Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Count GRPO training in optimizer updates instead of draws, retire a problem whose group comes out flat, reserve the subset's last ten problems from training, and report them as their own group.

**Architecture:** `mini_reasoning/trainer.py`'s loop changes from a fixed number of draws to a budget of updates, with retirement, three named stop conditions and per-draw seeding. `run_pipeline.py` splits the subset before the model loads, trains on the pool only, and writes the reserved ids into the training history. `mini_reasoning/report.py` reads that record to report a `held_out` group, and keeps the old three-group output for histories that predate it.

**Tech Stack:** Python 3.13, PyTorch 2.13, transformers 5.15, pytest, uv.

**Spec:** `docs/specs/2026-09-25-dynamic-sampling-held-out-design.md`

## Global Constraints

- Python follows the Google Python Style Guide: every new or changed function carries a docstring, with Args, Returns and Raises where they apply. Comments are optional.
- No em-dashes (—) in code, docstrings, docs or commit messages. En-dashes in numeric ranges and arrows are fine.
- Commits follow Conventional Commits 1.0.0. No `Co-Authored-By`, no session links, no "Generated with" lines.
- `--held-out` defaults to 10 and reserves the last N problems in subset file order. Both evals always run over every problem.
- `--steps` counts optimizer updates. `--max-draws` defaults to 3 × `--steps` and may sit below it.
- Stop reasons are exactly `"updates"`, `"max_draws"`, `"pool_exhausted"`, checked in that order.
- Each draw is seeded with `torch.manual_seed(seed + draw)`, `draw` counted from 0, when a seed is given.
- The history keeps its per-draw list under `"steps"` and adds `"updates"`, `"draws"`, `"stopped"`, and, from `run_pipeline.py`, `"held_out"`: always present, `[]` under `--held-out 0`.
- `docs/runs/2026-09-21-float32/report.json` must regenerate byte-identical.
- No estimated number replaces a measured one in `README.md`.
- Tests are written first and watched failing. Run the suite with `uv run --group dev pytest -q`.

## Review Focus

- An invalid `--held-out`, such as `--held-out 50` on the 50-problem subset, must fail before the model loads, not after the hour-long baseline eval. Pinned in Task 3.
- `--held-out 0` must reserve nothing, not everything: in Python `problems[-0:]` is the whole list, so the obvious slice silently holds out all fifty. Pinned in Task 3.
- A history in which training drew a reserved problem must be refused by the report, not reported in two groups at once. Pinned in Task 1.
- A pool where every problem goes flat must end with `pool_exhausted` and zero updates, and the run must still validate and report. Pinned in Tasks 2 and 3.
- `--reuse-baseline` with a split: a baseline stored over all fifty problems must still be accepted, because the split never touches the evals. Pinned in Task 3.

---

## File Structure

| file | change | responsibility after the change |
|---|---|---|
| `mini_reasoning/report.py` | modify | groups by training's reach, with a `held_out` group when the history records a split |
| `tests/test_report.py` | modify | the split groups, old-run compatibility, the reservation check |
| `mini_reasoning/trainer.py` | modify | the update budget, retirement, stop conditions, per-draw seeding |
| `tests/test_trainer.py` | modify | dynamic sampling, stopping, seeding |
| `run_pipeline.py` | modify | `--held-out`, `--max-draws`, the split, the history's `held_out`, the summary line |
| `tests/test_run_pipeline.py` | modify | the split and its wiring; existing helpers pass `--held-out 0` |
| `docs/design-notes.md` | modify | retirement in the flat-group section; a new held-out section |
| `README.md` | modify | the new flags and meaning of `--steps`; a caveat on the timings |

Tasks 1 and 2 are independent of each other. Task 3 consumes both. Task 4 documents them. Task 5 verifies with the real model.

---

### Task 1: The report reads a reserved held-out group

**Files:**
- Modify: `mini_reasoning/report.py` (module docstring bullet on groups; constants at lines 39-49; `training_groups` at 52-73; `report` at 135-163; `format_table` at 166-179)
- Test: `tests/test_report.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `SPLIT_GROUPS = ("stepped", "flat", "held_out", "untouched")`; `training_groups(problem_ids, steps, held_out=None) -> dict[str, list[str]]`, raising `ValueError` naming any reserved id that was drawn; `report(baseline, validation, history)` reading `history.get("held_out")`; `format_table(result)` iterating `result["groups"]`.

- [ ] **Step 0: Branch, and commit the spec this plan implements**

```bash
git checkout main && git pull --ff-only
git checkout -b dynamic-sampling
git add docs/specs/2026-09-25-dynamic-sampling-held-out-design.md docs/plans/2026-09-25-dynamic-sampling-held-out.md
git commit -m "docs(specs): add the dynamic sampling and held-out design" -m "The design and its implementation plan, committed ahead of the code so the branch argues from them."
```

- [ ] **Step 1: Write the failing tests**

In `tests/test_report.py`, add `from pathlib import Path` after `import json`, and replace the report import with:

```python
from mini_reasoning.report import (
    GROUPS, SPLIT_GROUPS, format_table, main, report, sample_scores,
    training_groups,
)

COMMITTED_RUN = (Path(__file__).resolve().parents[1]
                 / "docs" / "runs" / "2026-09-21-float32")
```

Append these classes to the end of the file:

```python
class TestHeldOut:
    """A reserved split, read from the history rather than inferred."""

    def test_a_split_history_yields_four_groups(self):
        groups = training_groups(
            ["p1", "p2", "p3", "p4"],
            [step("p1", True), step("p2", False)],
            held_out=["p4"])
        assert groups == {"stepped": ["p1"], "flat": ["p2"],
                          "held_out": ["p4"], "untouched": ["p3"]}

    def test_held_out_comes_from_the_reservation_not_the_draws(self):
        """Never drawn is what untouched means. Reserved is a promise, so
        a reserved problem is held out even when nothing else was drawn
        either."""
        groups = training_groups(["p1", "p2"], [], held_out=["p2"])
        assert groups["held_out"] == ["p2"]
        assert groups["untouched"] == ["p1"]

    def test_a_reserved_problem_that_was_drawn_is_refused(self):
        """The split exists so training cannot reach these problems. A
        history that reached one broke that guarantee, and reporting it
        in two groups at once would hide the breach."""
        with pytest.raises(ValueError, match="p2"):
            training_groups(["p1", "p2"], [step("p2", True)],
                            held_out=["p2"])

    def test_an_empty_reservation_still_uses_the_split_format(self):
        """--held-out 0 writes held_out: [], which says a split was asked
        for and none was made, rather than looking like an old run."""
        got = report(evaluation(scored("p1", 1)),
                     evaluation(scored("p1", 2)),
                     {"steps": [step("p1", True)], "held_out": []})
        assert tuple(got["groups"]) == SPLIT_GROUPS
        assert got["comparisons"]["held_out"]["n"] == 0

    def test_a_split_report_compares_every_split_group(self):
        got = report(evaluation(scored("p1", 1), scored("p2", 1)),
                     evaluation(scored("p1", 3), scored("p2", 2)),
                     {"steps": [step("p1", True)], "held_out": ["p2"]})
        assert set(got["comparisons"]) == {"all", *SPLIT_GROUPS}
        assert got["comparisons"]["held_out"]["reward"]["delta"] == (
            pytest.approx(0.25))


class TestOldRuns:
    """Runs written before the split keep the report they had."""

    def test_a_history_without_a_reservation_keeps_three_groups(self):
        got = report(evaluation(scored("p1", 1)),
                     evaluation(scored("p1", 2)),
                     {"steps": [step("p1", True)]})
        assert tuple(got["groups"]) == GROUPS

    def test_the_committed_run_regenerates_byte_identical(self):
        """The cleanest proof that the split moved no number in the
        analysis already published."""
        loaded = {name: json.loads((COMMITTED_RUN / f"{name}.json")
                                   .read_text())
                  for name in ("baseline", "validation", "training")}
        rebuilt = json.dumps(report(loaded["baseline"],
                                    loaded["validation"],
                                    loaded["training"]), indent=2) + "\n"
        assert rebuilt == (COMMITTED_RUN / "report.json").read_text()


class TestSplitTable:
    """The printed rows for a split run."""

    def test_an_empty_reservation_says_nothing_was_reserved(self):
        got = format_table(report(
            evaluation(scored("p1", 1)), evaluation(scored("p1", 2)),
            {"steps": [step("p1", True)], "held_out": []}))
        line = next(row for row in got.splitlines()
                    if row.startswith("held_out"))
        assert "nothing was reserved" in line

    def test_an_empty_untouched_speaks_only_of_the_pool(self):
        """Held-out problems have their own row in a split run, so the
        untouched row must not make claims about them."""
        got = format_table(report(
            evaluation(scored("p1", 1), scored("p2", 1)),
            evaluation(scored("p1", 2), scored("p2", 1)),
            {"steps": [step("p1", True)], "held_out": ["p2"]}))
        line = next(row for row in got.splitlines()
                    if row.startswith("untouched"))
        assert "in its pool" in line and "held out" not in line
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run --group dev pytest tests/test_report.py -q`
Expected: collection ERROR, `ImportError: cannot import name 'SPLIT_GROUPS'`.

- [ ] **Step 3: Implement**

In `mini_reasoning/report.py`, replace the module docstring bullet that begins `- Groups are derived from the training history, not reserved.` (five lines) with:

```text
- Groups come from the training history. A run with a reserved split
  records it as held_out, and those problems form their own group,
  checked never to have been drawn; untouched is then only the pool
  problems training happened not to reach. A history written before
  splits existed has no held_out and keeps the three groups it was
  reported with.
```

Replace the constants block (`GROUPS = ...` through the closing brace of `EMPTY`) with:

```python
GROUPS = ("stepped", "flat", "untouched")
SPLIT_GROUPS = ("stepped", "flat", "held_out", "untouched")
EMPTY = {
    "all": "none: there were no problems",
    "stepped": "none: no drawn problem gave a gradient",
    "flat": "none: every drawn problem stepped",
    "held_out": "none: nothing was reserved",
    "untouched": "none: training drew every problem, so nothing here is "
                 "held out",
}
SPLIT_UNTOUCHED = "none: training drew every problem in its pool"
```

Replace `training_groups` with:

```python
def training_groups(problem_ids: Sequence[str], steps: Sequence[Mapping],
                    held_out: Sequence[str] | None = None
                    ) -> dict[str, list[str]]:
    """Which problems training stepped on, skipped as flat, or never drew.

    Steps cycle the pool, so a problem can be drawn more than once. It
    counts as stepped if any of its draws stepped: the group asks
    whether the weights ever moved on it.

    Args:
        problem_ids: The subset's unique ids, in order.
        steps: The training history's per-draw reports.
        held_out: The ids a split reserved, or None for a history
            written before splits existed. Given, the reserved problems
            form their own group and untouched narrows to pool problems
            training never drew.

    Returns:
        Each group's ids, in subset order: stepped, flat and untouched,
        with held_out before untouched when a split was recorded.

    Raises:
        ValueError: if training drew a problem the split reserved.
    """
    drawn = {s["unique_id"] for s in steps}
    stepped = {s["unique_id"] for s in steps if s["stepped"]}
    groups = {
        "stepped": [i for i in problem_ids if i in stepped],
        "flat": [i for i in problem_ids if i in drawn and i not in stepped],
    }
    if held_out is None:
        groups["untouched"] = [i for i in problem_ids if i not in drawn]
        return groups
    reserved = set(held_out)
    breached = sorted(reserved & drawn)
    if breached:
        raise ValueError(f"training drew reserved problems {breached}, so "
                         "the held-out split was not held out")
    groups["held_out"] = [i for i in problem_ids if i in reserved]
    groups["untouched"] = [i for i in problem_ids
                           if i not in drawn and i not in reserved]
    return groups
```

In `report`, update the docstring's `history:` line to `history: The training history: {"steps": [...]}, plus "held_out" for a run that reserved a split.`, add `if training drew a problem the split reserved.` to its Raises, and replace the three lines from `groups = training_groups(...)` through the `for` loop with:

```python
    groups = training_groups(ids, history["steps"], history.get("held_out"))
    comparisons = {"all": _compare(before, after, ids)}
    for name in groups:
        comparisons[name] = _compare(before, after, groups[name])
```

In `format_table`, replace the loop header and the empty-row branch with:

```python
    split = "held_out" in result["groups"]
    for name in ("all", *result["groups"]):
        row = result["comparisons"][name]
        if row["n"] == 0:
            message = (SPLIT_UNTOUCHED if split and name == "untouched"
                       else EMPTY[name])
            lines.append(f"{name:10s} {0:3d}  {message}")
            continue
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run --group dev pytest -q`
Expected: all pass, including `test_the_committed_run_regenerates_byte_identical`.

- [ ] **Step 5: Commit**

```bash
git add mini_reasoning/report.py tests/test_report.py
git commit -m "feat(report): report a reserved held-out group" -m "A run that reserves a split records it in its history, and the report now gives those problems their own group, refusing a history that drew one. Histories from before the split keep their three groups, and the committed run's report regenerates byte-identical."
```

---

### Task 2: The trainer counts updates and retires flat problems

**Files:**
- Modify: `mini_reasoning/trainer.py` (module docstring, the bullet ending at line 24; `train` from line 189)
- Test: `tests/test_trainer.py`

**Interfaces:**
- Consumes: `train_step(...)` unchanged, returning `{"unique_id", "mean_reward", "grad_norm", "stepped"}`.
- Produces: `train(model, tokenizer, problems, *, steps, rollout, group_size=7, lr=1e-6, max_grad_norm=MAX_GRAD_NORM, max_draws=None, seed=None) -> dict` returning `{"steps": [per-draw reports], "updates": int, "draws": int, "stopped": "updates" | "max_draws" | "pool_exhausted"}`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_trainer.py`:

```python
SPLIT, ALL_WRONG, ALL_RIGHT = [RIGHT, WRONG], [WRONG, WRONG], [RIGHT, RIGHT]


def problem(uid):
    return dict(PROBLEM, unique_id=uid, problem=f"problem {uid}")


def rollout_by(table):
    """Rollout texts looked up by problem statement."""
    return lambda model, tokenizer, statement, n: rollouts_of(table[statement])


def drawn(history):
    return [s["unique_id"] for s in history["steps"]]


class TestDynamicSampling:
    """steps counts updates, and a flat problem retires for the run."""

    def test_steps_counts_updates_not_draws(self):
        history = train(policy(), None, [problem("flat"), problem("live")],
                        steps=2,
                        rollout=rollout_by({"problem flat": ALL_WRONG,
                                            "problem live": SPLIT}),
                        group_size=2, lr=1e-3)
        assert history["updates"] == 2 and history["draws"] == 3
        assert drawn(history) == ["flat", "live", "live"]

    def test_a_retired_problem_is_never_drawn_again(self):
        history = train(policy(), None, [problem("flat"), problem("live")],
                        steps=4,
                        rollout=rollout_by({"problem flat": ALL_WRONG,
                                            "problem live": SPLIT}),
                        group_size=2, lr=1e-3)
        assert drawn(history).count("flat") == 1

    def test_an_all_right_group_retires_like_an_all_wrong_one(self):
        """All right is as gradient-free as all wrong."""
        history = train(policy(), None, [problem("solved"), problem("live")],
                        steps=3,
                        rollout=rollout_by({"problem solved": ALL_RIGHT,
                                            "problem live": SPLIT}),
                        group_size=2, lr=1e-3)
        assert drawn(history).count("solved") == 1

    def test_a_problem_that_steps_then_goes_flat_retires(self):
        """A third draw of "fading" would exhaust its outcomes and raise,
        so this also fails loudly if retirement is skipped."""
        outcomes = iter([SPLIT, ALL_WRONG])

        def rollout(model, tokenizer, statement, n):
            if statement == "problem fading":
                return rollouts_of(next(outcomes))
            return rollouts_of(SPLIT)

        history = train(policy(), None, [problem("fading"), problem("live")],
                        steps=4, rollout=rollout, group_size=2, lr=1e-3)
        assert [s["stepped"] for s in history["steps"]
                if s["unique_id"] == "fading"] == [True, False]


class TestStopping:
    """Whichever limit comes first ends the run, and says which it was."""

    def test_reaching_the_target_stops_on_updates(self):
        history = train(policy(), None, [problem("live")], steps=2,
                        rollout=rollout_by({"problem live": SPLIT}),
                        group_size=2, lr=1e-3)
        assert history["stopped"] == "updates" and history["draws"] == 2

    def test_the_draw_cap_stops_on_max_draws_even_below_steps(self):
        """A runtime cap, not a validation rule, so it may sit under
        steps."""
        history = train(policy(), None, [problem("live")], steps=5,
                        max_draws=2,
                        rollout=rollout_by({"problem live": SPLIT}),
                        group_size=2, lr=1e-3)
        assert history["stopped"] == "max_draws"
        assert history["updates"] == 2 and history["draws"] == 2

    def test_a_pool_that_all_goes_flat_stops_on_pool_exhausted(self):
        history = train(policy(), None, [problem("a"), problem("b")],
                        steps=3,
                        rollout=rollout_by({"problem a": ALL_WRONG,
                                            "problem b": ALL_WRONG}),
                        group_size=2, lr=1e-3)
        assert history["stopped"] == "pool_exhausted"
        assert history["updates"] == 0 and history["draws"] == 2

    def test_max_draws_defaults_to_three_times_steps(self):
        pool = [problem(str(i)) for i in range(5)]
        history = train(policy(), None, pool, steps=1,
                        rollout=rollout_by({f"problem {i}": ALL_WRONG
                                            for i in range(5)}),
                        group_size=2, lr=1e-3)
        assert history["draws"] == 3 and history["stopped"] == "max_draws"

    def test_a_draw_cap_below_one_raises(self):
        with pytest.raises(ValueError, match="max_draws"):
            train(policy(), None, [PROBLEM], steps=1, max_draws=0,
                  rollout=rollout_of(SPLIT), group_size=2)


def random_rollout(model, tokenizer, statement, n):
    """Rewards drawn from torch's global RNG with nothing reseeding it,
    so only the trainer's own seeding can make two runs agree."""
    bits = torch.randint(0, 2, (n,)).tolist()
    ids = torch.randint(0, VOCAB, (n, PROMPT_LEN + 4))
    return Rollouts(sequences=ids, prompt_len=PROMPT_LEN,
                    texts=[RIGHT if bit else WRONG for bit in bits])


class TestSeeding:
    """Each draw is seeded from the run's seed plus its index."""

    def run(self, seed, disturbance):
        model = policy()
        torch.manual_seed(disturbance)  # whatever the process did first
        history = train(model, None, [problem(str(i)) for i in range(6)],
                        steps=4, rollout=random_rollout, group_size=4,
                        lr=1e-3, seed=seed)
        return [(s["unique_id"], s["mean_reward"], s["stepped"])
                for s in history["steps"]]

    def test_the_same_seed_repeats_the_run_whatever_came_before(self):
        """The --reuse-baseline gap: skipping the baseline eval leaves the
        RNG somewhere else, and training must not notice."""
        assert self.run(7, disturbance=1) == self.run(7, disturbance=2)

    def test_a_different_seed_draws_differently(self):
        assert self.run(7, disturbance=1) != self.run(8, disturbance=1)

    def test_draw_k_is_seeded_with_seed_plus_k(self):
        seen = []

        def recording(model, tokenizer, statement, n):
            seen.append(torch.initial_seed())
            return rollouts_of(SPLIT)

        train(policy(), None, [problem("live")], steps=3,
              rollout=recording, group_size=2, lr=1e-3, seed=40)
        assert seen == [40, 41, 42]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run --group dev pytest tests/test_trainer.py -q`
Expected: the new tests FAIL (`KeyError: 'updates'`, and `TypeError: train() got an unexpected keyword argument` for `max_draws` and `seed`). The existing `TestTrain` tests still pass.

- [ ] **Step 3: Implement**

In the `mini_reasoning/trainer.py` module docstring, after the bullet ending `an all-same-reward group must not move anything.`, add:

```text
- A flat problem is also retired, and the budget counts updates rather
  than draws, so training makes further passes over the problems that
  still give gradient instead of spending draws on ones that do not.
```

Replace `train` entirely with:

```python
def train(model, tokenizer, problems, *, steps: int, rollout,
          group_size: int = 7, lr: float = 1e-6,
          max_grad_norm: float = MAX_GRAD_NORM,
          max_draws: int | None = None, seed: int | None = None) -> dict:
    """Runs GRPO until `steps` updates, cycling the problems in order.

    A draw whose group is flat takes no step and retires its problem for
    the rest of the run, whether every rollout was wrong or every one
    was right: neither gives a gradient, and at this scale a problem
    flat once is almost always flat again. The budget counts updates
    rather than draws, so the loop makes further passes over the
    problems that still give gradient, until the first of three limits:
    `steps` updates, `max_draws` draws, or no live problem left.

    Each draw is seeded from `seed` plus its index, counting from 0, so
    which problems retire repeats, and a run that skipped the baseline
    eval trains exactly like one that ran it.

    Args:
        model: The policy to train, in place.
        tokenizer: Passed through to the rollout function.
        problems: The training pool, cycled in order.
        steps: Optimizer updates to reach.
        rollout: Returns Rollouts for (model, tokenizer, problem, n).
        group_size: Rollouts per problem.
        lr: Adam learning rate.
        max_grad_norm: The clip bound.
        max_draws: A runtime cap on draws; None means 3 * steps. It may
            sit below steps.
        seed: Seeds each draw as seed + draw; None leaves the global
            RNG alone.

    Returns:
        "steps", one report per draw in order (the name predates dynamic
        sampling, and stored runs depend on it), and "updates", "draws"
        and "stopped": which limit ended the run, "updates",
        "max_draws" or "pool_exhausted", checked in that order.

    Raises:
        ValueError: if steps or max_draws is below one, or problems is
            empty.
    """
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")
    if not problems:
        raise ValueError("train needs at least one problem")
    if max_draws is None:
        max_draws = 3 * steps
    if max_draws < 1:
        raise ValueError(f"max_draws must be >= 1, got {max_draws}")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    pool = {p["unique_id"] for p in problems}
    history, retired = [], set()
    updates = draws = position = 0
    while updates < steps and draws < max_draws and retired != pool:
        problem = problems[position % len(problems)]
        position += 1
        if problem["unique_id"] in retired:
            continue
        if seed is not None:
            torch.manual_seed(seed + draws)
        report = train_step(model, optimizer, problem, tokenizer,
                            rollout=rollout, group_size=group_size,
                            max_grad_norm=max_grad_norm)
        history.append(report)
        draws += 1
        if report["stepped"]:
            updates += 1
        else:
            retired.add(problem["unique_id"])
    if updates == steps:
        stopped = "updates"
    elif draws == max_draws:
        stopped = "max_draws"
    else:
        stopped = "pool_exhausted"
    return {"steps": history, "updates": updates, "draws": draws,
            "stopped": stopped}
```

`retired != pool` compares sets of ids, so a pool that repeats an id still terminates once that id retires.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run --group dev pytest -q`
Expected: all pass. If `test_a_different_seed_draws_differently` fails, seeds 7 and 8 happened to give identical histories. The test is deterministic, not flaky, so change 8 to another value and note it in the commit.

- [ ] **Step 5: Commit**

```bash
git add mini_reasoning/trainer.py tests/test_trainer.py
git commit -m "feat(trainer): count updates and retire flat problems" -m "A budget of draws stopped the first float32 run after one pass, with 17 of its 40 draws spent on flat groups. steps now counts optimizer updates, a flat problem retires for the run, and the loop stops on updates, a draw cap or an exhausted pool, recording which. Each draw is seeded from the run's seed, so retirement repeats and a --reuse-baseline run trains like a full one."
```

---

### Task 3: The pipeline reserves the held-out split

**Files:**
- Modify: `run_pipeline.py` (module docstring; `build_parser` at line 32, the `--steps` argument at 39; a new `split_held_out` after `load_problems`; `run` at 215, lines 230 and 252-255; `main` at 269, line 277)
- Test: `tests/test_run_pipeline.py` (`args_for` at line 79; the hand-built argv lists in `test_a_baseline_from_other_problems_is_refused` and `test_a_baseline_at_another_sample_count_is_refused`; new classes)

**Interfaces:**
- Consumes: `trainer.train(..., max_draws=..., seed=...)` and its `updates`, `draws`, `stopped` keys from Task 2; `report.report` reading `history["held_out"]` from Task 1.
- Produces: `split_held_out(problems: list[dict], n: int) -> tuple[list[dict], list[dict]]`; CLI flags `--held-out` (int, default 10) and `--max-draws` (int, default None).

- [ ] **Step 1: Write the failing tests**

In `tests/test_run_pipeline.py`, add `from pathlib import Path` after `import json`, replace the `run_pipeline` import with:

```python
from run_pipeline import (
    build_parser, build_scorer, load_problems, run, split_held_out,
    vote_accuracy,
)

ROOT = Path(__file__).resolve().parents[1]
COMMITTED_RUN = ROOT / "docs" / "runs" / "2026-09-21-float32"
```

Replace `args_for`'s first lines with:

```python
def args_for(tmp_path, **overrides):
    """Parsed args over the two-problem subset. --held-out is 0 because
    the default of ten would reserve both problems."""
    argv = ["--subset", str(subset_file(tmp_path)),
            "--out-dir", str(tmp_path / "results"), "--n-samples", "3",
            "--held-out", "0"]
```

In both `test_a_baseline_from_other_problems_is_refused` and `test_a_baseline_at_another_sample_count_is_refused`, add `"--held-out", "0",` to the hand-built argv list, just before `"--reuse-baseline"`.

In `TestCli`, add:

```python
    def test_the_split_defaults_to_ten_and_the_draw_cap_to_unset(self):
        args = build_parser().parse_args([])
        assert args.held_out == 10 and args.max_draws is None
```

Append these classes to the end of the file:

```python
class TestHeldOutSplit:
    """The last N problems are reserved; only the rest are trained on."""

    def test_it_reserves_the_last_n(self):
        assert split_held_out([P1, P2], 1) == ([P1], [P2])

    def test_zero_reserves_nothing(self):
        """problems[-0:] is the whole list in Python, so the obvious slice
        holds out everything when asked to hold out nothing."""
        assert split_held_out([P1, P2], 0) == ([P1, P2], [])

    def test_a_negative_count_raises(self):
        with pytest.raises(ValueError, match="held-out"):
            split_held_out([P1, P2], -1)

    def test_a_count_that_leaves_nothing_to_train_on_raises(self):
        with pytest.raises(ValueError, match="nothing to train"):
            split_held_out([P1, P2], 2)

    def test_the_default_holds_out_what_the_committed_run_never_drew(self):
        """The committed run is this one's control only while the default
        reserves exactly the problems it left untouched. If the subset's
        order or the default moves, this is where it shows."""
        defaults = build_parser().parse_args([])
        problems = load_problems(ROOT / defaults.subset)
        _, held_out = split_held_out(problems, defaults.held_out)
        steps = json.loads((COMMITTED_RUN / "training.json")
                           .read_text())["steps"]
        drawn = {s["unique_id"] for s in steps}
        assert [p["unique_id"] for p in held_out] == [
            p["unique_id"] for p in problems if p["unique_id"] not in drawn]


def recording_rollout(seen):
    """canned_rollout, noting which problem statements training reached."""
    def rollout(model, tokenizer, problem, n):
        seen.append(problem)
        return canned_rollout(model, tokenizer, problem, n)
    return rollout


def all_wrong_rollout(model, tokenizer, problem, n):
    torch.manual_seed(0)
    return Rollouts(sequences=torch.randint(0, 32, (n, 7)), prompt_len=3,
                    texts=[r"\boxed{0}"] * n)


class TestHeldOutWiring:
    """The split reaches training and nothing else."""

    def test_training_never_sees_a_held_out_problem(self, tmp_path):
        seen = []
        run(args_for(tmp_path, **{"--steps": 3, "--held-out": 1}),
            load_model=loader, sampler=canned,
            rollout=recording_rollout(seen))
        assert set(seen) == {P1["problem"]}

    def test_both_evals_still_cover_every_problem(self, tmp_path):
        results = run(args_for(tmp_path, **{"--steps": 1, "--held-out": 1}),
                      load_model=loader, sampler=canned,
                      rollout=canned_rollout)
        for stage in ("baseline", "validation"):
            assert [r["unique_id"] for r in results[stage]["records"]] == [
                "p1", "p2"]

    def test_the_reserved_ids_are_written_with_the_history(self, tmp_path):
        run(args_for(tmp_path, **{"--steps": 1, "--held-out": 1}),
            load_model=loader, sampler=canned, rollout=canned_rollout)
        written = json.loads(
            (tmp_path / "results" / "training.json").read_text())
        assert written["held_out"] == ["p2"]
        assert {"updates", "draws", "stopped"} <= set(written)

    def test_the_report_carries_the_held_out_group(self, tmp_path):
        results = run(args_for(tmp_path, **{"--steps": 1, "--held-out": 1}),
                      load_model=loader, sampler=canned,
                      rollout=canned_rollout)
        assert results["report"]["groups"]["held_out"] == ["p2"]

    def test_a_bad_split_fails_before_the_model_loads(self, tmp_path):
        """An hour of baseline eval is the price of finding out late."""
        def never(name):
            raise AssertionError("the model loaded before the split was "
                                 "checked")
        with pytest.raises(ValueError, match="nothing to train"):
            run(args_for(tmp_path, **{"--held-out": 2}), load_model=never,
                sampler=canned)

    def test_a_pool_that_all_goes_flat_still_validates_and_reports(
            self, tmp_path):
        """Zero updates is a result, not a crash: the run still ends in a
        validation eval and a report saying nothing stepped."""
        results = run(args_for(tmp_path, **{"--steps": 2}),
                      load_model=loader, sampler=canned,
                      rollout=all_wrong_rollout)
        assert results["training"]["stopped"] == "pool_exhausted"
        assert results["report"]["groups"]["stepped"] == []
        assert "validation" in results

    def test_reuse_baseline_accepts_a_baseline_over_every_problem(
            self, tmp_path):
        """The split never touches the evals, so a baseline stored before
        it is still this run's before half."""
        run(args_for(tmp_path, **{"--skip-training": True}),
            load_model=loader, sampler=canned)
        results = run(args_for(tmp_path, **{"--reuse-baseline": True,
                                            "--held-out": 1,
                                            "--steps": 1}),
                      load_model=loader, sampler=canned,
                      rollout=canned_rollout)
        assert results["report"]["groups"]["held_out"] == ["p2"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run --group dev pytest tests/test_run_pipeline.py -q`
Expected: collection ERROR, `ImportError: cannot import name 'split_held_out'`.

- [ ] **Step 3: Implement**

In the `run_pipeline.py` module docstring, add this paragraph after the one ending `at the same hour they cost the first time.`:

```text
The held-out split is also the entry point's. The last --held-out
problems are kept from training and from nothing else, so both evals
still grade the whole subset, and the split is checked before the model
loads, since a bad one found after the baseline eval costs an hour.
```

In `build_parser`, replace the `--steps` argument with:

```python
    parser.add_argument("--steps", type=int, default=40,
                        help="GRPO optimizer updates. A flat group takes "
                             "no update and retires its problem, so "
                             "training draws until this many updates, "
                             "--max-draws, or no live problem is left")
    parser.add_argument("--max-draws", type=int, default=None,
                        help="cap on training draws, a runtime backstop; "
                             "default 3 x --steps")
    parser.add_argument("--held-out", type=int, default=10,
                        help="reserve the last N problems of the subset "
                             "from training; both evals still cover "
                             "every problem")
```

After `load_problems`, add:

```python
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
```

Slice at `len(problems) - n`, never `-n`: `problems[-0:]` is the whole list.

In `run`, replace `problems = load_problems(args.subset)` with:

```python
    problems = load_problems(args.subset)
    pool, held_out = split_held_out(problems, args.held_out)
```

and replace the `trainer.train(...)` call and the `write_results(args.out_dir, "training", history)` line after it with:

```python
    history = trainer.train(model, tokenizer, pool, steps=args.steps,
                            rollout=rollout, group_size=args.n_samples,
                            lr=args.lr, max_draws=args.max_draws,
                            seed=args.seed)
    history["held_out"] = [p["unique_id"] for p in held_out]
    write_results(args.out_dir, "training", history)
```

In `main`, replace the two statements that compute `stepped` and print `training steps taken` with:

```python
        training = results["training"]
        print(f"training updates       {training['updates']} over "
              f"{training['draws']} draws (stopped: {training['stopped']})")
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run --group dev pytest -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add run_pipeline.py tests/test_run_pipeline.py
git commit -m "feat(pipeline): reserve a held-out split from training" -m "Dynamic sampling makes further passes over the pool, which would reach the problems the first float32 run left untouched. --held-out reserves the last N, ten by default, which is exactly that set, so the committed run stays this one's control. Both evals still cover every problem, the split is checked before the model loads, and the history records what was reserved."
```

---

### Task 4: Document retirement and the held-out split

**Files:**
- Modify: `docs/design-notes.md` (section "What the trainer does with a flat group", lines 284-303; a new section before "## What the entry point owns")
- Modify: `README.md` (the Reproduction section and the timing paragraph)

**Interfaces:** none.

- [ ] **Step 1: Apply the edits**

Run this from the repository root. Each replacement asserts it matched exactly once, so a drifted document fails loudly instead of being edited in the wrong place. The README paragraphs are wrapped with `break_on_hyphens=False`: the default would split a flag such as `--max-draws` across two lines at its hyphen.

```bash
uv run python - <<'PY'
import textwrap
from pathlib import Path


def replace(path, old, new):
    p = Path(path)
    src = p.read_text()
    assert src.count(old) == 1, f"{path}: matched {src.count(old)} times"
    p.write_text(src.replace(old, new))


notes = "docs/design-notes.md"
replace(notes,
    "Every other group runs one backward per rollout, each scaled by 1/G, then a clip to norm 1.0 and one step.\n",
    "Every other group runs one backward per rollout, each scaled by 1/G, then a clip to norm 1.0 and one step. A skipped group also retires its problem for the rest of the run, and the budget counts optimizer updates rather than draws, so training passes over the pool again for the problems that still give gradient.\n")
replace(notes,
    "Raising on a flat group. The trainer steps through them by design; with a binary reward a problem the policy always gets right, or always gets wrong, is an ordinary thing to encounter.\n",
    "Raising on a flat group. The trainer steps through them by design; with a binary reward a problem the policy always gets right, or always gets wrong, is an ordinary thing to encounter.\n"
    "Redrawing a flat problem on every pass. It protects a problem that was flat by chance, since one solved 10% of the time comes out flat 48% of the time at seven rollouts, but it pays a generation per redraw on problems that are almost all out of reach.\n")
replace(notes,
    "(3) With no KL term there is nothing else acting on a zero-advantage step, which is what makes \"no step\" the correct behaviour rather than merely a cheap one.\n",
    "(3) With no KL term there is nothing else acting on a zero-advantage step, which is what makes \"no step\" the correct behaviour rather than merely a cheap one.\n"
    "(4) In the first float32 run, 17 of 40 draws were flat, and 16 of those 17 problems also scored zero boxed reward in the independent baseline eval: at this scale a problem flat once is flat for good, so retiring it costs about one wrongly retired problem in seventeen.\n")
replace(notes,
    "A run's step count is an upper bound rather than a count of updates, so the history has to say which steps actually moved the policy, and it does. A schedule keyed to the optimizer's step count advances differently from one keyed to the loop's, and nothing here uses a schedule yet.\n",
    "Updates are exact and draws are the upper bound: `--steps` counts optimizer updates, `--max-draws` caps the draws spent reaching them, and the history records which limit ended the run. About one chance-flat problem in seventeen is retired wrongly. A schedule keyed to the optimizer's step count now agrees with the budget, and nothing here uses one yet.\n")
replace(notes,
    "The first fails if the step is taken on zeros, which is the mistake the test exists for.\n",
    "The first fails if the step is taken on zeros, which is the mistake the test exists for. `TestDynamicSampling`, `TestStopping` and `TestSeeding` in the same file pin retirement, the three limits in their order, and the per-draw seed.\n")
replace(notes, "## What the entry point owns\n",
    "## The held-out split\n\n"
    "**Decision.**\n"
    "The last `--held-out` problems of the frozen subset, ten by default, are kept from training and from nothing else: both evals still grade all fifty. The last ten are exactly the problems the first float32 run never drew, so that run is this one's control, with the same pool, the same held-out set and the same eval, and the one variable between them is how training reaches the pool.\n\n"
    "**Rejected.**\n"
    "An external training pool from MATH's training split. Every eval problem would be held out and the pool would never run dry, but the train-on-test measurement the report keeps would be lost, and a second frozen artifact would need its own generator and pins.\n"
    "A level-stratified seeded draw, at ten or twenty. More representative, or more sensitive, but either breaks comparability with the committed run by changing the pool and the held-out set at once.\n"
    "Pre-filtering training problems from the stored baseline. It selects with the eval's own draw, so validation regresses toward the mean on exactly the groups the report compares.\n\n"
    "**Evidence.**\n"
    "(1) The held-out result can only ever be directional. The per-problem change in boxed reward has a standard deviation of 0.124 in the committed run, so a held-out group of k problems detects, at 80% power, an effect of about 2.8 × 0.124 / √k: 10.9 points at ten, 7.7 at twenty, 4.9 at all fifty. The trained problems moved 6.2 points, and a generalization effect is normally smaller than that.\n"
    "(2) Dynamic sampling makes the reservation necessary rather than cosmetic. The first run left problems 41-50 untouched only because forty draws in file order stop at forty, and a second pass would have reached them.\n\n"
    "**Cost accepted.**\n"
    "A held-out gain or loss here is a direction, not a finding, whatever its p. The split stays because it keeps training off those ten problems and keeps the report's two claims apart.\n\n"
    "**Pinned by.**\n"
    "`test_the_default_holds_out_what_the_committed_run_never_drew` in `tests/test_run_pipeline.py` reads the committed run's training history, so the control relationship fails loudly if the subset's order or the default ever moves. `test_a_reserved_problem_that_was_drawn_is_refused` in `tests/test_report.py` refuses a history that broke the reservation.\n\n"
    "## What the entry point owns\n")

readme = Path("README.md")
src = readme.read_text()
anchor = "stored sample rather than one selected answer per problem.\n"
assert src.count(anchor) == 1
new_para = textwrap.fill(
    "Training draws only from the first 40 problems: `--held-out 10`, "
    "the default, reserves the last ten, and both evals still grade all "
    "fifty. `--steps` counts optimizer updates rather than draws. A "
    "problem whose seven rollouts all score the same gives no gradient "
    "and is retired for the rest of the run, so training passes over the "
    "pool again until it reaches 40 updates, runs out of live problems, "
    "or hits `--max-draws`, three times `--steps` by default.", width=70,
    break_on_hyphens=False, break_long_words=False)
src = src.replace(anchor, anchor + "\n" + new_para + "\n")
start = src.index("Expect roughly 70 seconds per problem")
end = src.index("\n\nA note on what the numbers mean", start)
para = " ".join(src[start:end].split())
marker = "two and a half hours in all."
assert para.count(marker) == 1
para = para.replace(marker, marker + " Those were measured before dynamic "
                    "sampling, when training stopped at 40 draws; it now "
                    "runs to 40 optimizer updates, so the training stage "
                    "takes longer, and the next measured run replaces "
                    "these figures.")
readme.write_text(src[:start] + textwrap.fill(
    para, width=70, break_on_hyphens=False, break_long_words=False)
    + src[end:])
print("docs updated")
PY
```

- [ ] **Step 2: Check the result**

Run: `git diff --stat && grep -c '—' README.md docs/design-notes.md`
Expected: two files changed. The em-dash counts must not exceed what `git show HEAD:README.md | grep -c '—'` and `git show HEAD:docs/design-notes.md | grep -c '—'` report, since the edits add none.

Run: `uv run --group dev pytest -q`
Expected: all pass.

- [ ] **Step 3: Commit**

```bash
git add docs/design-notes.md README.md
git commit -m "docs: record retirement and the held-out split's limits" -m "The design notes gain retirement in the flat-group section, the inverted draws-and-updates cost, and a held-out section whose power table says its result is directional only. The README documents --held-out and the new meaning of --steps, and marks its timings as measured before dynamic sampling rather than replacing them with an estimate."
```

---

### Task 5: Verify with the real model

**Files:** none changed.

**Interfaces:**
- Consumes: the finished branch.

- [ ] **Step 1: The full suite**

Run: `uv run --group dev pytest -q`
Expected: all pass.

- [ ] **Step 2: A tiny real run through the pipeline**

```bash
S=$(mktemp -d)
head -n 3 data/math500_subset50.jsonl > "$S/three.jsonl"
uv run python run_pipeline.py --subset "$S/three.jsonl" --n-samples 2 \
  --steps 2 --held-out 1 --out-dir "$S/out"
uv run python - "$S/out" <<'PY'
import json
import sys

out = sys.argv[1]
training = json.load(open(f"{out}/training.json"))
report = json.load(open(f"{out}/report.json"))
held = training["held_out"]
assert len(held) == 1, held
assert held[0] not in {s["unique_id"] for s in training["steps"]}, \
    "a held-out problem was drawn"
assert {"updates", "draws", "stopped"} <= set(training), sorted(training)
assert report["groups"]["held_out"] == held
print("ok:", training["updates"], "updates over", training["draws"],
      "draws, stopped:", training["stopped"])
PY
```

Expected: about four minutes. The run ends with a summary line of the form `training updates       N over M draws (stopped: ...)` and a table with a `held_out` row, then the check prints `ok:`. At two samples many groups are flat, so `pool_exhausted` with zero updates is a valid outcome here. What is being verified is the mechanics, not a result.

- [ ] **Step 3: The committed report is untouched**

```bash
uv run python -m mini_reasoning.report docs/runs/2026-09-21-float32
git diff --exit-code docs/runs/2026-09-21-float32/report.json && echo "byte-identical"
```

Expected: `byte-identical`.

- [ ] **Step 4: Nothing to commit**

Run: `git status --short`
Expected: empty. Verification writes only to the temporary directory and regenerates one file identically.
