# Checkpoint and Resume for GRPO Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a training run that is killed resume to records identical to an uninterrupted run, and let baseline, training and validation run as separate processes, all opt-in through `--checkpoint-dir`.

**Architecture:** `mini_reasoning/trainer.py` learns to resume (history, pool position, Adam state) and to call an injected checkpointer every N updates and at the stop, while never touching disk. A new `mini_reasoning/checkpoint.py` owns the files: atomic saves under `LATEST`, identity checks, the weights handoff and the lost-work log. `run_pipeline.py` adds `--checkpoint-dir`, `--checkpoint-every` and `--stage`, resumes automatically, and checks everything that can refuse a run before the model loads.

**Tech Stack:** Python 3.13, PyTorch 2.13, transformers 5.15, safetensors 0.8, pytest, uv.

**Spec:** `docs/specs/2026-10-09-checkpoint-resume-design.md`

## Global Constraints

- Python follows the Google Python Style Guide: every new or changed function carries a docstring, with Args, Returns and Raises where they apply. Comments are optional.
- No em-dashes (—) in code, docstrings, docs or commit messages. En-dashes in numeric ranges and arrows are fine.
- Commits follow Conventional Commits 1.0.0 and end with a `Co-Authored-By: Claude <model> <noreply@anthropic.com>` trailer naming the model that wrote them. No session links, no "Generated with" lines.
- Checkpointing is opt-in. Without `--checkpoint-dir`, every command behaves exactly as before, and every existing test passes unchanged.
- No record format changes. `baseline.json`, `training.json`, `validation.json` and `report.json` from a resumed or staged run are byte-identical to one uninterrupted process's. `progress.json` and `resumes.jsonl` live only in the checkpoint directory.
- The default cadence is 5 optimizer updates.
- Checkpoint layout: `LATEST`, `draw-NNNN/` (draw count, zero-padded to four digits) holding `model.safetensors`, `optimizer.pt` and `state.json`, plus `progress.json` and `resumes.jsonl`. `state.json` carries `"format": 1`.
- Identity: `model`, `seed` (the training seed), `steps`, `max_draws` (resolved, never None), `group_size`, `lr`, `pool`, compared before the model loads. `dtype` is compared when the weights load, against the loaded model.
- Everything that can refuse a run is checked before the model loads, the dtype excepted.
- `safetensors>=0.8` is declared directly. Resolved versions do not change.
- Tests are written first and watched failing. Run the suite with `uv run --group dev pytest -q`. It holds 208 tests before this plan and 259 after.

## Review Focus

- A save that fails while writing its files, for example on a full disk, must leave the previous checkpoint current. Pinned in Task 2.
- A resume started with a different `--checkpoint-every` must still write identical records, since a save changes nothing it saves. Pinned in Task 3.
- A stored `baseline.json` that automatic reuse finds but that came from another dtype must be refused, not reused. Pinned in Task 3.
- `--stage validate` given a different split from the train stage must be refused before the model loads, naming `pool`. Pinned in Task 3.
- A finished full run started again, say by a pod that finished as it was evicted, must draw nothing and leave every record as it was. Pinned in Task 3.

---

## File Structure

| file | change | responsibility after the change |
|---|---|---|
| `mini_reasoning/trainer.py` | modify | the loop, plus `draw_cap`, `Resume`, the `Checkpointer` protocol, resume and save cadence |
| `tests/test_trainer.py` | modify | resume exactness, Adam as load-bearing, cadence, refusals |
| `mini_reasoning/checkpoint.py` | create | checkpoint files: `Writer`, `read_latest`, `load_weights`, `resume_from`, `log_resume` |
| `tests/test_checkpoint.py` | create | round trip, tied weights, atomicity, identity, lost work |
| `pyproject.toml`, `uv.lock` | modify | `safetensors` declared directly |
| `run_pipeline.py` | modify | the three flags, the stages, automatic resume, pre-load checks |
| `tests/test_run_pipeline.py` | modify | flags, kill and resume, stage split, validate refusals |
| `README.md` | modify | a Checkpoints and stages section, and the Provenance updates |
| `docs/design-notes.md` | modify | a Checkpoint and resume section, and its provenance line |
| `.gitignore` | modify | `checkpoints/`, where the README's examples write |

Task 2 consumes Task 1's `Resume`. Task 3 consumes both. Task 4 documents them. Task 5 verifies with the real model.

---

### Task 1: The trainer resumes and says when to save

**Files:**
- Modify: `mini_reasoning/trainer.py` (module docstring; after `MAX_GRAD_NORM`; the whole of `train`)
- Test: `tests/test_trainer.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `DRAWS_PER_STEP = 3`; `draw_cap(steps: int, max_draws: int | None) -> int`; `Resume(history: list[dict], position: int, optimizer_state: dict)`, a frozen dataclass; `Checkpointer`, a protocol with `progress(draws: int, updates: int) -> None` and `save(optimizer, history: list[dict], position: int, *, complete: bool) -> None`; `train(..., resume: Resume | None = None, checkpointer: Checkpointer | None = None, checkpoint_every: int = 5) -> dict`, raising `ValueError` for a resume without a seed, a resumed history outside the pool, and `checkpoint_every < 1`.

- [ ] **Step 0: Work on the branch that carries the spec and this plan**

```bash
git switch checkpoint-resume
git log --oneline -3
```

Expected: the two newest commits add `docs/plans/2026-10-09-checkpoint-resume.md` and `docs/specs/2026-10-09-checkpoint-resume-design.md`, on top of `main`.

Run: `uv run --group dev pytest -q`
Expected: `208 passed`.

- [ ] **Step 1: Write the failing tests**

In `tests/test_trainer.py`, add `import copy` above `import pytest`, separated by a blank line, and replace the trainer import with:

```python
from mini_reasoning.trainer import (
    Resume, Rollouts, draw_cap, rollout_rewards, sequence_logprobs, train,
    train_step,
)
```

Append to the end of the file, after two blank lines:

```python
class Recorder:
    """A Checkpointer that keeps what each save would write, in memory."""

    def __init__(self, model):
        self.model = model
        self.saves, self.progress_calls = [], []

    def progress(self, draws, updates):
        self.progress_calls.append((draws, updates))

    def save(self, optimizer, history, position, *, complete):
        self.saves.append({
            "weights": copy.deepcopy(self.model.state_dict()),
            "optimizer": copy.deepcopy(optimizer.state_dict()),
            "history": copy.deepcopy(history), "position": position,
            "complete": complete,
            "updates": sum(s["stepped"] for s in history)})


def resume_from(save):
    return Resume(history=save["history"], position=save["position"],
                  optimizer_state=save["optimizer"])


class TestResume:
    """A run stopped at a save and resumed in fresh objects ends exactly
    where an unbroken run does: weights, Adam state and every draw."""

    POOL = [problem(str(i)) for i in range(4)]
    SETTINGS = dict(steps=6, rollout=random_rollout, group_size=3, lr=1e-2,
                    seed=12)

    def unbroken(self):
        model = policy()
        recorder = Recorder(model)
        history = train(model, None, self.POOL, checkpointer=recorder,
                        checkpoint_every=2, **self.SETTINGS)
        return model, history, recorder

    def test_a_resumed_run_ends_where_an_unbroken_one_does(self):
        """Resumed from the save after update 2, so the draws after it
        are replayed, flat ones included: the per-draw seeds and the
        restored retired set are both under test."""
        model, history, recorder = self.unbroken()
        save = recorder.saves[0]
        assert save["updates"] == 2 and not save["complete"]
        stepped = [s["stepped"] for s in history["steps"]]
        cut = len(save["history"])
        assert not all(stepped[:cut]) and not all(stepped[cut:]), (
            "the run must hold flat draws before and after the save, or "
            "this test no longer exercises the retired set")

        resumed = policy(seed=1)
        resumed.load_state_dict(save["weights"])
        torch.manual_seed(99)  # whatever the new process did first
        got = train(resumed, None, self.POOL, resume=resume_from(save),
                    **self.SETTINGS)
        assert got == history
        for want, have in zip(model.parameters(), resumed.parameters()):
            assert torch.equal(want, have)

    def test_restarting_adam_instead_changes_the_run(self):
        """The negative control: the optimizer state is load-bearing."""
        model, history, recorder = self.unbroken()
        save = recorder.saves[0]
        fresh = torch.optim.Adam(policy().parameters(), lr=1e-2)
        resumed = policy(seed=1)
        resumed.load_state_dict(save["weights"])
        got = train(resumed, None, self.POOL,
                    resume=Resume(history=save["history"],
                                  position=save["position"],
                                  optimizer_state=fresh.state_dict()),
                    **self.SETTINGS)
        assert got["steps"] != history["steps"]

    def test_a_complete_resume_returns_the_run_and_saves_nothing(self):
        """How the entry point reads a finished checkpoint back."""
        model, history, recorder = self.unbroken()
        final = recorder.saves[-1]
        assert final["complete"]
        resumed = policy(seed=1)
        resumed.load_state_dict(final["weights"])
        again = Recorder(resumed)
        got = train(resumed, None, self.POOL, resume=resume_from(final),
                    checkpointer=again, **self.SETTINGS)
        assert got == history
        assert again.saves == [] and again.progress_calls == []

    def test_a_resume_needs_the_seed(self):
        optimizer = torch.optim.Adam(policy().parameters())
        with pytest.raises(ValueError, match="seed"):
            train(policy(), None, [PROBLEM], steps=1,
                  rollout=rollout_of(SPLIT), group_size=2,
                  resume=Resume(history=[], position=0,
                                optimizer_state=optimizer.state_dict()))

    def test_a_resume_from_another_pool_is_refused(self):
        optimizer = torch.optim.Adam(policy().parameters())
        stray = [{"unique_id": "elsewhere", "mean_reward": 0.5,
                  "grad_norm": 1.0, "stepped": True}]
        with pytest.raises(ValueError, match="elsewhere"):
            train(policy(), None, [PROBLEM], steps=2,
                  rollout=rollout_of(SPLIT), group_size=2, seed=0,
                  resume=Resume(history=stray, position=1,
                                optimizer_state=optimizer.state_dict()))


class TestCheckpointCadence:
    """When the trainer asks for a save, and what it reports."""

    def run(self, steps, every):
        model = policy()
        recorder = Recorder(model)
        history = train(model, None, [problem("live")], steps=steps,
                        rollout=rollout_by({"problem live": SPLIT}),
                        group_size=2, lr=1e-3, seed=0, checkpointer=recorder,
                        checkpoint_every=every)
        return history, recorder

    def test_saves_fall_on_every_nth_update_and_at_the_stop(self):
        _, recorder = self.run(steps=5, every=2)
        assert [(s["updates"], s["complete"]) for s in recorder.saves] == [
            (2, False), (4, False), (5, True)]

    def test_a_save_due_on_the_stopping_draw_is_the_final_one(self):
        _, recorder = self.run(steps=4, every=2)
        assert [(s["updates"], s["complete"]) for s in recorder.saves] == [
            (2, False), (4, True)]

    def test_flat_draws_after_a_save_get_a_final_save_of_their_own(self):
        """Same update count, more draws: the reason checkpoints are
        named by draw. "fading" steps, then goes flat, and "flat" never
        steps, so the pool runs out one update after the last save."""
        outcomes = iter([SPLIT, ALL_WRONG])

        def rollout(model, tokenizer, statement, n):
            if statement == "problem fading":
                return rollouts_of(next(outcomes))
            return rollouts_of(ALL_WRONG)

        model = policy()
        recorder = Recorder(model)
        history = train(model, None, [problem("fading"), problem("flat")],
                        steps=5, rollout=rollout, group_size=2, lr=1e-3,
                        seed=0, checkpointer=recorder, checkpoint_every=1)
        assert history["stopped"] == "pool_exhausted"
        assert [(len(s["history"]), s["updates"], s["complete"])
                for s in recorder.saves] == [(1, 1, False), (3, 1, True)]

    def test_progress_follows_every_draw(self):
        history, recorder = self.run(steps=3, every=2)
        assert recorder.progress_calls == [(1, 1), (2, 2), (3, 3)]

    def test_a_cadence_below_one_raises(self):
        with pytest.raises(ValueError, match="checkpoint_every"):
            train(policy(), None, [PROBLEM], steps=1,
                  rollout=rollout_of(SPLIT), group_size=2,
                  checkpoint_every=0)

    def test_the_draw_cap_defaults_to_three_times_steps(self):
        assert draw_cap(40, None) == 120 and draw_cap(40, 7) == 7
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run --group dev pytest -q tests/test_trainer.py`
Expected: a collection error, `ImportError: cannot import name 'Resume' from 'mini_reasoning.trainer'`.

- [ ] **Step 3: Implement**

In `mini_reasoning/trainer.py`, add `from typing import Protocol` on the line after `from dataclasses import dataclass`.

Append this bullet to the module docstring's list, after the bullet that ends "instead of spending draws on ones that do not.":

```
- A resume is exact because every draw reseeds from its index: train()
  takes back the history, the pool position and Adam's state, and the
  draws after a checkpoint replay as the interrupted run made them. The
  files are checkpoint.py's; the loop only says when to save.
```

Insert after `MAX_GRAD_NORM = 1.0`, keeping two blank lines before the existing `@dataclass(frozen=True)` above `class Rollouts`:

```python
DRAWS_PER_STEP = 3


def draw_cap(steps: int, max_draws: int | None) -> int:
    """The draw cap train() enforces: max_draws, or 3 x steps when unset.

    Exposed so a checkpoint's identity records the cap the loop really
    used, and a run that names the default explicitly is the same run.
    """
    return DRAWS_PER_STEP * steps if max_draws is None else max_draws


@dataclass(frozen=True)
class Resume:
    """Where a checkpointed run picks up.

    The weights are not here: the caller loads them into the model
    before calling train(), which owns the optimizer and so restores
    its state itself.

    Attributes:
        history: The per-draw reports made before the checkpoint, in
            order, as train() returns them under "steps".
        position: The pool index the loop examines next.
        optimizer_state: Adam's state_dict at the checkpoint.
    """
    history: list[dict]
    position: int
    optimizer_state: dict


class Checkpointer(Protocol):
    """What train() calls to record progress and save its state.

    The trainer decides when, since the cadence is counted in updates.
    The checkpointer decides how, and holds the model it saves.
    """

    def progress(self, draws: int, updates: int) -> None:
        """Records how far training has got. Called after every draw."""

    def save(self, optimizer, history: list[dict], position: int, *,
             complete: bool) -> None:
        """Saves the full training state.

        Args:
            optimizer: The optimizer whose state is saved.
            history: The per-draw reports so far.
            position: The pool index the loop examines next.
            complete: True on the save made when the loop stops.
        """
```

Replace the whole of `train`, from `def train(` to the end of the file, with:

```python
def train(model, tokenizer, problems, *, steps: int, rollout,
          group_size: int = 7, lr: float = 1e-6,
          max_grad_norm: float = MAX_GRAD_NORM,
          max_draws: int | None = None, seed: int | None = None,
          resume: Resume | None = None,
          checkpointer: Checkpointer | None = None,
          checkpoint_every: int = 5) -> dict:
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
    eval trains exactly like one that ran it. The same seeding is what
    makes a resume exact: the draws after a checkpoint replay with the
    seeds the interrupted run gave them, so no RNG state is saved.

    Args:
        model: The policy to train, in place. On a resume, the caller
            has already loaded the checkpoint's weights into it.
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
            RNG alone. The caller keeps this stream apart from any other
            stage's seeds: a draw sharing an eval's seed and sampling
            call replays that eval's samples.
        resume: Where a checkpointed run picks up, or None to start
            from the first draw.
        checkpointer: Told of progress after every draw and asked to
            save after every `checkpoint_every` updates and once when
            the loop stops; None saves nothing.
        checkpoint_every: The save cadence, in updates.

    Returns:
        "steps", one report per draw in order (the name predates dynamic
        sampling, and stored runs depend on it), and "updates", "draws"
        and "stopped": which limit ended the run, "updates",
        "max_draws" or "pool_exhausted", checked in that order. A
        resumed run returns the draws from before the resume as well.

    Raises:
        ValueError: if steps, max_draws or checkpoint_every is below
            one, if problems is empty, or if a resume comes without a
            seed or names a problem outside the pool.
    """
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")
    if not problems:
        raise ValueError("train needs at least one problem")
    max_draws = draw_cap(steps, max_draws)
    if max_draws < 1:
        raise ValueError(f"max_draws must be >= 1, got {max_draws}")
    if checkpoint_every < 1:
        raise ValueError(
            f"checkpoint_every must be >= 1, got {checkpoint_every}")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    pool = {p["unique_id"] for p in problems}
    history, retired = [], set()
    updates = draws = position = 0
    if resume is not None:
        if seed is None:
            raise ValueError("a resume needs the seed: without per-draw "
                             "seeds the replayed draws cannot match")
        outside = sorted({s["unique_id"] for s in resume.history} - pool)
        if outside:
            raise ValueError(f"the resumed history drew {outside}, which "
                             "are not in this pool")
        optimizer.load_state_dict(resume.optimizer_state)
        history = [dict(s) for s in resume.history]
        draws = len(history)
        updates = sum(s["stepped"] for s in history)
        retired = {s["unique_id"] for s in history if not s["stepped"]}
        position = resume.position
    resumed_at = draws

    def going() -> bool:
        return updates < steps and draws < max_draws and retired != pool

    while going():
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
        if checkpointer is not None:
            checkpointer.progress(draws, updates)
            # A save due on the stopping draw waits for the final one
            # below, so the same state is never written twice.
            if (report["stepped"] and updates % checkpoint_every == 0
                    and going()):
                checkpointer.save(optimizer, history, position,
                                  complete=False)
    if checkpointer is not None and draws > resumed_at:
        checkpointer.save(optimizer, history, position, complete=True)
    if updates == steps:
        stopped = "updates"
    elif draws == max_draws:
        stopped = "max_draws"
    else:
        stopped = "pool_exhausted"
    return {"steps": history, "updates": updates, "draws": draws,
            "stopped": stopped}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run --group dev pytest -q tests/test_trainer.py`
Expected: `39 passed`.

Run: `uv run --group dev pytest -q`
Expected: `219 passed`.

- [ ] **Step 5: Commit**

```bash
git add mini_reasoning/trainer.py tests/test_trainer.py
git commit -F - <<'EOF'
feat(trainer): resume from a saved state and call a checkpointer

train() takes back a draw history, a pool position and Adam's state,
and continues as if never stopped: the per-draw seeds make the replayed
draws identical, so no RNG state is saved. It calls an injected
checkpointer after every draw and every checkpoint_every updates, with
a final save when the loop stops, and never touches disk itself.

Co-Authored-By: Claude <model> <noreply@anthropic.com>
EOF
```

---

### Task 2: Checkpoint files

**Files:**
- Create: `mini_reasoning/checkpoint.py`
- Create: `tests/test_checkpoint.py`
- Modify: `pyproject.toml` (dependencies), `uv.lock`

**Interfaces:**
- Consumes: `Resume` from Task 1.
- Produces: constants `FORMAT = 1`, `LATEST`, `PROGRESS`, `RESUMES`, `PREFIX = "draw-"`, `WEIGHTS`, `OPTIMIZER`, `STATE`, and `IDENTITY = ("model", "seed", "steps", "max_draws", "group_size", "lr", "pool")`; `Saved(path, history, position, complete, dtype)` with `draws` and `updates` properties; `Writer(directory, model, identity)`, implementing the `Checkpointer` protocol; `read_latest(directory, identity) -> Saved | None`, raising `ValueError` for a missing named checkpoint, an unknown format, or an identity mismatch naming the field; `load_weights(saved, model) -> None`, raising `ValueError` on a dtype mismatch; `resume_from(saved, model) -> Resume`; `log_resume(directory, saved) -> str`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_checkpoint.py`:

```python
"""Contract tests for checkpoint files: what a save writes, what a load
refuses, and that a kill mid-save never leaves a run without a
checkpoint to resume from.

The policy is a real (tiny) module and the optimizer a real Adam that
has taken steps, so the state round-tripped is the state that matters.

Written with Claude (AI-written); see the README's Provenance section.
"""
import json

import pytest
import torch
from safetensors import safe_open
from torch import nn

from mini_reasoning import checkpoint
from mini_reasoning.checkpoint import (
    IDENTITY, LATEST, PROGRESS, RESUMES, Writer, load_weights, log_resume,
    read_latest, resume_from,
)

VOCAB = 32


class TinyPolicy(nn.Module):
    """A real module with an embedding and a head, like the policy's ends."""

    def __init__(self, vocab=VOCAB, dim=8, tied=False):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.head = nn.Linear(dim, vocab)
        if tied:
            self.head.weight = self.embed.weight


def policy(seed=0, **kwargs):
    torch.manual_seed(seed)
    return TinyPolicy(**kwargs)


def stepped_adam(model, steps=3):
    """Adam with moments and step counts to save, not an empty state."""
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(steps):
        optimizer.zero_grad()
        model.head(model.embed(torch.arange(4))).sum().backward()
        optimizer.step()
    return optimizer


IDENTITY_VALUES = {"model": "tiny", "seed": 92, "steps": 40,
                   "max_draws": 120, "group_size": 7, "lr": 1e-6,
                   "pool": ["p1", "p2"]}
HISTORY = [{"unique_id": "p1", "mean_reward": 0.5, "grad_norm": 3.0,
            "stepped": True},
           {"unique_id": "p2", "mean_reward": 0.0, "grad_norm": 0.0,
            "stepped": False}]


def saved_run(tmp_path, model=None, history=HISTORY, complete=False):
    model = model if model is not None else policy()
    optimizer = stepped_adam(model)
    Writer(tmp_path, model, IDENTITY_VALUES).save(
        optimizer, history, position=2, complete=complete)
    return model, optimizer


class TestRoundTrip:
    """What a save writes is what a resume gets back, exactly."""

    def test_weights_optimizer_history_and_position_come_back(self, tmp_path):
        model, optimizer = saved_run(tmp_path)
        saved = read_latest(tmp_path, IDENTITY_VALUES)
        fresh = policy(seed=1)
        resume = resume_from(saved, fresh)
        for want, got in zip(model.parameters(), fresh.parameters()):
            assert torch.equal(want, got)
        want_state = optimizer.state_dict()
        for index, tensors in want_state["state"].items():
            for key, tensor in tensors.items():
                assert torch.equal(resume.optimizer_state["state"][index][key],
                                   tensor)
        assert (resume.optimizer_state["param_groups"]
                == want_state["param_groups"])
        assert resume.history == HISTORY and resume.position == 2
        assert saved.draws == 2 and saved.updates == 1
        assert saved.complete is False

    def test_tied_weights_are_stored_once_and_load_still_tied(self, tmp_path):
        """Qwen3 ties its embedding to its output head. A copy that untied
        them would train the two apart after the resume."""
        model, _ = saved_run(tmp_path, model=policy(tied=True))
        saved = read_latest(tmp_path, IDENTITY_VALUES)
        with safe_open(saved.path / checkpoint.WEIGHTS, "pt") as stored:
            assert len(list(stored.keys())) == 2  # embed (= head) and bias
        fresh = policy(seed=1, tied=True)
        load_weights(saved, fresh)
        assert fresh.head.weight is fresh.embed.weight
        assert torch.equal(fresh.embed.weight, model.embed.weight)

    def test_progress_is_written_whole(self, tmp_path):
        Writer(tmp_path, policy(), IDENTITY_VALUES).progress(5, 3)
        assert json.loads((tmp_path / PROGRESS).read_text()) == {
            "draws": 5, "updates": 3}


class TestAtomicity:
    """A kill at any moment leaves LATEST naming a complete checkpoint."""

    def test_a_save_that_dies_before_latest_moves_leaves_the_old_one(
            self, tmp_path, monkeypatch):
        model = policy()
        optimizer = stepped_adam(model)
        writer = Writer(tmp_path, model, IDENTITY_VALUES)
        writer.save(optimizer, HISTORY[:1], position=1, complete=False)

        def killed(*args, **kwargs):
            raise KeyboardInterrupt("killed mid-save")

        monkeypatch.setattr(checkpoint.os, "replace", killed)
        with pytest.raises(KeyboardInterrupt):
            writer.save(optimizer, HISTORY, position=2, complete=False)
        monkeypatch.undo()

        saved = read_latest(tmp_path, IDENTITY_VALUES)
        assert saved.path.name == "draw-0001"
        assert (tmp_path / "draw-0002").exists()  # the partial write

        writer.save(optimizer, HISTORY + HISTORY[:1], position=3,
                    complete=False)
        assert sorted(p.name for p in tmp_path.glob("draw-*")) == [
            "draw-0003"]

    def test_a_save_that_fails_while_writing_leaves_the_old_one(
            self, tmp_path, monkeypatch):
        """A full disk fails a save partway through its files, before
        LATEST is touched."""
        model = policy()
        optimizer = stepped_adam(model)
        writer = Writer(tmp_path, model, IDENTITY_VALUES)
        writer.save(optimizer, HISTORY[:1], position=1, complete=False)

        def full(*args, **kwargs):
            raise OSError(28, "No space left on device")

        monkeypatch.setattr(checkpoint.torch, "save", full)
        with pytest.raises(OSError):
            writer.save(optimizer, HISTORY, position=2, complete=False)
        monkeypatch.undo()
        assert read_latest(tmp_path, IDENTITY_VALUES).path.name == "draw-0001"

    def test_saving_over_the_current_checkpoint_is_refused(self, tmp_path):
        model = policy()
        optimizer = stepped_adam(model)
        writer = Writer(tmp_path, model, IDENTITY_VALUES)
        writer.save(optimizer, HISTORY, position=2, complete=False)
        with pytest.raises(ValueError, match="current"):
            writer.save(optimizer, HISTORY, position=2, complete=True)


class TestRefusals:
    """A checkpoint that is not this run's, or not whole, is refused."""

    def test_no_checkpoint_reads_as_none(self, tmp_path):
        assert read_latest(tmp_path, IDENTITY_VALUES) is None
        assert read_latest(tmp_path / "absent", IDENTITY_VALUES) is None

    @pytest.mark.parametrize("field", IDENTITY)
    def test_each_identity_field_is_checked(self, tmp_path, field):
        saved_run(tmp_path)
        other = dict(IDENTITY_VALUES,
                     **{field: ["p9"] if field == "pool" else "other"})
        with pytest.raises(ValueError, match=field):
            read_latest(tmp_path, other)

    def test_another_dtype_is_refused_at_load(self, tmp_path):
        """The copy would cast float32 weights silently into the wrong
        model, so the dtype is checked against the loaded model."""
        saved_run(tmp_path)
        saved = read_latest(tmp_path, IDENTITY_VALUES)
        with pytest.raises(ValueError, match="float64"):
            load_weights(saved, policy().to(torch.float64))

    def test_latest_naming_a_missing_checkpoint_raises(self, tmp_path):
        """Not a fresh start: someone removed part of a checkpoint."""
        saved_run(tmp_path)
        (tmp_path / LATEST).write_text("draw-0099\n")
        with pytest.raises(ValueError, match="draw-0099"):
            read_latest(tmp_path, IDENTITY_VALUES)

    def test_an_unknown_format_raises(self, tmp_path):
        saved_run(tmp_path)
        state_path = tmp_path / "draw-0002" / checkpoint.STATE
        state = json.loads(state_path.read_text())
        state["format"] = 2
        state_path.write_text(json.dumps(state))
        with pytest.raises(ValueError, match="format"):
            read_latest(tmp_path, IDENTITY_VALUES)


class TestLostWork:
    """What an interruption cost, for the lab's eviction-cost table."""

    def test_lost_work_is_progress_minus_the_checkpoint(self, tmp_path):
        saved_run(tmp_path)
        (tmp_path / PROGRESS).write_text(
            json.dumps({"draws": 5, "updates": 2}))
        saved = read_latest(tmp_path, IDENTITY_VALUES)
        line = log_resume(tmp_path, saved)
        assert line == "resuming at draw 2, update 1 (lost 3 draws, 1 update)"
        entries = [json.loads(row) for row in
                   (tmp_path / RESUMES).read_text().splitlines()]
        assert len(entries) == 1
        assert entries[0]["lost"] == {"draws": 3, "updates": 1}
        assert entries[0]["resumed_at"] == {"draws": 2, "updates": 1}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run --group dev pytest -q tests/test_checkpoint.py`
Expected: a collection error, `ImportError: cannot import name 'checkpoint' from 'mini_reasoning'`.

- [ ] **Step 3: Implement**

Declare the dependency. In `pyproject.toml`, add `"safetensors>=0.8",` to `dependencies`, after `"matplotlib>=3.9",`. Then relock with a current uv:

```bash
uvx --from 'uv==0.12.23' uv lock
git diff --stat uv.lock
```

Expected: `uv.lock` changes by its `revision` line and two `safetensors` lines in the project's own entry, nothing else. An older uv also rewrites unrelated nvidia markers, which is noise in the diff and the reason for pinning the version here. Resolution is unchanged: transformers already brought in safetensors 0.8.0.

Create `mini_reasoning/checkpoint.py`:

```python
"""Saving and restoring GRPO training, so an interrupted run resumes exactly.

The pipeline's records are reproducible from their seeds, and a run
evicted mid-training has to come back to the same records, so a resume
is exact or it is a failure. Exactness needs less state than it might
seem: every draw reseeds from the training seed plus its index, so no
RNG state is saved. What a killed process loses is the state no seed
rebuilds, and that is what a checkpoint holds:

- model.safetensors: the policy's weights, float32, written with
  safetensors' save_model, which stores Qwen3's tied embedding and
  output head once.
- optimizer.pt: Adam's state_dict, its moments and step counts.
  Restarting Adam would change the first update after a resume.
- state.json: the draw history and the pool position, the run's
  identity, and whether the loop had stopped.

A save writes a new draw-NNNN directory, then repoints LATEST at it
with os.replace, then deletes the older ones, so a kill at any moment
leaves LATEST naming a complete checkpoint. Directories are named by
draw count because a final save can follow a periodic one at the same
update count, when flat draws come after the last update.

progress.json and resumes.jsonl beside them measure the work an
eviction cost. They stay out of the run's records on purpose: a resumed
run's training.json must equal a clean run's byte for byte.

Written with Claude (AI-written); see the README's Provenance section.
"""
import json
import os
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch
from safetensors.torch import load_model, save_model

from mini_reasoning.trainer import Resume

FORMAT = 1
LATEST = "LATEST"
PROGRESS = "progress.json"
RESUMES = "resumes.jsonl"
PREFIX = "draw-"
WEIGHTS = "model.safetensors"
OPTIMIZER = "optimizer.pt"
STATE = "state.json"
# Compared before the model loads. dtype is compared after, against the
# loaded model itself, since only the model can say what it holds.
IDENTITY = ("model", "seed", "steps", "max_draws", "group_size", "lr",
            "pool")


@dataclass(frozen=True)
class Saved:
    """A checkpoint's state.json, read and checked against this run.

    Attributes:
        path: The draw-NNNN directory.
        history: The per-draw reports up to the save.
        position: The pool index the loop examines next.
        complete: True when the save was made as the loop stopped.
        dtype: The dtype of the saved weights.
    """
    path: Path
    history: list[dict]
    position: int
    complete: bool
    dtype: str

    @property
    def draws(self) -> int:
        return len(self.history)

    @property
    def updates(self) -> int:
        return sum(s["stepped"] for s in self.history)


def _dtype(model) -> str:
    return str(next(iter(model.parameters())).dtype)


def _sync(path: Path) -> None:
    """Flushes a file or directory to disk, so a node that dies after a
    save does not take the save with it."""
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _replace_text(path: Path, text: str) -> None:
    """Writes path whole or not at all: a reader sees the old text or the
    new, never a torn one."""
    partial = path.with_name(path.name + ".partial")
    partial.write_text(text)
    _sync(partial)
    os.replace(partial, path)
    _sync(path.parent)


class Writer:
    """The trainer's Checkpointer: progress after every draw, and full
    state when the trainer asks for it.

    Args:
        directory: The run's --checkpoint-dir, created if absent.
        model: The policy whose weights each save writes.
        identity: This run's identity, the IDENTITY fields. The saved
            weights' dtype is added at each save.
    """

    def __init__(self, directory, model, identity: Mapping):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.model = model
        self.identity = dict(identity)

    def progress(self, draws: int, updates: int) -> None:
        """Records how far training has got, for the lost-work count."""
        _replace_text(self.directory / PROGRESS,
                      json.dumps({"draws": draws, "updates": updates}) + "\n")

    def save(self, optimizer, history: list[dict], position: int, *,
             complete: bool) -> None:
        """Writes a checkpoint, repoints LATEST at it, then prunes.

        Raises:
            ValueError: if a save at this draw count is already current.
                The trainer never asks for that, and overwriting the one
                complete checkpoint would leave none.
        """
        name = f"{PREFIX}{len(history):04d}"
        latest = self.directory / LATEST
        if latest.exists() and latest.read_text().strip() == name:
            raise ValueError(f"{name} is the current checkpoint already")
        target = self.directory / name
        if target.exists():
            shutil.rmtree(target)  # a partial write from a killed save
        target.mkdir()
        save_model(self.model, str(target / WEIGHTS))
        torch.save(optimizer.state_dict(), target / OPTIMIZER)
        state = {"format": FORMAT,
                 "identity": {**self.identity, "dtype": _dtype(self.model)},
                 "history": history, "position": position,
                 "complete": complete}
        (target / STATE).write_text(json.dumps(state, indent=2) + "\n")
        for item in (target / WEIGHTS, target / OPTIMIZER, target / STATE,
                     target):
            _sync(item)
        _replace_text(latest, name + "\n")
        for old in self.directory.glob(PREFIX + "*"):
            if old.name != name:
                shutil.rmtree(old)


def read_latest(directory, identity: Mapping) -> Saved | None:
    """The checkpoint LATEST names, checked against this run.

    Nothing is loaded but state.json, so a run that cannot resume finds
    out before its model loads.

    Args:
        directory: The run's --checkpoint-dir.
        identity: This run's IDENTITY fields.

    Returns:
        The checkpoint, or None when the directory holds none.

    Raises:
        ValueError: if LATEST names a directory that is missing or has
            no state.json, if the format is not FORMAT, or if an
            identity field differs from this run's. The message names
            the field.
    """
    directory = Path(directory)
    latest = directory / LATEST
    if not latest.exists():
        return None
    name = latest.read_text().strip()
    path = directory / name
    if not (path / STATE).exists():
        raise ValueError(
            f"{latest} names {name}, which is missing or incomplete. Part "
            "of the checkpoint was removed by hand, so it cannot be "
            "trusted: start again with a new --checkpoint-dir.")
    state = json.loads((path / STATE).read_text())
    if state.get("format") != FORMAT:
        raise ValueError(f"{path} is checkpoint format "
                         f"{state.get('format')!r}, not {FORMAT}")
    saved_identity = state["identity"]
    for field in IDENTITY:
        if saved_identity.get(field) != identity.get(field):
            raise ValueError(
                f"the checkpoint in {directory} was made with {field}="
                f"{saved_identity.get(field)!r} and this run has {field}="
                f"{identity.get(field)!r}, so it belongs to another run. "
                "Use a new --checkpoint-dir for this one.")
    return Saved(path=path, history=state["history"],
                 position=state["position"], complete=state["complete"],
                 dtype=saved_identity["dtype"])


def load_weights(saved: Saved, model) -> None:
    """Copies the saved weights into a model already loaded by name.

    Loading by name and then copying keeps the model's own config, and
    so the provenance stamp evaluate writes, the same as a clean run's.

    Raises:
        ValueError: if the model's dtype is not the saved weights'. The
            copy would cast silently, and a float32 run resumed in
            bfloat16 is another run.
    """
    if _dtype(model) != saved.dtype:
        raise ValueError(f"the checkpoint holds {saved.dtype} weights and "
                         f"the model is {_dtype(model)}")
    load_model(model, str(saved.path / WEIGHTS))


def resume_from(saved: Saved, model) -> Resume:
    """Loads the saved weights into model and returns where training
    picks up, optimizer state included."""
    load_weights(saved, model)
    optimizer_state = torch.load(saved.path / OPTIMIZER, weights_only=True)
    return Resume(history=saved.history, position=saved.position,
                  optimizer_state=optimizer_state)


def _count(n: int, noun: str) -> str:
    return f"{n} {noun}{'' if n == 1 else 's'}"


def log_resume(directory, saved: Saved) -> str:
    """Appends one line to resumes.jsonl and returns it as printed.

    Lost work is what progress.json says the interrupted process reached,
    minus what the checkpoint holds.
    """
    directory = Path(directory)
    at = {"draws": saved.draws, "updates": saved.updates}
    progress = directory / PROGRESS
    reached = json.loads(progress.read_text()) if progress.exists() else at
    lost = {key: reached[key] - at[key] for key in at}
    entry = {"time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "resumed_at": at, "reached": reached, "lost": lost}
    with open(directory / RESUMES, "a") as log:
        log.write(json.dumps(entry) + "\n")
    return (f"resuming at draw {at['draws']}, update {at['updates']} "
            f"(lost {_count(lost['draws'], 'draw')}, "
            f"{_count(lost['updates'], 'update')})")
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run --group dev pytest -q tests/test_checkpoint.py`
Expected: `18 passed`.

Run: `uv run --group dev pytest -q`
Expected: `237 passed`.

- [ ] **Step 5: Commit**

```bash
git add mini_reasoning/checkpoint.py tests/test_checkpoint.py pyproject.toml uv.lock
git commit -F - <<'EOF'
feat(checkpoint): save and restore training state atomically

A save writes draw-NNNN/ with the weights in safetensors (tied tensors
stored once), Adam's state and a state.json, then repoints LATEST with
os.replace and prunes older saves, so a kill at any moment leaves a
complete checkpoint. A load checks the run's identity before the model
loads and the weights' dtype when they do. progress.json and
resumes.jsonl record the work an interruption cost.

Co-Authored-By: Claude <model> <noreply@anthropic.com>
EOF
```

---

### Task 3: The entry point checkpoints, resumes and runs stages

**Files:**
- Modify: `run_pipeline.py` (module docstring; imports and constants; before `build_parser`; inside `build_parser`; from `def run(` to the end)
- Test: `tests/test_run_pipeline.py`

**Interfaces:**
- Consumes: `trainer.draw_cap`, `trainer.train(..., resume=, checkpointer=, checkpoint_every=)` from Task 1; `checkpoint.read_latest`, `Writer`, `load_weights`, `resume_from`, `log_resume` from Task 2.
- Produces: `STAGES = ("all", "baseline", "train", "validate")`; `CHECKPOINT_EVERY = 5`; `PipelineParser`; `check_flags(parser, args)`; flags `--stage`, `--checkpoint-dir`, `--checkpoint-every` (resolved to 5 when unset); `run_identity(args, pool, training_seed) -> dict`; `check_validate_inputs(out_dir, saved)`; `baseline_stage(...)`; `training_stage(...)`; `run(args, *, load_model, sampler, rollout) -> dict` returning only the stages that ran.

- [ ] **Step 1: Write the failing tests**

Append to the end of `tests/test_run_pipeline.py`, after two blank lines:

```python
RECORDS = ("baseline", "training", "validation", "report")


class Killed(Exception):
    """Stands in for SIGKILL: the process ends wherever it was."""


def killed_after(draws):
    """canned_rollout that dies on its call after `draws` draws, once."""
    calls = []

    def rollout(model, tokenizer, problem, n):
        if len(calls) == draws:
            calls.append("killed")
            raise Killed
        calls.append(problem)
        return canned_rollout(model, tokenizer, problem, n)
    return rollout


def run_dir(tmp_path, name, **overrides):
    """args_for, in its own directory, with a checkpoint directory."""
    root = tmp_path / name
    root.mkdir()
    return args_for(root, **{"--checkpoint-dir": str(root / "ckpt"),
                             **overrides})


def records(args):
    return {name: (Path(args.out_dir) / f"{name}.json").read_bytes()
            for name in RECORDS}


class TestCheckpointFlags:
    """Checkpointing is opt-in, and a combination that cannot run fails
    at parse time, before the model loads."""

    def test_defaults_leave_checkpointing_off(self):
        args = build_parser().parse_args([])
        assert args.checkpoint_dir is None and args.stage == "all"
        assert args.checkpoint_every == 5

    def test_skip_training_is_the_baseline_stage(self):
        assert build_parser().parse_args(
            ["--skip-training"]).stage == "baseline"

    @pytest.mark.parametrize("argv", [
        ["--checkpoint-every", "3"],
        ["--stage", "train"],
        ["--stage", "validate"],
        ["--skip-training", "--stage", "train", "--checkpoint-dir", "c"],
        ["--checkpoint-dir", "c", "--checkpoint-every", "0"],
    ])
    def test_a_combination_that_cannot_run_is_refused(self, argv):
        with pytest.raises(SystemExit):
            build_parser().parse_args(argv)


class TestResumeAfterAKill:
    """The same command, started again, ends where an unbroken run does."""

    def test_a_killed_run_resumes_to_the_same_records(self, tmp_path):
        steady = run_dir(tmp_path, "steady", **{"--steps": 4,
                                                "--checkpoint-every": 2})
        run(steady, load_model=loader, sampler=canned, rollout=canned_rollout)

        broken = run_dir(tmp_path, "broken", **{"--steps": 4,
                                                "--checkpoint-every": 2})
        # p2 is flat under canned_rollout, so the save at update 2 comes
        # after draw 3. Dying on the fifth call loses the finished draw 4,
        # which the resume has to replay.
        rollout = killed_after(4)
        with pytest.raises(Killed):
            run(broken, load_model=loader, sampler=canned, rollout=rollout)
        run(broken, load_model=loader, sampler=canned, rollout=rollout)
        assert records(broken) == records(steady)

    def test_the_resume_says_what_the_kill_cost(self, tmp_path, capsys):
        broken = run_dir(tmp_path, "broken", **{"--steps": 4,
                                                "--checkpoint-every": 2})
        rollout = killed_after(4)
        with pytest.raises(Killed):
            run(broken, load_model=loader, sampler=canned, rollout=rollout)
        run(broken, load_model=loader, sampler=canned, rollout=rollout)
        assert ("resuming at draw 3, update 2 (lost 1 draw, 1 update)"
                in capsys.readouterr().out)

    def test_a_resumed_full_run_reuses_the_stored_baseline(self, tmp_path):
        """A kill before the first save must not cost the baseline's
        hour again: the eval is deterministic, so the stored one is it."""
        args = run_dir(tmp_path, "run", **{"--steps": 2})
        with pytest.raises(Killed):
            run(args, load_model=loader, sampler=canned,
                rollout=killed_after(0))
        calls = []

        def counting(model, tokenizer, problem, n):
            calls.append(problem)
            return canned(model, tokenizer, problem, n)

        run(args, load_model=loader, sampler=counting,
            rollout=canned_rollout)
        assert len(calls) == 2  # the validation eval alone

    def test_a_resume_at_another_cadence_writes_the_same_records(
            self, tmp_path):
        """A save changes nothing it saves, so the cadence is not part of
        a run's identity and may change between attempts."""
        steady = run_dir(tmp_path, "steady", **{"--steps": 4})
        run(steady, load_model=loader, sampler=canned, rollout=canned_rollout)
        broken = run_dir(tmp_path, "broken", **{"--steps": 4,
                                                "--checkpoint-every": 2})
        rollout = killed_after(4)
        with pytest.raises(Killed):
            run(broken, load_model=loader, sampler=canned, rollout=rollout)
        broken.checkpoint_every = 1
        run(broken, load_model=loader, sampler=canned, rollout=rollout)
        assert records(broken) == records(steady)

    def test_a_stored_baseline_from_another_dtype_is_refused(self, tmp_path):
        """Automatic reuse applies --reuse-baseline's checks, so a stale
        before half is refused, not silently reused."""
        args = run_dir(tmp_path, "run", **{"--steps": 2})
        with pytest.raises(Killed):
            run(args, load_model=loader, sampler=canned,
                rollout=killed_after(0))
        stored = Path(args.out_dir) / "baseline.json"
        written = json.loads(stored.read_text())
        written["provenance"]["dtype"] = "torch.bfloat16"
        stored.write_text(json.dumps(written))
        with pytest.raises(ValueError, match="dtype"):
            run(args, load_model=loader, sampler=canned,
                rollout=canned_rollout)

    def test_a_finished_full_run_started_again_draws_nothing(self, tmp_path):
        """A requeue after the end, say of a pod that finished as it was
        evicted, must not train again or change a record."""
        args = run_dir(tmp_path, "run", **{"--steps": 3})
        run(args, load_model=loader, sampler=canned, rollout=canned_rollout)
        before = records(args)

        def forbidden(*args, **kwargs):
            raise AssertionError("a finished run drew again")

        run(args, load_model=loader, sampler=canned, rollout=forbidden)
        assert records(args) == before

    def test_a_checkpoint_from_another_run_is_refused_before_loading(
            self, tmp_path):
        args = run_dir(tmp_path, "run", **{"--steps": 4,
                                           "--checkpoint-every": 2})
        with pytest.raises(Killed):
            run(args, load_model=loader, sampler=canned,
                rollout=killed_after(3))
        args.lr = 1e-5

        def never(name):
            raise AssertionError("the model loaded before the checkpoint "
                                 "was checked")

        with pytest.raises(ValueError, match="lr"):
            run(args, load_model=never, sampler=canned,
                rollout=canned_rollout)


def stage_args(tmp_path, stage, **overrides):
    """One stage of a three-stage run, all sharing one directory."""
    root = tmp_path / "staged"
    root.mkdir(exist_ok=True)
    return args_for(root, **{"--checkpoint-dir": str(root / "ckpt"),
                             "--stage": stage, **overrides})


def named_loader(name):
    """loader, with the config a pretrained model carries its name in."""
    model, tokenizer = loader(name)
    model.config = type("Config", (), {"name_or_path": name})()
    return model, tokenizer


class TestStages:
    """Three processes, handing over through the out and checkpoint
    directories, write what one process writes."""

    def staged(self, tmp_path, load_model=loader):
        for stage in ("baseline", "train", "validate"):
            run(stage_args(tmp_path, stage, **{"--steps": 3}),
                load_model=load_model, sampler=canned,
                rollout=canned_rollout)
        return stage_args(tmp_path, "validate", **{"--steps": 3})

    def test_three_stages_write_what_one_run_writes(self, tmp_path):
        whole = tmp_path / "whole"
        whole.mkdir()
        one = args_for(whole, **{"--steps": 3})
        run(one, load_model=loader, sampler=canned, rollout=canned_rollout)
        assert records(self.staged(tmp_path)) == records(one)

    def test_the_handoff_keeps_the_base_models_name_in_provenance(
            self, tmp_path):
        """Loading the trained weights from a path would stamp the path,
        and the records would differ from a clean run's."""
        args = self.staged(tmp_path, load_model=named_loader)
        written = json.loads(
            (Path(args.out_dir) / "validation.json").read_text())
        assert written["provenance"]["model"] == args.model

    def test_the_train_stage_on_a_finished_checkpoint_draws_nothing(
            self, tmp_path):
        self.staged(tmp_path)

        def forbidden(*args, **kwargs):
            raise AssertionError("a finished run drew again")

        results = run(stage_args(tmp_path, "train", **{"--steps": 3}),
                      load_model=loader, sampler=canned, rollout=forbidden)
        assert results["training"]["updates"] == 3


def never_loads(name):
    raise AssertionError("the model loaded before validate's inputs were "
                         "checked")


class TestValidateRefusals:
    """--stage validate checks its inputs before the model loads."""

    def test_no_checkpoint(self, tmp_path):
        with pytest.raises(ValueError, match="complete checkpoint"):
            run(stage_args(tmp_path, "validate"), load_model=never_loads,
                sampler=canned)

    def test_an_unfinished_checkpoint(self, tmp_path):
        args = stage_args(tmp_path, "train", **{"--steps": 4,
                                                "--checkpoint-every": 2})
        with pytest.raises(Killed):
            run(args, load_model=loader, sampler=canned,
                rollout=killed_after(3))
        with pytest.raises(ValueError, match="before training ended"):
            run(stage_args(tmp_path, "validate", **{"--steps": 4}),
                load_model=never_loads, sampler=canned)

    def test_a_missing_training_record(self, tmp_path):
        TestStages().staged(tmp_path)
        (tmp_path / "staged" / "results" / "training.json").unlink()
        with pytest.raises(FileNotFoundError, match="training.json"):
            run(stage_args(tmp_path, "validate", **{"--steps": 3}),
                load_model=never_loads, sampler=canned)

    def test_another_split_than_the_train_stage_is_refused(self, tmp_path):
        """Each stage is given the run's flags again, and one given
        differently would validate another run's weights."""
        TestStages().staged(tmp_path)
        with pytest.raises(ValueError, match="pool"):
            run(stage_args(tmp_path, "validate",
                           **{"--steps": 3, "--held-out": 1}),
                load_model=never_loads, sampler=canned)

    def test_a_training_record_from_another_run(self, tmp_path):
        TestStages().staged(tmp_path)
        path = tmp_path / "staged" / "results" / "training.json"
        written = json.loads(path.read_text())
        written["steps"][0]["grad_norm"] += 1.0
        path.write_text(json.dumps(written))
        with pytest.raises(ValueError, match="does not match"):
            run(stage_args(tmp_path, "validate", **{"--steps": 3}),
                load_model=never_loads, sampler=canned)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run --group dev pytest -q tests/test_run_pipeline.py`
Expected: `17 failed, 47 passed`. The five parametrized cases of `test_a_combination_that_cannot_run_is_refused` already pass, because argparse rejects the flags as unknown. They pin the rules once the flags exist.

- [ ] **Step 3: Implement**

In `run_pipeline.py`, append a paragraph to the module docstring, after the paragraph that ends "since a bad one found after the baseline eval costs an hour.":

```
Checkpointing is the entry point's to switch on. With --checkpoint-dir,
training saves its full state every --checkpoint-every updates, and the
same command started again resumes from the latest save, because a
requeued pod cannot change its own arguments. --stage runs one stage of
the three alone, so baseline, training and validation can be separate
processes that hand over through --out-dir and --checkpoint-dir.
```

Replace the `mini_reasoning` import and add two constants after `EVAL_FALLBACK`:

```python
from mini_reasoning import (
    checkpoint, consistency, evaluate as evaluation, report, scorers,
    trainer,
)

EVAL_FALLBACK = "number"
STAGES = ("all", "baseline", "train", "validate")
CHECKPOINT_EVERY = 5
```

Insert before `def build_parser`:

```python
class PipelineParser(argparse.ArgumentParser):
    """An ArgumentParser that also applies the rules between flags.

    argparse checks one flag at a time. These rules concern pairs, and
    are applied here so a combination that cannot run fails at parse
    time, before the model loads, like every other bad flag.
    """

    def parse_args(self, args=None, namespace=None):
        parsed = super().parse_args(args, namespace)
        check_flags(self, parsed)
        return parsed


def check_flags(parser: argparse.ArgumentParser,
                args: argparse.Namespace) -> None:
    """Applies the rules between flags and resolves their defaults.

    --skip-training is --stage baseline under the name the README
    documents. --checkpoint-every means nothing without
    --checkpoint-dir, and the train and validate stages hand the trained
    weights over through one, so each needs it.

    Raises:
        SystemExit: through parser.error, for a combination that cannot
            run.
    """
    if args.skip_training:
        if args.stage not in ("all", "baseline"):
            parser.error("--skip-training is --stage baseline, so it "
                         f"cannot be combined with --stage {args.stage}")
        args.stage = "baseline"
    if args.checkpoint_every is not None and args.checkpoint_dir is None:
        parser.error("--checkpoint-every needs --checkpoint-dir")
    if args.stage in ("train", "validate") and args.checkpoint_dir is None:
        parser.error(f"--stage {args.stage} needs --checkpoint-dir: the "
                     "train stage leaves its final weights there for the "
                     "validate stage")
    if args.checkpoint_every is None:
        args.checkpoint_every = CHECKPOINT_EVERY
```

In `build_parser`, change `parser = argparse.ArgumentParser(` to `parser = PipelineParser(`, and replace the `--skip-training` argument (its two lines) with:

```python
    parser.add_argument("--skip-training", action="store_true",
                        help="run the inference/eval stages only; the "
                             "same as --stage baseline")
    parser.add_argument("--stage", choices=STAGES, default="all",
                        help="run one stage alone, sharing --out-dir and "
                             "--checkpoint-dir with the others; all runs "
                             "them in turn")
    parser.add_argument("--checkpoint-dir", default=None,
                        help="save training state here and resume from it "
                             "when the same run starts again; off by "
                             "default. A save of the default model is "
                             "about 7.2 GB, and two exist during a save")
    parser.add_argument("--checkpoint-every", type=positive_int,
                        default=None,
                        help="the save cadence, in optimizer updates; "
                             f"default {CHECKPOINT_EVERY}. Needs "
                             "--checkpoint-dir")
```

Replace everything from `def run(` to the end of the file with:

```python
def run_identity(args, pool: list[dict], training_seed: int) -> dict:
    """What a checkpoint must share with this run for a resume to be exact.

    The model's name, the training seed, and every argument that decides
    the draws. A checkpoint that differs in any of them was made by
    another run. The weights' dtype is checked when they load, against
    the loaded model.
    """
    return {"model": args.model, "seed": training_seed,
            "steps": args.steps,
            "max_draws": trainer.draw_cap(args.steps, args.max_draws),
            "group_size": args.n_samples, "lr": args.lr,
            "pool": [p["unique_id"] for p in pool]}


def check_validate_inputs(out_dir, saved) -> None:
    """What --stage validate needs, checked before the model loads.

    Args:
        out_dir: Where the baseline and train stages wrote.
        saved: The checkpoint read_latest found, or None.

    Raises:
        ValueError: if the checkpoint is missing or incomplete, or if
            training.json's draws differ from its history, which would
            pair one run's training with another's weights.
        FileNotFoundError: if baseline.json or training.json is missing.
    """
    if saved is None or not saved.complete:
        found = ("none" if saved is None else
                 f"one saved at draw {saved.draws}, before training ended")
        raise ValueError("--stage validate needs a complete checkpoint and "
                         f"found {found}. Run --stage train to the end "
                         "first.")
    out_dir = Path(out_dir)
    for name, stage in (("baseline", "baseline"), ("training", "train")):
        if not (out_dir / f"{name}.json").exists():
            raise FileNotFoundError(
                f"--stage validate needs {name}.json in {out_dir}, which "
                f"--stage {stage} writes")
    steps = json.loads((out_dir / "training.json").read_text())["steps"]
    if steps != saved.history:
        raise ValueError(
            f"training.json in {out_dir} does not match the checkpoint's "
            "history, so the report would pair one run's training with "
            "another's weights")


def baseline_stage(args, model, tokenizer, problems, scorer,
                   eval_args: dict) -> dict:
    """The baseline eval, or the stored one where it may stand in.

    It is reused when --reuse-baseline asks, when --stage validate needs
    the before half for its report, and when a checkpointed full run
    finds one stored. The eval is deterministic in its seed, so a
    requeued run that reuses it loses only the hour it would cost again.
    """
    stored = Path(args.out_dir) / "baseline.json"
    reuse = (args.reuse_baseline or args.stage == "validate"
             or (args.stage == "all" and args.checkpoint_dir is not None
                 and stored.exists()))
    if reuse:
        return load_baseline(args.out_dir, problems, evaluation.provenance(
            model, seed=args.seed, n_samples=args.n_samples,
            fallback=EVAL_FALLBACK))
    baseline = evaluation.evaluate(model, tokenizer, problems, scorer,
                                   **eval_args)
    write_results(args.out_dir, "baseline", baseline)
    return baseline


def training_stage(args, model, tokenizer, pool, held_out, *,
                   training_seed: int, identity: dict, saved,
                   rollout) -> dict:
    """Trains, resuming from the checkpoint when there is one.

    A complete checkpoint resumes into a loop with nothing left to do,
    so its history comes back without a draw and training.json is
    written again: a kill between the final save and that write loses
    nothing.
    """
    resume = writer = None
    if args.checkpoint_dir is not None:
        if saved is not None:
            print(checkpoint.log_resume(args.checkpoint_dir, saved))
            resume = checkpoint.resume_from(saved, model)
        writer = checkpoint.Writer(args.checkpoint_dir, model, identity)
    history = trainer.train(model, tokenizer, pool, steps=args.steps,
                            rollout=rollout, group_size=args.n_samples,
                            lr=args.lr, max_draws=args.max_draws,
                            seed=training_seed, resume=resume,
                            checkpointer=writer,
                            checkpoint_every=args.checkpoint_every)
    history["held_out"] = [p["unique_id"] for p in held_out]
    history["seed"] = training_seed
    write_results(args.out_dir, "training", history)
    return history


def run(args, *, load_model=_load_model,
        sampler=evaluation.sample_solutions,
        rollout=trainer.sample_rollouts) -> dict:
    """Runs the stages the flags ask for and writes each one out.

    Everything that can refuse the run is checked before the model
    loads: the split, a checkpoint's identity, and what --stage validate
    needs.

    Args:
        args: Parsed arguments from build_parser.
        load_model: Returns (model, tokenizer) for a checkpoint name.
        sampler: Passed through to evaluate.
        rollout: Passed through to the trainer.

    Returns:
        What ran: "baseline" and "vote_accuracy" for every stage but
        train, "training" for all, train and validate, and "validation"
        and "report" for all and validate.

    Raises:
        ValueError: from the split, from a checkpoint made by another
            run, or from check_validate_inputs.
        FileNotFoundError: from check_validate_inputs.
    """
    problems = load_problems(args.subset)
    # The split only concerns training, so an eval-only run skips it; a
    # default of ten would otherwise refuse any smaller smoke subset.
    pool, held_out = split_held_out(
        problems, 0 if args.stage == "baseline" else args.held_out)
    # The evals seed problem i with seed + i. Training's stream starts past
    # the last of those: sharing one would replay the baseline's own
    # samples, and first-pass retirement would be decided on that draw.
    training_seed = args.seed + len(problems)
    identity = run_identity(args, pool, training_seed)
    saved = (checkpoint.read_latest(args.checkpoint_dir, identity)
             if args.checkpoint_dir is not None else None)
    if args.stage == "validate":
        check_validate_inputs(args.out_dir, saved)

    model, tokenizer = load_model(args.model)
    scorer = build_scorer()
    eval_args = dict(n_samples=args.n_samples, fallback=EVAL_FALLBACK,
                     sampler=sampler, seed=args.seed)
    results = {}

    if args.stage != "train":
        baseline = baseline_stage(args, model, tokenizer, problems, scorer,
                                  eval_args)
        results["baseline"] = baseline
        results["vote_accuracy"] = vote_accuracy(baseline["records"])
    if args.stage == "baseline":
        return results

    if args.stage == "validate":
        history = json.loads(
            (Path(args.out_dir) / "training.json").read_text())
        checkpoint.load_weights(saved, model)
        # A clean run validates the policy train() left in train mode.
        # Dropout is 0.0, so the mode should change no number, but the
        # handoff matches it rather than rely on that.
        model.train()
    else:
        history = training_stage(args, model, tokenizer, pool, held_out,
                                 training_seed=training_seed,
                                 identity=identity, saved=saved,
                                 rollout=rollout)
    results["training"] = history
    if args.stage == "train":
        return results

    validation = evaluation.evaluate(model, tokenizer, problems, scorer,
                                     **eval_args)
    write_results(args.out_dir, "validation", validation)
    results["validation"] = validation

    breakdown = report.report(results["baseline"], validation, history)
    write_results(args.out_dir, "report", breakdown)
    results["report"] = breakdown
    return results


def main():
    args = build_parser().parse_args()
    results = run(args)
    if "baseline" in results:
        print("baseline accuracy      "
              f"{results['baseline']['accuracy']:.3f}")
        print("level-reweighted       "
              f"{results['baseline']['level_reweighted_accuracy']:.3f}")
        print(f"self-consistency vote  {results['vote_accuracy']:.3f}")
    if "training" in results:
        training = results["training"]
        print(f"training updates       {training['updates']} over "
              f"{training['draws']} draws (stopped: {training['stopped']})")
    if "validation" in results:
        print("validation accuracy    "
              f"{results['validation']['accuracy']:.3f}")
        print("level-reweighted       "
              f"{results['validation']['level_reweighted_accuracy']:.3f}")
        print()
        print(report.format_table(results["report"]))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run --group dev pytest -q tests/test_run_pipeline.py`
Expected: `64 passed`.

Run: `uv run --group dev pytest -q`
Expected: `259 passed`.

- [ ] **Step 5: Commit**

```bash
git add run_pipeline.py tests/test_run_pipeline.py
git commit -F - <<'EOF'
feat(pipeline): checkpoint, resume and run stages on their own

--checkpoint-dir turns checkpointing on and makes the same command,
started again, resume from the latest save, because a requeued pod
cannot change its arguments. --stage runs baseline, training or
validation alone, handing over through the out and checkpoint
directories, and the validation stage loads the base model by name so
its provenance matches a single process. Everything that can refuse a
run is checked before the model loads.

Co-Authored-By: Claude <model> <noreply@anthropic.com>
EOF
```

---

### Task 4: Document checkpoints and stages

**Files:**
- Modify: `README.md` (a new section before `## Provenance`; two Provenance bullets)
- Modify: `docs/design-notes.md` (the provenance note at the top; a new section before `## Forensics: obligation or convention`)
- Modify: `.gitignore`

**Interfaces:**
- Consumes: the flags and behaviour of Tasks 1 to 3.

- [ ] **Step 1: Apply the edits**

In `README.md`, insert before `## Provenance`:

````markdown
## Checkpoints and stages

A run can survive being killed. Given a directory, training saves its
full state there every five optimizer updates, and the same command
started again resumes from the latest save:

```bash
uv run python run_pipeline.py --checkpoint-dir checkpoints/run-1
```

A resumed run writes the same records as one never interrupted. Each
training draw reseeds from its own index, so a draw replayed after a
resume repeats the one the killed process made, and a save holds what
no seed can rebuild: the weights, Adam's state and the loop's place in
the pool. Checkpointing is off unless the flag is given, and
`--checkpoint-every` sets the cadence. A save of the default model is
about 7.2 GB, and two exist while one is written, so the directory
needs about 15 GB free. A fresh run needs a fresh directory: one that
holds another run's checkpoint is refused, naming the setting that
differs.

`--stage` runs the three stages as separate processes, handing over
through the out and checkpoint directories, and together they write
what the single command writes:

```bash
uv run python run_pipeline.py --stage baseline
uv run python run_pipeline.py --stage train --checkpoint-dir checkpoints/run-1
uv run python run_pipeline.py --stage validate --checkpoint-dir checkpoints/run-1
```

`--skip-training` is `--stage baseline` under its older name. The
design is in
[docs/specs/2026-10-09-checkpoint-resume-design.md](docs/specs/2026-10-09-checkpoint-resume-design.md).

````

In the same file's Provenance section, replace

```
- **Written with AI assistance:** GRPO, the trainer, the evaluator and
  the entry point (`grpo.py`, `trainer.py`, `evaluate.py` and
  `run_pipeline.py`). `report.py` and its tests are AI-written, and
  their headers say so.
```

with

```
- **Written with AI assistance:** GRPO, the trainer, the evaluator and
  the entry point (`grpo.py`, `trainer.py`, `evaluate.py` and
  `run_pipeline.py`). `report.py`, `checkpoint.py` and their tests are
  AI-written, and their headers say so.
```

and replace

```
  brought in the ported `_normalize_text`, has none. Commits from #12
  onward were AI-assisted as well but carry no trailer; this section
  is their disclosure.
```

with

```
  brought in the ported `_normalize_text`, has none. Commits from #12
  through #25 were AI-assisted as well but carry no trailer; this
  section is their disclosure. From the checkpoint and resume work on,
  AI-assisted commits carry the trailer again.
```

In `docs/design-notes.md`, replace the end of the provenance note at the top,

```
> contract* (verifier), the two `grpo.py` sections, and the three
> `evaluate.py` sections, and the two `trainer.py` sections. The
> section skeleton was scaffolded; the content is Naoki's.
```

with

```
> contract* (verifier), the two `grpo.py` sections, and the three
> `evaluate.py` sections, the two `trainer.py` sections, and
> *Checkpoint and resume*. The section skeleton was scaffolded; the
> content is Naoki's.
```

and insert this section before `## Forensics: obligation or convention`:

````markdown
## Checkpoint and resume

**Decision.**
With `--checkpoint-dir`, training saves its full state every `--checkpoint-every` updates, five by default: the policy's weights, Adam's state, the draw history and the pool position, with a stamp of the run's identity. The same command started again resumes from the latest save, and the resumed run's records must equal an uninterrupted run's byte for byte. `--stage` splits baseline, training and validation into separate processes on the same terms.

**Rejected.**
Saving the weights alone and restarting Adam. A third of the size, but Adam's moments set the size of every later update, so the first step after a resume differs from the clean run's, and a resume that changes the records is the failure this exists to prevent.
Replaying from the first draw with no checkpoint. Exact, since every draw is seeded, and free, but an interruption then costs everything training had done.
Saving the RNG state. Each draw reseeds from the training seed plus its index, so the draw count rebuilds it.
A `--resume` flag. A requeued Kubernetes pod runs its original command, so a resume that needs different arguments never happens. A checkpoint is found, checked against the run, and resumed from automatically.
Loading the trained weights for the validation stage with `from_pretrained` on the checkpoint. The provenance stamp records the model's `name_or_path`, so it would name a local directory, and the validation records would differ from a single process's.
A save on SIGTERM. It depends on the grace period covering a 7.2 GB write, and is better judged against measured evictions.

**Evidence.**
(1) `TestResume` trains a tiny real policy with a real Adam straight through, and again from a save after update 2 in fresh objects: the weights are `torch.equal` and every draw matches, flat draws on both sides of the save included. Restarting Adam instead changes the draws.
(2) Qwen3-0.6B ties its embedding to its output head. safetensors' `save_model` stores the pair once and `load_model` restores it still tied. Checked on the real model: 2.38 GB of float32 weights, saved in under a second and read back equal, tensor for tensor.
(3) Qwen3's attention dropout is 0.0, so a training pass draws nothing random, and validating in train mode, where a single process leaves the model after `train()`, changes no number. The handoff sets train mode anyway, so the two paths match without relying on that.
(4) On the real model, a three-problem run at four samples wrote byte-identical records three ways: as one process, as three stage processes, and killed with SIGKILL one update past its save at update 2, then started again with the same command, which resumed at draw 3 and replayed the lost draw.

**Cost accepted.**
A checkpoint is about 7.2 GB, 2.4 GB of weights and 4.8 GB of Adam moments, and two exist during a save. An interruption loses up to `--checkpoint-every` updates of work. A directory that holds a checkpoint resumes, so a fresh run needs a fresh directory, and the identity stamp is what stops one run from continuing another. Validation runs in train mode on both paths rather than in eval mode on both, because changing the single-process path would need its own reproduction run.

**Pinned by.**
`TestResume` and `TestCheckpointCadence` in `tests/test_trainer.py`. `tests/test_checkpoint.py`: the round trip, tied weights, a save killed before `LATEST` moves or failing while it writes, each identity field refused, and the lost-work count. In `tests/test_run_pipeline.py`: a run killed mid-training and started again writes the same records, at the same or another cadence, three stages write what one process writes, the handoff keeps the base model's name, and `--stage validate` refuses incomplete or mismatched inputs before the model loads.

````

In `.gitignore`, add a line `checkpoints/` directly after the existing `results/` line.

- [ ] **Step 2: Check the result**

Run: `git diff --stat && git diff | grep -c '^+.*—'`
Expected: three files changed, and `0`: the edits add no em-dash.

Run: `git check-ignore -v checkpoints/run-1/LATEST`
Expected: the new `.gitignore` line.

Run: `uv run --group dev pytest -q`
Expected: `259 passed`.

- [ ] **Step 3: Commit**

```bash
git add README.md docs/design-notes.md .gitignore
git commit -F - <<'EOF'
docs: document checkpoints, stages and their provenance

The README gains a Checkpoints and stages section with the flags, the
disk a checkpointed run needs and the rule that a fresh run takes a
fresh directory, and its Provenance section names checkpoint.py and
records that AI-assisted commits carry the trailer again. The design
notes record why the state is full, why resume is automatic, and why
the handoff loads the base model by name.

Co-Authored-By: Claude <model> <noreply@anthropic.com>
EOF
```

---

### Task 5: Verify with the real model

**Files:** none changed.

**Interfaces:**
- Consumes: the finished branch.

- [ ] **Step 1: The full suite**

Run: `uv run --group dev pytest -q`
Expected: `259 passed`.

- [ ] **Step 2: One process, three stages, and a killed run write the same bytes**

The script runs a three-problem subset three ways: as one process with no checkpoint, as three stage processes, and as a run killed with SIGKILL once `progress.json` shows update 3 (past the save at update 2) and then started again with the same command. It calls the venv's Python directly, since a SIGKILL sent to `uv run` would not reach its child. `--held-out 1` trains on the subset's first two problems, which stepped on every draw in the committed run, so the training actually updates.

```bash
uv sync --frozen
S=$(mktemp -d)
head -n 3 data/math500_subset50.jsonl > "$S/three.jsonl"
COMMON=(--subset "$S/three.jsonl" --n-samples 4 --held-out 1 --steps 4)
PY=.venv/bin/python

"$PY" run_pipeline.py "${COMMON[@]}" --out-dir "$S/whole"

for stage in baseline train validate; do
  "$PY" run_pipeline.py "${COMMON[@]}" --stage "$stage" \
    --out-dir "$S/staged" --checkpoint-dir "$S/staged-ckpt" --checkpoint-every 2
done

"$PY" run_pipeline.py "${COMMON[@]}" --out-dir "$S/killed" \
  --checkpoint-dir "$S/killed-ckpt" --checkpoint-every 2 &
pid=$!
until grep -qs '"updates": 3' "$S/killed-ckpt/progress.json"; do
  kill -0 "$pid" 2>/dev/null || { echo "ended before update 3"; break; }
  sleep 2
done
kill -9 "$pid"; wait "$pid" 2>/dev/null
"$PY" run_pipeline.py "${COMMON[@]}" --out-dir "$S/killed" \
  --checkpoint-dir "$S/killed-ckpt" --checkpoint-every 2

for run in staged killed; do
  same=yes
  for name in baseline training validation report; do
    cmp "$S/whole/$name.json" "$S/$run/$name.json" || same=no
  done
  echo "$run: byte-identical: $same"
done
cat "$S/killed-ckpt/resumes.jsonl"
du -sh "$S"/killed-ckpt/draw-*
```

Expected: about 25 minutes on the Apple M4 Pro this plan was drafted on, where the three-way check took 24. Both lines read `byte-identical: yes`. The resume prints `resuming at draw 3, update 2 (lost 1 draw, 1 update)`, and `resumes.jsonl` holds the same counts. Training reaches 4 updates over 5 draws, one of them flat, and leaves one `draw-0005` directory: 2.38 GB of weights and 4.77 GB of Adam state, which `du` reports as 6.7G in binary units. If the run ends before update 3, the kill never happens and the comparison proves nothing: report that instead of a pass.

- [ ] **Step 3: Clean up**

Run: `rm -rf "$S"`
Expected: about 14 GB freed, the final checkpoints of the staged and killed runs.

- [ ] **Step 4: The committed report is untouched**

```bash
uv run python -m mini_reasoning.report docs/runs/2026-09-28-dynamic-sampling
git diff --exit-code docs/runs/ && echo "byte-identical"
```

Expected: `byte-identical`. This change leaves `report.py` alone, and this confirms it.

- [ ] **Step 5: Nothing to commit**

Run: `git status --short`
Expected: empty. Verification writes only to the temporary directory and regenerates one file identically.
