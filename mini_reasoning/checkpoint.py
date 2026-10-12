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

That guarantee needs an atomic rename, which a local disk or a block
volume gives. On an object-storage mount such as gcsfuse a rename can
be a copy and a delete, so checkpoints belong on the former. Such
mounts may also refuse to sync a directory, which only loosens how
soon a rename is durable, so that refusal is tolerated.

progress.json and resumes.jsonl beside them measure the work an
eviction cost. They stay out of the run's records on purpose: a resumed
run's training.json must equal a clean run's byte for byte.

Written with Claude (AI-written); see the README's Provenance section.
"""
import errno
import json
import os
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch
from safetensors.torch import load_model, save_model

from mini_reasoning.evaluate import library_versions
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
# What a volume that cannot sync a directory answers. ENOTSUP and
# EOPNOTSUPP are one value on Linux and two on macOS.
UNSYNCABLE_DIRECTORY = {errno.EINVAL, errno.ENOTSUP, errno.EOPNOTSUPP}


@dataclass(frozen=True)
class Saved:
    """A checkpoint's state.json, read and checked against this run.

    Attributes:
        path: The draw-NNNN directory.
        history: The per-draw reports up to the save.
        position: The pool index the loop examines next.
        complete: True when the save was made as the loop stopped.
        dtype: The dtype of the saved weights.
        versions: The library versions the save was made under.
    """
    path: Path
    history: list[dict]
    position: int
    complete: bool
    dtype: str
    versions: dict

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
    save does not take the save with it.

    A directory sync makes a rename survive a crash. A volume that
    refuses one, as FUSE mounts can with EINVAL or ENOTSUP, is
    tolerated: LATEST may then come back naming the previous
    checkpoint, which is still complete. A file sync is never optional,
    since a file that cannot reach the disk is a checkpoint that may not
    exist.
    """
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    except OSError as err:
        if not (err.errno in UNSYNCABLE_DIRECTORY and path.is_dir()):
            raise
    finally:
        os.close(fd)


def replace_text(path: Path, text: str) -> None:
    """Writes path whole or not at all: a reader sees the old text or the
    new, never a torn one. The entry point writes its records this way
    too, since a requeued run reads back whatever it finds."""
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
        replace_text(self.directory / PROGRESS,
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
                 "versions": library_versions(),
                 "history": history, "position": position,
                 "complete": complete}
        (target / STATE).write_text(json.dumps(state, indent=2) + "\n")
        for item in (target / WEIGHTS, target / OPTIMIZER, target / STATE,
                     target):
            _sync(item)
        replace_text(latest, name + "\n")
        for old in self.directory.glob(PREFIX + "*"):
            # Checkpoints are directories. Anything else named like one
            # is not this module's to delete.
            if old.name != name and old.is_dir():
                shutil.rmtree(old)


def started(directory) -> bool:
    """Whether training began in this checkpoint directory.

    A save exists, or progress was recorded, which a run does as soon as
    its training starts. Either comes after that run's baseline eval.
    """
    directory = Path(directory)
    return (directory / LATEST).exists() or (directory / PROGRESS).exists()


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
                 dtype=saved_identity["dtype"], versions=state["versions"])


def version_warning(saved: Saved) -> str | None:
    """A warning naming the libraries that changed since the save, or None.

    Versions are recorded and compared but never refused, as the eval's
    provenance stamp treats them: a patch bump need not change a number,
    but a resume under a rebuilt image is the first suspect when the
    records stop matching.
    """
    current = library_versions()
    changed = [name for name in current
               if saved.versions.get(name) != current[name]]
    if not changed:
        return None
    then = ", ".join(f"{name} {saved.versions.get(name)}" for name in changed)
    now = ", ".join(f"{name} {current[name]}" for name in changed)
    return (f"warning: the checkpoint was saved under {then}, and this run "
            f"has {now}. Its records may not match an uninterrupted run's.")


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


def log_resume(directory, saved: Saved | None) -> str | None:
    """Appends one line to resumes.jsonl and returns it as printed.

    Lost work is what progress.json says the interrupted attempt reached,
    minus where this one starts: the checkpoint, or the first draw when
    no save was made yet. With neither a checkpoint nor any recorded
    progress, nothing was interrupted, and nothing is logged.
    """
    directory = Path(directory)
    progress = directory / PROGRESS
    if saved is None and not progress.exists():
        return None
    at = ({"draws": saved.draws, "updates": saved.updates}
          if saved is not None else {"draws": 0, "updates": 0})
    reached = json.loads(progress.read_text()) if progress.exists() else at
    lost = {key: reached[key] - at[key] for key in at}
    entry = {"time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "resumed_at": at, "reached": reached, "lost": lost}
    with open(directory / RESUMES, "a") as log:
        log.write(json.dumps(entry) + "\n")
    return (f"resuming at draw {at['draws']}, update {at['updates']} "
            f"(lost {_count(lost['draws'], 'draw')}, "
            f"{_count(lost['updates'], 'update')})")
