"""Contract tests for checkpoint files: what a save writes, what a load
refuses, and that a kill mid-save never leaves a run without a
checkpoint to resume from.

The policy is a real (tiny) module and the optimizer a real Adam that
has taken steps, so the state round-tripped is the state that matters.

Written with Claude (AI-written); see the README's Provenance section.
"""
import errno
import json
import os
import stat

import pytest
import torch
from safetensors import safe_open
from torch import nn

from mini_reasoning import checkpoint
from mini_reasoning.checkpoint import (
    IDENTITY, LATEST, PROGRESS, RESUMES, Writer, load_weights, log_resume,
    read_latest, resume_from, version_warning,
)
from mini_reasoning.evaluate import library_versions

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

    def test_pruning_leaves_files_alone(self, tmp_path):
        """Found in review: pruning removed anything named draw-*, and a
        stray file of that name made every save crash after LATEST had
        moved. Only checkpoint directories are pruned."""
        (tmp_path / "draw-notes.txt").write_text("kept")
        saved_run(tmp_path)
        assert (tmp_path / "draw-notes.txt").read_text() == "kept"
        assert read_latest(tmp_path, IDENTITY_VALUES) is not None

    def test_leftovers_are_pruned_before_a_save_writes(
            self, tmp_path, monkeypatch):
        """Found in review: leftovers from a killed save or prune went only
        after the next save, so a requeue could need room for three
        checkpoints at once. While a save writes, only the checkpoint
        LATEST names may remain beside it."""
        model = policy()
        optimizer = stepped_adam(model)
        writer = Writer(tmp_path, model, IDENTITY_VALUES)
        writer.save(optimizer, HISTORY[:1], position=1, complete=False)
        (tmp_path / "draw-0005").mkdir()  # left by a killed save
        present = []
        real_save_model = checkpoint.save_model

        def recording(model, filename):
            present.append(sorted(p.name for p in tmp_path.glob("draw-*")))
            real_save_model(model, filename)

        monkeypatch.setattr(checkpoint, "save_model", recording)
        writer.save(optimizer, HISTORY, position=2, complete=False)
        assert present == [["draw-0001", "draw-0002"]]

    def test_saving_over_the_current_checkpoint_is_refused(self, tmp_path):
        model = policy()
        optimizer = stepped_adam(model)
        writer = Writer(tmp_path, model, IDENTITY_VALUES)
        writer.save(optimizer, HISTORY, position=2, complete=False)
        with pytest.raises(ValueError, match="current"):
            writer.save(optimizer, HISTORY, position=2, complete=True)


class TestVolumes:
    """Volumes that cannot sync a directory, such as FUSE mounts."""

    @staticmethod
    def failing_fsync(monkeypatch, error, *, directories):
        """os.fsync raising error for directories, or for files."""
        real = os.fsync

        def fsync(fd):
            if stat.S_ISDIR(os.fstat(fd).st_mode) == directories:
                raise OSError(error, os.strerror(error))
            real(fd)

        monkeypatch.setattr(checkpoint.os, "fsync", fsync)

    @pytest.mark.parametrize("error", [errno.EINVAL, errno.ENOTSUP])
    def test_a_volume_that_cannot_sync_a_directory_still_checkpoints(
            self, tmp_path, monkeypatch, error):
        """Found in review: gcsfuse and similar mounts refuse a directory
        fsync, which crashed training at its first draw."""
        self.failing_fsync(monkeypatch, error, directories=True)
        model, _ = saved_run(tmp_path)
        Writer(tmp_path, model, IDENTITY_VALUES).progress(3, 1)
        assert read_latest(tmp_path, IDENTITY_VALUES) is not None
        assert json.loads((tmp_path / PROGRESS).read_text()) == {
            "draws": 3, "updates": 1}

    def test_a_file_that_cannot_sync_still_fails_the_save(
            self, tmp_path, monkeypatch):
        """Only a directory's sync may be refused. A file that cannot
        reach the disk is a checkpoint that may not exist."""
        self.failing_fsync(monkeypatch, errno.EINVAL, directories=False)
        with pytest.raises(OSError):
            saved_run(tmp_path)


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


class TestVersions:
    """A checkpoint records the library versions it was saved under."""

    def test_a_save_records_the_library_versions(self, tmp_path):
        saved_run(tmp_path)
        saved = read_latest(tmp_path, IDENTITY_VALUES)
        assert saved.versions == library_versions()

    def test_matching_versions_warn_of_nothing(self, tmp_path):
        saved_run(tmp_path)
        assert version_warning(read_latest(tmp_path, IDENTITY_VALUES)) is None

    def test_other_versions_are_named_in_a_warning(self, tmp_path):
        """Found in review: a requeue on a rebuilt image resumed without
        a word, and the records it wrote then need not match."""
        saved_run(tmp_path)
        state_path = tmp_path / "draw-0002" / checkpoint.STATE
        state = json.loads(state_path.read_text())
        state["versions"]["torch"] = "0.0.1"
        state_path.write_text(json.dumps(state))
        warning = version_warning(read_latest(tmp_path, IDENTITY_VALUES))
        assert "torch 0.0.1" in warning
        assert f"torch {library_versions()['torch']}" in warning


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

    def test_an_interrupted_log_write_leaves_whole_lines(
            self, tmp_path, monkeypatch):
        """Found in review: resumes.jsonl was appended in place, so a kill
        mid-append could leave a torn last line its readers cannot parse.
        It is written whole or not at all, like every other file here."""
        saved_run(tmp_path)
        saved = read_latest(tmp_path, IDENTITY_VALUES)
        log_resume(tmp_path, saved)
        before = (tmp_path / RESUMES).read_text()

        def killed(*args, **kwargs):
            raise KeyboardInterrupt("killed mid-write")

        monkeypatch.setattr(checkpoint.os, "replace", killed)
        with pytest.raises(KeyboardInterrupt):
            log_resume(tmp_path, saved)
        monkeypatch.undo()
        assert (tmp_path / RESUMES).read_text() == before

    def test_a_restart_before_any_save_counts_from_the_first_draw(
            self, tmp_path):
        """Found in review: an eviction before the first save left no
        checkpoint, so nothing was logged and its cost went unmeasured."""
        (tmp_path / PROGRESS).write_text(
            json.dumps({"draws": 4, "updates": 3}))
        line = log_resume(tmp_path, None)
        assert line == "resuming at draw 0, update 0 (lost 4 draws, 3 updates)"
        entry = json.loads((tmp_path / RESUMES).read_text())
        assert entry["lost"] == {"draws": 4, "updates": 3}

    def test_a_fresh_start_logs_nothing(self, tmp_path):
        assert log_resume(tmp_path, None) is None
        assert not (tmp_path / RESUMES).exists()
