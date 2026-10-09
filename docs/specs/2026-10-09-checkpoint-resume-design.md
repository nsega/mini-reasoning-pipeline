# Checkpoint and resume for GRPO training

Status: design approved in conversation, 2026-10-09. Awaiting spec review before an implementation plan.

## Problem

[kueue-training-lab](https://github.com/nsega/kueue-training-lab) runs this pipeline under Kueue, which preempts and evicts running workloads. An evicted run restarts from nothing today: the baseline eval (about an hour) and training (about 69 minutes) both run again. The lab measures what an eviction costs under three strategies (one Job, one Job checkpointed, three Workloads), and it judges every experiment by this repository's own standard: the records must equal a clean run's. So a resume has to be exact. A resume that is merely close is a failed experiment.

Most of what exactness needs is already here. Each eval seeds problem i with `--seed` + i, and training seeds draw d with the training seed + d, so the global RNG never has to be saved: the draw count rebuilds it. Qwen3's attention dropout is 0.0, so a training forward pass draws nothing random either. What a killed process loses is the state no seed can rebuild: the policy's weights, Adam's moments and step counts, and the loop's place in its pool.

## Decisions

| question | decision | rejected, and why |
|---|---|---|
| What a checkpoint holds | Full state at update boundaries: weights, Adam state, the draw history and the pool position | Weights only, with Adam restarted: a third of the size, but the first update after a resume differs from a clean run's, so it can never pass EXP-0. Replay from draw 0 with no checkpoint: exact and free, but it loses up to 69 minutes per eviction. It is the eviction-cost table's no-checkpoint baseline, which needs no code |
| How often | Every 5 optimizer updates, set by a flag | Every update: at most about 2 minutes lost, but 40 saves and about 290 GB written per run, at a save time nobody has measured. Every 5 updates sits exactly at EXP-1's limit of 5 updates lost |
| Where it lives | `main`, opt-in through `--checkpoint-dir` | On by default: it changes the README's default run and its disk needs. A long-lived lab branch: `main` stays as published, but every fix has to cross between branches, and the lab would pin a commit off `main` |
| Scope | Mid-training resume, and a final-weights handoff so the three stages can run as separate processes | Resume only: week 5's three-Workload split would need a second design. A save on SIGTERM: it depends on the grace period covering a 7 GB write, and is better judged against EXP-1's measurements |
| Who owns the files | A new module, `mini_reasoning/checkpoint.py`. The trainer calls a checkpointer it is handed | The trainer writing files itself: fewer parts, but every trainer test would touch disk |
| How a requeued pod resumes | Automatically, when `--checkpoint-dir` holds a checkpoint of the same run | A `--resume` flag: Kueue requeues a pod with its original command, so resuming cannot depend on changing the arguments |

## Design

### 1. Checkpoint contents and layout

```
<checkpoint-dir>/
  LATEST              one line naming the newest complete checkpoint
  draw-0032/
    model.safetensors weights, float32, tied tensors stored once (2.4 GB)
    optimizer.pt      Adam's state_dict: exp_avg, exp_avg_sq and step per parameter (4.8 GB)
    state.json        loop state and identity
  progress.json       draws and updates reached, rewritten after every draw
  resumes.jsonl       one line per resume
```

- `model.safetensors` is written with `safetensors.torch.save_model`, which stores Qwen3's tied embedding and output head once, and read with `safetensors.torch.load_model` into a model already loaded by name. Everything stays float32, so the round trip is exact.
- `optimizer.pt` is `optimizer.state_dict()` through `torch.save`, read with `torch.load(..., weights_only=True)`.
- `state.json` holds:
  - Loop state: `history`, the per-draw reports exactly as `training.json` stores them under `"steps"`, and `position`, the pool index the loop examines next. Draws, updates and the retired set all derive from `history`. They are not stored, so they cannot disagree with it.
  - Identity: `model`, `dtype`, `seed` (the training seed), `steps`, `max_draws` (resolved, never None), `group_size`, `lr` and `pool` (the ids, in order). A load compares each field with the run about to resume and refuses on the first mismatch, naming the field, the way `load_baseline` treats its provenance stamp. Reusing a directory for another run therefore fails loudly instead of continuing that run's training.
  - `complete`: true only on the save made when the loop stops. That save is the final-weights handoff.
  - `format`: 1. A load refuses any other value.
- A checkpoint directory is named by its draw count, zero-padded to four digits. Draws name it because a final save can follow a periodic save at the same update count, when flat draws come after the last update.
- A save writes and flushes the new directory, then replaces `LATEST` with `os.replace`, which is atomic on POSIX, then deletes every other `draw-*` directory. A kill at any point leaves `LATEST` naming a complete checkpoint. A directory `LATEST` does not name is a partial write: a load ignores it, and the next save removes it.
- `LATEST` naming a directory that does not exist is an error, not a fresh start. It means someone deleted part of a checkpoint by hand.
- `progress.json` holds `{"draws": d, "updates": u}` and is replaced atomically after every draw. A resume appends one line to `resumes.jsonl`: where it resumed, what `progress.json` says the killed process had reached, the difference as lost draws and lost updates, and the time. These two files exist for EXP-1's lost-work measurement, and they stay out of `training.json` so a resumed run's records match a clean run's byte for byte.
- Size: 596M parameters in float32 are 2.4 GB of weights and 4.8 GB of Adam moments, about 7.2 GB per checkpoint and 14.4 GB at the peak of a save.

### 2. Trainer

`train()` gains three keyword arguments. Their defaults keep today's behaviour, so every existing caller and test is unaffected.

```python
train(model, tokenizer, problems, *, steps, rollout, group_size=7, lr=1e-6,
      max_grad_norm=MAX_GRAD_NORM, max_draws=None, seed=None,
      resume: Resume | None = None,
      checkpointer: Checkpointer | None = None, checkpoint_every: int = 5)
```

- `Resume` and `Checkpointer` are defined in `trainer.py`, since they are the trainer's interface. `checkpoint.py` imports them, so the dependency points from the files to the loop.
- `Resume` carries `history`, `position` and `optimizer_state`. The caller has already loaded the weights into `model`. `train()` builds Adam as today and then loads `optimizer_state` into it. It sets `draws` to the length of `history`, `updates` to the count of stepped draws, and the retired set to the problems of the draws that did not step. The loop then runs unchanged, and its next draw seeds with the training seed + `draws`, the seed the killed process used for that same draw.
- `train()` raises `ValueError` for a resume without a `seed`, since unseeded draws cannot replay, for a resume whose history names a problem outside the pool, and for `checkpoint_every` below 1.
- `Checkpointer` is a protocol with two methods. The trainer decides when, because cadence is counted in updates, a training concept. The checkpointer decides how, and is built around the model it saves.
  - `progress(draws, updates)` runs after every draw.
  - `save(optimizer, history, position, *, complete)` runs after each draw that brings the update count to a multiple of `checkpoint_every`, and once more when the loop stops, with `complete=True`. A periodic save due on the stopping draw becomes that final save, so the same state is never written twice.
- The order within a draw stays as it is: the step, then the history entry and the counters, then `progress`, then any save. Everything after the last save is replayed on resume, and the per-draw seeds make the replay identical.
- The return value is unchanged. Its `steps` list includes the draws made before the resume, so a resumed run writes the same `training.json` as a clean one.

### 3. Entry point

New flags, checked at parse time before the model loads, as `--held-out` is today:

- `--checkpoint-dir DIR` turns checkpointing on. Without it every command behaves exactly as today. It is separate from `--out-dir` so the lab can keep 7 GB checkpoints on a volume inside the cluster while the small JSON records land where its comparison tool reads them.
- `--checkpoint-every N` counts updates and defaults to 5. Giving it without `--checkpoint-dir` is an error.
- `--stage {all,baseline,train,validate}` defaults to `all`. `--skip-training` stays as the README documents it, as a synonym for `--stage baseline`. Combining it with any other stage is an error.

| stage | needs | does | writes |
|---|---|---|---|
| `baseline` | nothing | the baseline eval | `baseline.json` |
| `train` | `--checkpoint-dir` | trains, resuming from a checkpoint if one exists. A complete checkpoint means training is done: it only rewrites `training.json` from that checkpoint, so a kill between the final save and the write loses nothing | `training.json`, checkpoints |
| `validate` | a complete checkpoint, `baseline.json` and `training.json` | loads the base model by name, loads the final weights into it, puts it in train mode, runs the validation eval | `validation.json`, `report.json` |
| `all` | nothing | today's flow, with automatic resume when `--checkpoint-dir` is given | all four |

- `train` writes `training.json` the way `run()` does today, with `held_out` and `seed` added to the history.
- `validate` checks the complete checkpoint's identity against its own flags, as a resume does, and checks that the draws in `training.json` equal the checkpoint's history, all before it loads the model, so the report pairs with the weights it validated. It reads the baseline through `load_baseline`, with its usual checks.
- The handoff loads the base model by name because the provenance stamp records `config.name_or_path`. Loading the trained weights with `from_pretrained` on the checkpoint directory would stamp a local path where a clean run stamps `Qwen/Qwen3-0.6B-Base`, and the records would differ.
- The handoff puts the model in train mode because a clean run validates the model `train()` left in train mode. At a dropout of 0.0 the mode should change no number, but the criterion is bitwise identity, so the two paths match.
- Automatic resume with `--checkpoint-dir`, in `all` and `train`:
  - A `LATEST` checkpoint whose identity matches resumes. One whose identity differs refuses, naming the field. No checkpoint means a fresh start.
  - In `all`, an existing `baseline.json` in `--out-dir` is reused with the checks `--reuse-baseline` applies. The baseline eval is deterministic in its seed, so reuse cannot change the records. It only keeps a kill before the first save from costing the baseline's hour again.
  - In `all`, a complete checkpoint skips training: the weights are loaded, the model is put in train mode, `training.json` is written from the checkpoint, and validation runs.
  - Each resume prints one line, for example `resuming at draw 32, update 20 (lost 3 draws, 1 update)`, and appends it to `resumes.jsonl`.
- Saves prune older `draw-*` directories inside `--checkpoint-dir`, but nothing removes the `--checkpoint-dir` itself. A fresh run takes a new one.

How the lab's strategies map to commands:

- One Job, no checkpoint: today's command. A requeue replays everything.
- One Job, checkpointed: today's command plus `--checkpoint-dir`.
- Three Workloads: `--stage baseline`, then `--stage train`, then `--stage validate`, sharing `--out-dir` and `--checkpoint-dir`.

### 4. Records and documentation

- No record format changes. `baseline.json`, `training.json`, `validation.json` and `report.json` from a resumed or staged run are byte-identical to a clean run's. `progress.json` and `resumes.jsonl` live only in the checkpoint directory.
- `pyproject.toml` declares `safetensors` directly. The lock already resolves 0.8.0 through transformers, so no resolved version changes.
- `README.md` documents the three flags, automatic resume, the rule that a fresh run takes a new directory, and the disk a checkpointed run needs: 7.2 GB per checkpoint and 14.4 GB at the peak of a save.
- `docs/design-notes.md` gains a section on checkpoint and resume: why the state is full, why per-draw seeding makes saved RNG state unnecessary, why the handoff loads by name and keeps train mode, and why resume is automatic.

## Testing

Tests are written first and watched failing, as for every change here. None of them loads the real model.

- `tests/test_checkpoint.py`, over the existing tiny real policy, a real Adam optimizer and `tmp_path`:
  - A save and load round trip returns equal weights, equal optimizer moments and step counts, and an equal history and position.
  - A policy with tied tensors saves them once and loads them still tied.
  - A save that fails before `LATEST` is replaced, simulated with a raising `os.replace`, leaves the previous checkpoint current, and the next save removes the partial directory.
  - Each identity field, mismatched in turn, refuses with that field named. A `LATEST` naming a missing directory raises. An unknown `format` raises.
  - Lost work on resume is computed from `progress.json` minus the checkpoint, and appended to `resumes.jsonl`.
- `tests/test_trainer.py`:
  - The core property. One run trains straight to 6 updates. A second run, built from fresh objects, resumes from the in-memory snapshot of the save after update 2 (cadence 2). Its final weights are `torch.equal` to the first run's and its history is equal. The rollout double draws its rewards from torch's RNG and produces flat groups, so both the per-draw seeding and the restored retired set are exercised.
  - Saves fall on every Nth update and once at the stop with `complete=True`. A periodic save due on the stopping draw folds into the final save. `progress` runs after every draw.
  - A resume without a seed raises, as do a history outside the pool and a `checkpoint_every` below 1.
- `tests/test_run_pipeline.py`, with the injected model, sampler and rollout doubles:
  - A rollout double raises at draw k to simulate a kill, and `run()` is called again with the same arguments. The four record files equal an unbroken run's.
  - The three stages run in sequence produce the same four files as `--stage all`.
  - Each flag conflict fails at parse time. `validate` refuses an incomplete checkpoint, a missing `training.json`, and a `training.json` that disagrees with the checkpoint.
  - After the handoff, the validation provenance names the base model, not a path.
  - With `--checkpoint-dir`, an existing `baseline.json` is reused and a mismatched one refused.

## EXP-0

EXP-0 in kueue-training-lab's plan, run in week 3 on the Apple M4 Pro with the real model, float32, CPU only:

1. Reference: a clean run on `main` after this change merges, with checkpointing off. The committed `docs/runs/2026-09-28-dynamic-sampling` cannot be the reference as it stands, because it predates #18 and its `training.json` has no `seed` field, which every run now writes. The new clean run should reproduce the committed records apart from that field, which re-checks the baseline everything else rests on. It takes about 3 hours of unattended machine time.
2. Candidate: the same command with `--checkpoint-dir`, its out-dir seeded with the reference's `baseline.json`. Send SIGKILL once `progress.json` shows update 22, between the saves at updates 20 and 25, so the resume replays real draws. A kill exactly at update 20 would leave nothing to replay. Then rerun the identical command. It takes about 2.2 hours.
3. Pass: every record identical. That means the baseline and validation records and their provenance, all draws down to the gradient norms, and `report.json` byte-identical, checked with the lab's `make compare`. A mismatch is a finding, fixed before the lab's Stage 1 starts.
4. Record in the run note: measured save time and checkpoint size, which replace this design's estimates, and the `resumes.jsonl` line.

## Out of scope

- A save on SIGTERM. Worth revisiting with EXP-1's measurements of eviction timing.
- Keeping more than the latest checkpoint.
- Distributed training.
- GPU determinism settings. GPU runs are only ever compared with a GPU reference, and Stage 2 needs its own review.
- Sharded evaluation.
- Any change to the record formats or to `report.py`.
- The lab's Kueue objects, manifests and stage driver.

## Costs accepted

- 7.2 GB per checkpoint, 14.4 GB at the peak of a save, and about 58 GB written by a default run's 8 saves. Save time is unmeasured until EXP-0.
- Up to 5 updates of lost work per eviction: about 8 draws, or 9 minutes.
- A reused checkpoint directory resumes. The identity check is the guard against resuming the wrong run.
- The validate stage reproduces the single-process path's train mode during validation instead of switching both paths to eval mode. Changing the clean path would need its own reproduction run.
- `progress.json` adds one small atomic write per draw, about every 65 seconds.
