# Dynamic sampling with a reserved held-out split

Status: design approved in conversation, 2026-09-25. Awaiting spec review before an implementation plan.

## Problem

The first float32 run took 23 optimizer steps out of 40 draws. The other 17 draws produced flat groups (all seven rollouts scoring the same boxed-only reward) and were skipped by design, and 16 of those 17 problems also score zero boxed reward in the independent baseline eval draw. They are out of reach at this scale, not unlucky, so every flat draw spends a generation and buys no gradient.

The budget is what traps the run there. `--steps` counts draws, and 40 draws over a 40-problem training pool is exactly one pass, whatever each draw yields. Counting optimizer updates instead lets training make further passes over the problems that do give gradient.

Doing that naively would destroy the only held-out signal the run has. Problems 41-50 are untouched today only because 40 draws in file order stop at problem 40. A second pass would continue past them. The held-out set has to become a reservation rather than an accident, in the same change.

## Decisions

| question | decision | rejected, and why |
|---|---|---|
| Where training problems come from | A split within the frozen 50 | An external pool from MATH's training split: all 50 eval problems would be held out, but it drops the train-on-test measurement the project chose to keep, and adds a second frozen artifact with its own generator and pins |
| Which problems are held out | The last 10 of the subset: problems 41-50 | A level-stratified seeded draw at k=10 or k=20: more representative or more sensitive, but both break comparability with the committed run |
| What happens to a flat problem | Retired for the rest of the run after one flat draw | Redrawn every pass: protects a problem that was flat by chance, at about 50 s of generation per redraw, when 16 of 17 flat problems are unreachable |
| Pre-filtering from the stored baseline | Not done | It selects training problems with the eval's own draw, so validation regresses toward the mean on exactly the groups the report compares, biasing the before/after downward |

Holding out problems 41-50 makes the committed run `docs/runs/2026-09-21-float32` this change's control. Same training pool (1-40), same held-out set (41-50), same eval over all 50. The one variable that differs is how training reaches the pool.

## What the held-out result can and cannot claim

Measured from the committed run, the per-problem change in boxed reward has a standard deviation of 0.124. The smallest effect a held-out group of k problems detects at 80% power is about 2.8 × 0.124 / √k:

| held-out size | detectable effect |
|---|---|
| 10 | 10.9 points |
| 20 | 7.7 points |
| 25 | 6.9 points |
| all 50 | 4.9 points |

The effect measured on the problems training stepped on was 6.2 points, and a generalization effect is normally smaller than the effect on the training problems themselves. No split of these 50 problems can support a generalization claim. The held-out result is directional only, and the design notes must say so where the split is described.

## Design

### 1. Split and pool

- A new flag `--held-out N`, default 10, reserves the last N problems of the frozen subset. The subset's byte order is already pinned by its sha256 test, so "last N" is a stable definition.
- `split_held_out(problems, n) -> tuple[list, list]` in `run_pipeline.py` returns `(pool, held_out)`. It raises `ValueError` when `n < 0`, or when `n >= len(problems)` and nothing would be left to train on. `--held-out 0` reserves nothing.
- Both evals, baseline and validation, still run over all 50 problems. Only training receives the pool. The baseline eval is therefore identical before and after this change, and `--reuse-baseline` stays valid across it.
- `run()` writes the reserved ids into the history as `held_out` before writing `training.json`.

### 2. Trainer loop

- `--steps` counts optimizer updates, not draws. A new flag `--max-draws` bounds runtime, defaulting to 3 × `--steps`.
- Signature: `train(model, tokenizer, problems, *, steps, rollout, group_size=7, lr=1e-6, max_grad_norm=MAX_GRAD_NORM, max_draws=None, seed=None)`. `max_draws=None` means 3 × steps; `max_draws < 1` raises `ValueError`, as `steps < 1` and an empty pool already do. A `max_draws` below `steps` is allowed: it is a runtime cap, and the run stops on it with `"max_draws"`.
- The loop cycles the pool in order, skipping retired problems without generating for them. Each draw runs `train_step` unchanged. A draw that does not step retires its problem, whether the group was all wrong or all right. A problem that stepped on an earlier pass and goes flat later retires too.
- It stops at the first of three conditions, checked in this order: `steps` updates reached (`"updates"`), `max_draws` draws reached (`"max_draws"`), no live problem left (`"pool_exhausted"`). A run meeting its update target on its last permitted draw records `"updates"`.
- Each draw is seeded with `torch.manual_seed(seed + draw)`, where `draw` counts draws from 0, when `seed` is given. `run()` passes `--seed` plus the subset's size, so training's stream starts past every seed the evals use (they seed problem i with `--seed` + i). Amended after the final review: passing `--seed` itself made draw d share problem d's seed and sampling call, so the first pass replayed the baseline's own samples and first-pass retirement became the pre-filter rejected above. Training rollouts are unseeded today and inherit whatever state the global RNG has reached. That makes retirement unreproducible, and it means a `--reuse-baseline` run trains differently from a full run, because skipping the baseline eval, which seeds per problem, leaves the RNG elsewhere. Seeding per draw fixes both. The new run will not replay the committed run's first pass draw for draw, since that run's training was unseeded; the control is the design, not the random draws.
- The return value keeps the per-draw list under the existing `"steps"` key, each entry unchanged (`unique_id`, `mean_reward`, `grad_norm`, `stepped`), and adds `"updates"`, `"draws"` and `"stopped"`. Renaming `"steps"` would break `report.py` and the committed run.

### 3. Records and report

- `training.json` gains `held_out` (from `run()`) and `updates`, `draws`, `stopped` (from the trainer). A run always writes `held_out`, as `[]` under `--held-out 0`, so the format is never ambiguous.
- `report.training_groups` takes an optional `held_out` list. When the history carries `held_out`, the report has four groups: `stepped` (stepped on at least one draw), `flat` (drawn, never stepped), `held_out` (the reserved ids, read from the key rather than inferred from draws), and `untouched` (pool problems never drawn, empty unless training stopped before one full pass).
- A history without `held_out` gets today's three groups and today's output, so the committed run's `report.json` regenerates byte-identical. `GROUPS` keeps its current three names; the split format's four are a second constant, `SPLIT_GROUPS`.
- The table gains the `held_out` row. Its empty message is `none: nothing was reserved`. In the split format, `untouched`'s empty message is `none: training drew every problem in its pool`, since held-out problems have their own row; the old format keeps its current message.
- The comparison across runs is group to group: the committed run's `untouched` (41-50) against the new run's `held_out` (41-50).
- The end-of-run summary replaces `training steps taken 23 of 40` with the counts that now mean something, in the form `training updates        40 over 61 draws (stopped: updates)`.

### 4. Documentation

- `docs/design-notes.md`: the flat-group section gains retirement; the sentence "A run's step count is an upper bound rather than a count of updates" inverts, since updates become exact and draws the upper bound; a new section on the held-out split records why the last 10, and the power table above with the directional-only conclusion.
- `README.md`: documents `--held-out` and the new meaning of `--steps`. Its timings were measured with 40 draws of training, so it says the training stage now runs to 40 updates and that the next measured run replaces the figures. No estimated number replaces a measured one.

## Testing

Tests are written first and watched failing, as for every change here.

- Split, in `tests/test_run_pipeline.py`: the default reserves exactly the problems the committed run never drew, read from `docs/runs/2026-09-21-float32/training.json` rather than a copied list; `--held-out 0` reserves nothing; a negative N and an N leaving no pool both raise; training receives only the pool while both evals cover every problem; `training.json` records the reserved ids.
- Trainer, in `tests/test_trainer.py`, over the existing tiny real policy and a real Adam optimizer: steps count updates, so a pool with flat problems draws more than it steps and reaches exactly `steps` updates; a retired problem is never drawn again; an all-right group retires like an all-wrong one; a problem that steps and later goes flat retires after the flat draw; each stop condition ends the loop and is recorded; `max_draws` defaults to 3 × steps; the same seed gives identical histories and a different seed does not, using a rollout double that draws its rewards from torch's RNG.
- Report, in `tests/test_report.py`: a history with `held_out` yields the four split groups, with `held_out` taken from the key; `untouched` holds only pool problems never drawn; a history without the key yields today's three groups, and the committed run's `report.json` regenerates byte-identical, compared against the committed file.

Verification with the real model before the change is called done: a run over 3 problems with `--held-out 1`, 2 samples and 2 updates, checking that the held-out problem is never drawn and that `updates`, `draws`, `stopped` and the `held_out` group appear in `training.json` and the report. The committed run's report regenerates byte-identical on the finished branch.

## Out of scope

- The full experiment run (about three hours). It is the next step, and its measured timings are what update the README.
- A per-group count of the updates each group absorbed. Every draw is already in `training.json`, and the field would break the old report's byte identity for a number nothing yet needs.
- An external training pool, reward shaping (closed by `design-notes.md` on role 2's strictness), and learning-rate changes (1e-6 was chosen against the 1/T mean scaling).

## Costs accepted

- About one chance-flat problem in 17 is wrongly retired, going by the baseline evidence.
- The default run gets longer: an estimated 60 draws and about 70 minutes of training where the committed run took 40 draws and 44 minutes. The estimate is replaced by the first measured run.
- Stepped problems are trained about twice each, which strengthens the train-on-test effect the `stepped` group measures. The `held_out` group is untouched by construction.
- `--steps` changes meaning for the documented default command. A command that meant 40 draws now means 40 updates.
