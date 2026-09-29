# Run of 2026-09-28, dynamic sampling

The first run with dynamic sampling and the reserved held-out split
(PR #16), and the comparison that split was built for: this run
against [the 2026-09-21 float32 run](../2026-09-21-float32/), its
control.

## Provenance

`baseline.json` and `validation.json` carry their own `provenance`
stamp: model `Qwen/Qwen3-0.6B-Base`, dtype float32, seed 42, 7
samples, fallback `number`, torch 2.13.0, transformers 5.15.0. What
the stamp does not cover:

| | |
|---|---|
| code | `main` at `2c0242e` (PR #16 merged) |
| training | `--steps 40` updates, `--max-draws` default 120, Adam at 1e-6, group 7, no KL, clip 1.0 |
| held out | `--held-out 10`: problems 41-50, never drawn |
| training seed | 92, the run's 42 plus the subset's 50, so no training draw shares an eval seed |
| machine | Apple M4 Pro, 14 cores, 48 GB, macOS 27.0, CPU only |

## Timings

| stage | wall clock |
|---|---|
| baseline eval | 59m40s |
| training, 40 updates over 64 draws | 68m56s |
| validation eval and report | 51m04s |
| total | 2h59m40s |

## Results

Selected answer: 0.320 before, 0.300 after (level-reweighted 0.444 to
0.418). Per sample, from `report.json`:

| group | n | eval grader | boxed reward |
|---|---|---|---|
| all | 50 | 0.160 -> 0.160 (-0.000, p 1.000) | 0.140 -> 0.134 (-0.006, p 0.866) |
| stepped | 20 | 0.314 -> 0.286 (-0.029, p 0.594) | 0.271 -> 0.250 (-0.021, p 0.660) |
| flat | 20 | 0.021 -> 0.057 (+0.036, p 0.251) | 0.014 -> 0.029 (+0.014, p 0.502) |
| held_out | 10 | 0.129 -> 0.114 (-0.014, p 1.000) | 0.129 -> 0.114 (-0.014, p 1.000) |

**No group moved beyond noise.** Forty real optimizer updates at lr
1e-6 did not measurably change the policy, on the problems it trained
on or on the ten it never saw.

## Against the control

**The before half is identical.** This run's baseline records equal
the control's record for record: same seed, same dtype, same subset.

**The control's stepped gain does not survive.** The control's
stepped group rose 6.2 points on boxed reward (p = 0.10), which its
note called suggestive, not established. The two runs stepped on
different problems (23 and 20, sharing 17), so the fair comparison is
on the shared 17, against the shared baseline:

| run | updates | change on the 17 shared stepped problems |
|---|---|---|
| control | 23 | +0.034 (p 0.494) |
| this run | 40 | -0.034 (p 0.438) |

Nearly doubling the updates flipped the sign. That is the shape of
noise, not of a training effect.

**The held-out problems** fell 1.4 points here and 2.9 in the control,
where they were untouched rather than reserved. Neither is a finding;
the split's design notes explain why ten problems cannot be.

## How training spent its draws

- 64 draws reached 40 updates, stopping on `updates`, well under the
  120-draw cap.
- 20 problems went flat on their first draw and retired. One of them
  had any boxed reward in the baseline, which is the cost the design
  accepted: about one chance-flat problem in seventeen.
- 4 problems stepped and later went flat, retiring then.
- Updates per stepped problem: one for 4 problems, two for 12, three
  for 4.

## What it means

At this learning rate and budget, GRPO does not measurably move this
model on this subset. The instrumentation is what makes that
statement possible: the split kept the held-out set clean, the
per-sample report kept a lucky headline from standing, and the shared
baseline made the two runs comparable problem by problem. The
learning rate is the prime suspect for any follow-up. The design
notes record that 1e-6 was chosen against the 1/T scaling of the
per-rollout mean, not independently of it.

## Known gaps

- One seed. Every p here is within a single run's sampling noise.
- Peak memory was not measured, so the README's 8.6 GB figure is
  still the pre-split measurement.

**Later:** measured on 2026-09-29 through the first optimizer updates
of a short run, training peaks at about 14 GB, not 8.6. The README
now says so.
