# Run of 2026-09-21, float32

The first full pipeline run after the dtype fix, kept because nothing
in the JSON beside it records what produced it. Committed here rather
than in `results/`, which is ignored: `--reuse-baseline` reads that
directory and cannot verify the model or dtype behind a stored
baseline, so a committed one would be reused by a clone in silence.

## Provenance

| | |
|---|---|
| model | `Qwen/Qwen3-0.6B-Base` |
| dtype | float32, passed explicitly (commit `2316278`) |
| seed | 42 |
| subset | `data/math500_subset50.jsonl`, 50 problems |
| sampling | 7 samples, 512-token cap, temperature 0.7, top-p 0.95 |
| training | 40 steps, group 7, Adam at 1e-6, no KL, clip 1.0 |
| versions | python 3.13.1, torch 2.13.0, transformers 5.15.0 |
| machine | Apple M4 Pro, 14 cores, 48 GB, macOS 26.6.2, CPU only |

The baseline comes from a run that was killed during training by a
low-memory kill, after it had written `baseline.json`. Training and
validation come from the run that followed it with
`--reuse-baseline`, which is what that flag exists for. Both ran the
same code at the same seed, so the before half is the before half.

## Timings

| stage | wall clock |
|---|---|
| baseline eval | 56m56s |
| GRPO training, 40 steps | 44m19s |
| validation eval | 53m55s |
| total | 2h35m |

That is 68 seconds per problem, against the 40 the README claimed
before this run corrected it.

## Results

| | baseline | validation |
|---|---|---|
| accuracy | 0.320 (16/50) | 0.240 (12/50) |
| level-reweighted | 0.444 | 0.380 |
| self-consistency vote | 0.320 | |
| steps taken | | 23 of 40 |

Paired per problem: 11 kept, 5 lost, 1 gained, 33 wrong throughout.
Six discordant pairs, McNemar exact p = 0.219. **The decline is not
separable from noise.** Training produced no measurable gain; it did
not measurably hurt either.

Scored per sample instead, and split by what training did to each
problem (`report.json` beside this note, rebuilt by
`python -m mini_reasoning.report` on this directory), the picture
changes sign and shape:

| group | n | eval grader | boxed reward |
|---|---|---|---|
| all | 50 | 0.160 -> 0.186 (+0.026, p 0.212) | 0.140 -> 0.163 (+0.023, p 0.261) |
| stepped | 23 | 0.273 -> 0.329 (+0.056, p 0.155) | 0.236 -> 0.298 (+0.062, p 0.100) |
| flat | 17 | 0.025 -> 0.017 (-0.008, p 1.000) | 0.017 -> 0.017 (+0.000, p 1.000) |
| untouched | 10 | 0.129 -> 0.143 (+0.014, p 1.000) | 0.129 -> 0.100 (-0.029, p 0.745) |

The selected answer fell while the samples behind it rose, so the
headline's direction was never a finding. The rise sits where training
got gradient, on the reward it optimised, and nowhere else: suggestive
at p = 0.10, not established. The flat problems are out of reach rather
than unlucky, since 16 of the 17 also score zero boxed reward in the
independent baseline draw.

## What the numbers do not say

**The reward curve is confounded by problem order.** Mean reward falls
from 0.293 over the first 20 steps to 0.129 over the last 20, but
steps cycle the subset in file order and the file gets harder:
problems 1-20 average level 3.55 against 4.20 for 21-40, while reward
tracks level steeply (0.286 at level 3, 0.198 at level 4, 0.114 at
level 5). Read it as the ordering, not the trajectory.

**Half the compute bought no gradient.** 17 of 40 groups were flat,
almost always all-zero: boxed-only extraction lands on 56% of samples,
so on a hard problem all seven rollouts miss together and the step is
skipped by design.

**Nothing degenerated.** Mean sample length held at 990 characters
before and after, and the share stopping before the cap went 57% to
55%. Pre-clip gradient norms ran 54 to 845, median 133, against the
1.0 bound, so every step was clipped.

**The noise floor is wide.** The same code at bfloat16, same seed,
same subset, scored 0.400 / 0.514 with a 0.380 vote: eight points
apart from this run's baseline with no training involved at all. At
50 problems, a gain smaller than that is not resolvable, which is the
honest frame for the 0.320 to 0.240 move above.

## Why this note exists

`evaluate` did not stamp the model, dtype, seed or library versions
into what it returned, so the three JSONs beside this file say nothing
about what produced them and this note has to. Runs after the stamp
landed carry a `provenance` block and `--reuse-baseline` checks it,
which is also why these particular records are refused by that flag:
they predate it.
