# CUDA support

Status: design discussed in conversation and checked by a builder's read against the code, then approved by the author, 2026-10-26. Next: an implementation plan.

## Problem

The pipeline runs on CPU only. `_load_model` loads Qwen3 in float32 and never moves it to a device, so a machine with an NVIDIA GPU runs everything on its CPU. Three hours on an M4 Pro, four on kind.

[kueue-training-lab](https://github.com/nsega/kueue-training-lab) needs a GPU path for its Stage 2, which runs this pipeline under Kueue on GKE with NVIDIA L4 GPUs and repeats its eviction experiment there (EXP-E). The lab judges every experiment by this repository's standard: records equal a reference's, record for record. On a GPU that holds only if the run is deterministic, which CUDA is not by default.

Readers of this repository with an NVIDIA GPU want the same thing for a different reason: a run in an hour or two instead of three.

Most of the code is already device-agnostic. The evaluator and the trainer move their inputs to `model.device`, and `torch.manual_seed`, which seeds every eval problem and every training draw, seeds CUDA's generator as well. What is missing is choosing the device, putting the model on it, making CUDA deterministic when asked, and recording all of it so that records and checkpoints from different devices are never mistaken for each other. Two places are not device-agnostic yet, and section 1 fixes both: the trainer builds its rewards on the CPU, and the samplers' nucleus (top-p) step cannot run in deterministic mode.

## Decisions

The author's decisions, made in conversation on 2026-10-26:

| question | decision | rejected, and why |
|---|---|---|
| Who it is for | A public feature, documented in the README, opt-in like resume | The lab's private means to an end: a GPU path is useful to every reader with an NVIDIA GPU |
| Which GPUs | CUDA only | MPS on Apple Silicon: the trainer has hit an 18.7 GB out-of-memory error there before, and nothing in the lab needs it |
| The default | CPU, unchanged. A run that does not ask for CUDA takes today's path and writes byte-identical records | CUDA when present: it would silently change every existing reference on any machine with a GPU |
| How exact a CUDA run is | Two modes. Fast is CUDA's default, and its records come close to a rerun's without matching. Deterministic is an opt-in switch, and two runs on the same GPU type write identical records | Deterministic only: readers would pay its cost for a guarantee they may not need. Fast only: the lab would have no pass test for Stage 2 |
| What the lab uses | Deterministic mode, for every GPU run | A tolerance-based compare: it would weaken the one rule every experiment so far has met |
| What the README promises for fast mode | That its accuracy figures land close to a deterministic run's, with the gap measured, not assumed | No promise beyond speed: readers would have no way to judge a fast run's numbers |
| Where it lives | `main`, opt-in, as resume does | A lab branch: every fix would have to cross between branches, as the resume design found |
| When it merges | Only after a smoke run on a real GPU, from its branch, as the first step of the lab's first GKE session | Merging once the CPU checks pass and fixing forward: no machine available to this work has CUDA, so the first GPU execution would land on `main` untested, and reading the code already found one bug that only a GPU would hit |

## Design

### 1. Device and modes

Two new flags on the entry point:

- `--device {cpu,cuda}`, default `cpu`.
- `--deterministic`, a switch that needs `--device cuda`.

`--device cuda` on a machine where `torch.cuda.is_available()` is false stops with an error before the model loads. There is no fallback to CPU: a run that silently fell back would write CPU records under a GPU run's name.

`--deterministic` without `--device cuda` is refused by `check_flags`, with a message that a CPU run is already deterministic.

Fast mode keeps PyTorch's CUDA defaults. Deterministic mode adds, before the first CUDA call:

- `CUBLAS_WORKSPACE_CONFIG=:4096:8`, which cuBLAS reads when it initializes, so it must be set in the process environment before any CUDA work.
- `torch.use_deterministic_algorithms(True)`, which picks deterministic kernels and raises on an operation that has none. A raise there is a finding, not something to work around by turning the switch off.
- `torch.backends.cudnn.deterministic = True` and `torch.backends.cudnn.benchmark = False`.

Both modes keep float32 and PyTorch's default matmul precision, with no TF32. That way the measured gap between the two modes is the gap determinism alone makes, not one mixed with a change of precision.

`_load_model` gains the device and moves the model there after loading in float32. On CPU that move does nothing, so the CPU path is unchanged.

Two code changes are required before any CUDA run works, both found by reading the code against the installed libraries:

- **Rewards on the model's device.** `trainer.rollout_rewards` builds its tensor with `torch.tensor(...)` and no device, so it stays on the CPU, and `grpo_loss` would multiply CPU advantages by CUDA log-probabilities on the first draw that is not flat. The rewards move to the model's device. On CPU that is a no-op.
- **A deterministic nucleus step.** Both samplers sample with `top_p=0.95`. transformers' `TopPLogitsWarper` computes `sorted_logits.softmax(-1).cumsum(-1)`, and `torch.use_deterministic_algorithms(True)` raises on `torch.cumsum` of a floating-point CUDA tensor, so deterministic mode would fail on its first sample. Deterministic mode replaces that step with a logits processor that computes the same nucleus with a deterministic cumulative sum. CPU and fast mode keep transformers' own step, so CPU records stay byte-identical. The measured gap between fast and deterministic mode then covers determinism and this one replacement, and the README says so.

### 2. Records

`evaluate.provenance` stamps every eval's records with what produced them. A CUDA run adds:

- `device`: `cuda`.
- `deterministic`: true or false.
- `gpu`: the device name, for example `NVIDIA L4`, and `cuda`: `torch.version.cuda`.

`device` and `deterministic` are identity: a reuse check compares them, so a CUDA run never reuses a CPU baseline, and a deterministic run never reuses a fast one. `gpu` and `cuda` are recorded like the library versions are, for whoever has to explain a strange result, and the pipeline's own reuse check does not compare them. The lab's `make compare` checks an eval file's whole `provenance`, so there they do take part. That is harmless for runs on the same GPU type, which is the only kind the lab compares.

A CPU run adds none of these fields. A missing `device` means CPU. That is what keeps CPU records byte-identical: adding `"device": "cpu"` would make every existing CPU run differ from a new one in its provenance.

### 3. Checkpoints

The run identity a checkpoint must share with a resuming run (`run_identity`, and `checkpoint.IDENTITY`) gains `device` and `deterministic` on CUDA, under the same rule: absent on CPU. A resume refuses a checkpoint from the other device or the other mode, naming the field that differs, as it already does for a checkpoint from another run.

Weights load through `safetensors.torch.load_model` into a model already on the run's device. Adam's state loads with `torch.load`, which needs `map_location` set to the run's device, since a checkpoint saved on CUDA holds CUDA tensors.

### 4. README

A new section beside "Checkpoints and stages":

- How to run on a GPU, and how to switch on deterministic mode.
- That CPU and CUDA records never match each other, and that a deterministic run reproduces itself only on the same GPU type, driver, CUDA and PyTorch versions.
- Fast mode's measured gap against a deterministic run, from the lab's runs on an L4: the accuracy figures and the per-record differences.
- Measured timings for both modes on an L4, beside the M4 Pro's CPU timings.

Until the lab has measured them, the section says the numbers are to come rather than estimating them.

## Testing

No machine available to this work has an NVIDIA GPU, and the repository's CI has none either, so the tests cover everything that does not need one:

- Every existing test passes unchanged. The CPU path is the default and must not move.
- `check_flags`: `--deterministic` without `--device cuda` is refused, and `--device cuda` with no CUDA available stops before the model loads, with `torch.cuda.is_available` patched.
- `provenance` on CPU has exactly today's keys. With a model reporting a CUDA device, it adds `device`, `deterministic`, `gpu` and `cuda`.
- A reuse check refuses a baseline whose `device` or `deterministic` differs, and treats a missing `device` as CPU.
- `run_identity` on CPU has exactly today's keys, and a resume refuses a checkpoint from the other device or mode.
- The deterministic setup sets the environment variable and calls the PyTorch switches, checked by patching them.

One full CPU run after the change, on the M4 Pro, must reproduce `docs/runs/2026-09-28-dynamic-sampling` record for record, apart from `training.json`'s `seed` field, which that run predates. The lab's EXP-0 reference met the same bar on 2026-10-13.

## Verification on a GPU

The branch is pushed but not merged until a GPU has run it. The lab's first GKE session, on one L4, starts from the branch's commit:

0. A smoke run on the branch, in both modes: the lab's short configuration (a few problems, 2 samples, a few updates, a save every update), once with `--device cuda` and once with `--device cuda --deterministic`, each followed by a resume. It must finish without an error, report a CUDA device in its provenance, and resume. A failure is fixed on the branch and the smoke run repeated. Only then does the PR merge, and the lab pins `PIPELINE_SHA` to the merge commit and rebuilds its image for the steps below.

Then, on the merged commit, through the lab's own `make compare`:

1. Two clean runs with `--device cuda --deterministic`. They must compare identical. If they do not, deterministic mode does not hold on this hardware, and that is a finding to fix here before the lab's EXP-E relies on it.
2. One clean run in fast mode, compared with the first deterministic run. It is expected to differ in records. The README's measured gap comes from this pair: the report's accuracy figures side by side and the count of differing records.
3. Each run's wall clock, for the README's timings.

EXP-E itself, an evicted deterministic GPU run resumed from its checkpoint and compared with a clean one, is the lab's experiment, not part of this design.

## The lab's side

This design changes nothing in the lab, but it sets one requirement on the lab's GKE setup. `uv.lock` resolves torch 2.13.0 with CUDA 13.0 builds on Linux, and the lock stays as it is: changing it would change the lab's kind image and with it the Linux CPU reference its Stage 1 experiments matched. So the L4 nodes need an NVIDIA driver new enough for CUDA 13.0. The lab checks the GKE version's driver against NVIDIA's CUDA compatibility table before it creates a node pool, and a smoke run's `torch.cuda.is_available()` confirms it.

## Out of scope

- MPS, and any other accelerator.
- More than one GPU, or distributed training.
- bfloat16, mixed precision or TF32. The pipeline stays float32 on every device.
- A CPU-only torch build for the lab's kind image.
- The lab's image, manifests and GKE setup.

## Costs accepted

- Deterministic mode is slower than fast mode by an amount nobody has measured. The lab's runs measure it.
- A fast run's records are not reproducible, by design, and the README says so.
- A deterministic run reproduces itself only on matching hardware and software, so its reference is tied to a GPU type, as the lab's CPU references are tied to their platform.
- `torch.use_deterministic_algorithms(True)` may raise on an operation in the model or the trainer that has no deterministic CUDA kernel. If it does, that operation needs a deterministic alternative here, found in the first GPU session.

## Calls made in writing this design

These were not discussed and are the writer's choices, open to change:

- The flag names, `--device` and `--deterministic`.
- `--device cuda` with no GPU stops instead of falling back to CPU.
- Fast mode keeps PyTorch's CUDA defaults, with no TF32.
- `gpu` and `cuda` in provenance are recorded but not compared.
- The README's GPU numbers wait for the lab's measurements instead of being estimated.
- The deterministic nucleus step replaces transformers' only in deterministic mode. CPU and fast mode keep transformers' own.
- The rewards move to the model's device in every mode, a no-op on CPU.
- `uv.lock` keeps its CUDA 13.0 torch build, and the driver becomes a requirement on the lab's GKE nodes.
