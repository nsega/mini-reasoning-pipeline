# CUDA Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the pipeline run on an NVIDIA GPU when asked, in a fast mode and an opt-in deterministic mode whose runs reproduce each other, while a run that does not ask stays on CPU and writes byte-identical records.

**Architecture:** A new `mini_reasoning/device.py` owns everything CUDA-specific: the availability check, the deterministic switches, and a deterministic nucleus (top-p) step with the sampling chain around it. The samplers in `evaluate.py` and `trainer.py` gain a `deterministic` keyword that swaps generate's sampling arguments for that chain, and the trainer moves its rewards to the model's device. `evaluate.provenance`, the baseline reuse check and the checkpoint identity record `device` and `deterministic`, absent on CPU. `run_pipeline.py` adds `--device` and `--deterministic` and wires them in.

**Tech Stack:** Python 3.13, PyTorch 2.13 (CUDA 13.0 build on Linux), transformers 5.15, safetensors, pytest, uv.

**Spec:** `docs/specs/2026-10-26-cuda-support-design.md`

## Global Constraints

- Python follows the Google Python Style Guide: every new or changed function carries a docstring, with Args, Returns and Raises where they apply.
- No em-dashes (—) in code, docstrings, docs or commit messages. En-dashes in numeric ranges and arrows are fine.
- Commits follow Conventional Commits 1.0.0 and end with a `Co-Authored-By: Claude <model> <noreply@anthropic.com>` trailer naming the model that wrote them. No session links, no "Generated with" lines.
- CPU is the default and does not move. Without `--device cuda`, every command makes the same calls with the same arguments as before, and every existing test passes unchanged.
- No record format changes on CPU. `provenance` on CPU has exactly today's keys: `model`, `dtype`, `seed`, `n_samples`, `fallback` and the library versions.
- Flags: `--device {cpu,cuda}`, default `cpu`. `--deterministic`, a switch that needs `--device cuda`.
- `--device cuda` without CUDA stops before the model loads. There is no fallback to CPU.
- Deterministic mode sets `CUBLAS_WORKSPACE_CONFIG=:4096:8` (keeping an existing `:4096:8` or `:16:8`), `torch.use_deterministic_algorithms(True)`, `torch.backends.cudnn.deterministic = True` and `torch.backends.cudnn.benchmark = False`, before the first CUDA call.
- Both modes keep float32 and PyTorch's default matmul precision. No TF32.
- On CUDA, `provenance` adds `device` (`"cuda"`), `deterministic` (bool), `gpu` (the device name) and `cuda` (`torch.version.cuda`). `device` and `deterministic` are compared on reuse and on resume. `gpu` and `cuda` are not compared by the pipeline.
- Tests are written first and watched failing. Run the suite with `uv run --group dev pytest -q`. It holds 281 tests before this plan, and every one still passes after it.
- No machine available to this work has CUDA. Nothing here runs a GPU. GPU behaviour is checked in kueue-training-lab's first GKE session, and the PR merges only after it.

## Review Focus

- A `--stage validate --device cuda` run over a baseline a CPU run wrote: the reuse check must refuse it, naming `device`. Pinned in Task 3.
- A `--deterministic` run resuming a checkpoint a fast run wrote, or the reverse: the resume must refuse it, naming `deterministic`. Pinned in Task 3.
- A model whose `generation_config.top_k` is `None` or `0`: the deterministic chain must leave top-k out, as transformers does, not apply a top-k of 0. Pinned in Task 1.
- A reader who already exported `CUBLAS_WORKSPACE_CONFIG=:16:8`: deterministic mode keeps it, since it is also a deterministic setting, and replaces any other value. Pinned in Task 1.
- A fake or unusual model with no parameters reaching `provenance`: it is treated as CPU and the stamp has today's keys, never raising. Pinned in Task 3.

---

### Task 1: The device module

**Files:**
- Create: `mini_reasoning/device.py`
- Test: `tests/test_device.py`

**Interfaces:**
- Produces:
  - `require_cuda() -> None`, raising `RuntimeError` when `torch.cuda.is_available()` is false.
  - `configure_deterministic() -> None`, applying the four deterministic settings in Global Constraints.
  - `class DeterministicTopP(transformers.LogitsProcessor)`, `__init__(self, top_p: float, min_tokens_to_keep: int = 1)`, `__call__(self, input_ids, scores) -> torch.Tensor`.
  - `deterministic_sampling_kwargs(model, temperature: float, top_p: float) -> dict`, the keyword arguments a sampler passes to `model.generate` in deterministic mode.

- [ ] **Step 1: Write the failing tests** in `tests/test_device.py`:

```python
def test_require_cuda_raises_without_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="--device cuda"):
        device.require_cuda()

def test_configure_deterministic_sets_every_switch(monkeypatch):
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    calls = []
    monkeypatch.setattr(torch, "use_deterministic_algorithms", calls.append)
    device.configure_deterministic()
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert calls == [True]
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False

@pytest.mark.parametrize("existing, kept", [(":16:8", ":16:8"), (":4096:8", ":4096:8"), (":0:0", ":4096:8")])
def test_configure_deterministic_keeps_only_deterministic_values(monkeypatch, existing, kept): ...

def test_deterministic_top_p_matches_transformers_on_cpu():
    torch.manual_seed(0)
    scores = torch.randn(4, 1000)
    expected = TopPLogitsWarper(top_p=0.95)(None, scores.clone())
    assert torch.equal(device.DeterministicTopP(0.95)(None, scores.clone()), expected)

def test_chain_matches_transformers_order_on_cpu():
    # temperature 0.7, top_k 20, top_p 0.95, applied by transformers' warpers in its order,
    # against the chain deterministic_sampling_kwargs returns: torch.equal on random scores.

def test_kwargs_switch_off_transformers_own_warpers():
    kwargs = device.deterministic_sampling_kwargs(fake_model(top_k=20), 0.7, 0.95)
    assert (kwargs["temperature"], kwargs["top_k"], kwargs["top_p"]) == (1.0, 0, 1.0)
    assert [type(p) for p in kwargs["logits_processor"]] == [TemperatureLogitsWarper, TopKLogitsWarper, device.DeterministicTopP]

@pytest.mark.parametrize("top_k", [None, 0])
def test_chain_leaves_out_top_k_when_the_model_has_none(top_k):
    kwargs = device.deterministic_sampling_kwargs(fake_model(top_k=top_k), 0.7, 0.95)
    assert [type(p) for p in kwargs["logits_processor"]] == [TemperatureLogitsWarper, device.DeterministicTopP]
```

`fake_model(top_k)` is a `SimpleNamespace(generation_config=SimpleNamespace(top_k=top_k))`. `configure_deterministic` changes process-wide state: restore `cudnn.deterministic` and `cudnn.benchmark` in a fixture.

- [ ] **Step 2: Run the tests and watch them fail**

Run: `uv run --group dev pytest tests/test_device.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'mini_reasoning.device'`

- [ ] **Step 3: Implement `mini_reasoning/device.py`**

`DeterministicTopP.__call__` mirrors `transformers.TopPLogitsWarper.__call__` line for line, with one change: the cumulative sum runs on CPU, where it is deterministic, and the mask returns to the scores' device. `torch.use_deterministic_algorithms(True)` raises on `torch.cumsum` of a floating-point CUDA tensor.

```python
sorted_logits, sorted_indices = torch.sort(scores, descending=False)
cumulative_probs = sorted_logits.softmax(dim=-1).cpu().cumsum(dim=-1).to(scores.device)
sorted_indices_to_remove = cumulative_probs <= (1 - self.top_p)
sorted_indices_to_remove[..., -self.min_tokens_to_keep:] = 0
indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
return scores.masked_fill(indices_to_remove, -float("inf"))
```

`deterministic_sampling_kwargs` returns `temperature=1.0`, `top_k=0` and `top_p=1.0`, which keep transformers from adding its own temperature, top-k and top-p steps. It also returns `logits_processor`, a `LogitsProcessorList` of `TemperatureLogitsWarper(temperature)`, `TopKLogitsWarper(model.generation_config.top_k)` when that is neither `None` nor `0`, and `DeterministicTopP(top_p)`. Transformers merges custom processors in before it appends its sampling warpers (`GenerationMixin._get_logits_processor`), so the chain runs where transformers' own three would. The module docstring says so, and that it is AI-written.

- [ ] **Step 4: Run the tests and watch them pass**

Run: `uv run --group dev pytest tests/test_device.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add mini_reasoning/device.py tests/test_device.py
git commit -m "feat(device): add CUDA checks and a deterministic sampling chain"
```

### Task 2: Samplers and rewards on the model's device

**Files:**
- Modify: `mini_reasoning/evaluate.py` (`sample_solutions`, the `generate` call near line 90)
- Modify: `mini_reasoning/trainer.py` (`sample_rollouts`, the `generate` call near line 141, and the `rollout_rewards` call near line 222)
- Test: `tests/test_evaluate.py`, `tests/test_trainer.py`

**Interfaces:**
- Consumes: `device.deterministic_sampling_kwargs(model, temperature, top_p)` from Task 1.
- Produces: `sample_solutions(..., deterministic: bool = False)` and `sample_rollouts(..., deterministic: bool = False)`, keyword-only additions after the existing parameters. Task 4 binds them with `functools.partial`.

- [ ] **Step 1: Write the failing tests**

In each test file, a fake model records the keyword arguments `generate` receives:

```python
def test_sample_solutions_calls_generate_as_before_by_default():
    # captured kwargs hold temperature=0.7, top_p=0.95, and no logits_processor

def test_sample_solutions_deterministic_passes_the_chain():
    # with deterministic=True: temperature == 1.0, top_k == 0, top_p == 1.0,
    # and logits_processor ends with a DeterministicTopP whose top_p == 0.95

def test_sample_rollouts_calls_generate_as_before_by_default(): ...
def test_sample_rollouts_deterministic_passes_the_chain(): ...

def test_rewards_follow_the_logprobs_device(monkeypatch):
    # grpo_loss, patched to record its inputs, receives rewards (and the
    # advantages built from them) on the same device as the logprobs
```

Reuse the existing fakes in each test file where they already capture `generate`'s arguments. Fake models need not have a `device` attribute, which is why the rewards follow the logprobs, not the model.

- [ ] **Step 2: Run them and watch them fail**

Run: `uv run --group dev pytest tests/test_evaluate.py tests/test_trainer.py -q`
Expected: the new tests FAIL with `TypeError: ... unexpected keyword argument 'deterministic'`, and every existing test passes.

- [ ] **Step 3: Implement**

In both samplers: when `deterministic` is false, the `generate` call is unchanged, character for character. When it is true, `temperature=...` and `top_p=...` are replaced by `**deterministic_sampling_kwargs(model, temperature, top_p)`. In the trainer's loop, the rewards move to the device of the logprobs computed for the same rollouts, `rewards.to(logprobs.device)`, before the advantages are built. On CPU, that is a no-op.

- [ ] **Step 4: Run the whole suite and watch it pass**

Run: `uv run --group dev pytest -q`
Expected: PASS, the 281 existing tests unchanged.

- [ ] **Step 5: Commit**

```bash
git add mini_reasoning/evaluate.py mini_reasoning/trainer.py tests/test_evaluate.py tests/test_trainer.py
git commit -m "feat(sampling): sample deterministically on request, rewards on the model's device"
```

### Task 3: Device in records, reuse checks and checkpoints

**Files:**
- Modify: `mini_reasoning/evaluate.py` (`provenance`, near line 277)
- Modify: `run_pipeline.py` (`COMPARED`, line 202, and `run_identity`, line 360)
- Modify: `mini_reasoning/checkpoint.py` (`IDENTITY`, line 63, and `resume_from`'s `torch.load`, near line 298)
- Test: `tests/test_evaluate.py`, `tests/test_run_pipeline.py`, `tests/test_checkpoint.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `provenance(model, *, seed, n_samples, fallback) -> dict`, signature unchanged, adding the CUDA fields when the model's parameters are on CUDA. `COMPARED` and `IDENTITY` each gain `"device"` and `"deterministic"`. `run_identity(args, pool, training_seed) -> dict` adds `"device": "cuda"` and `"deterministic": args.deterministic` when `args.device == "cuda"`, and nothing on CPU.

- [ ] **Step 1: Write the failing tests**

```python
def test_provenance_on_cpu_has_exactly_todays_keys():
    assert set(provenance(cpu_model, seed=42, n_samples=7, fallback="number")) == {
        "model", "dtype", "seed", "n_samples", "fallback", "torch", "transformers"}

def test_provenance_on_a_model_without_parameters_is_cpu(): ...

def test_provenance_on_cuda_adds_the_device_fields(monkeypatch):
    # a fake model whose parameters report device type "cuda", with
    # torch.cuda.get_device_name, torch.version.cuda and
    # torch.are_deterministic_algorithms_enabled patched:
    # stamp["device"] == "cuda", stamp["deterministic"] is True,
    # stamp["gpu"] == "NVIDIA L4", stamp["cuda"] == "13.0"

def test_reuse_refuses_a_baseline_from_another_device():
    # load_baseline with expected device "cuda" over a stored CPU baseline:
    # ValueError matching "device=None" and "device='cuda'"

def test_reuse_refuses_a_baseline_from_the_other_mode(): ...

def test_run_identity_on_cpu_has_exactly_todays_keys(): ...

def test_resume_refuses_a_checkpoint_from_the_other_mode(tmp_path):
    # a checkpoint saved with identity deterministic=False, read with
    # deterministic=True: read_latest raises ValueError naming "deterministic"

def test_resume_loads_adam_state_onto_the_models_device(monkeypatch):
    # torch.load is called with map_location equal to the model's device
```

Use the checkpoint tests' existing fixtures for saving and reading. The library-version keys come from `library_versions()`, so assert against its keys rather than the literal names if they differ.

- [ ] **Step 2: Run them and watch them fail**

Run: `uv run --group dev pytest tests/test_evaluate.py tests/test_run_pipeline.py tests/test_checkpoint.py -q`
Expected: the new tests FAIL, and the existing ones pass.

- [ ] **Step 3: Implement**

`provenance` finds the model's device from its first parameter, with `None` for a model that has none. Only for `cuda` does it add the four fields, `deterministic` coming from `torch.are_deterministic_algorithms_enabled()`. `COMPARED` and `IDENTITY` compare with `.get`, so a CPU stamp's missing fields equal each other, and old CPU records and checkpoints still match. `resume_from` passes `map_location` as the model's parameter device to `torch.load`.

- [ ] **Step 4: Run the whole suite and watch it pass**

Run: `uv run --group dev pytest -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add mini_reasoning/evaluate.py mini_reasoning/checkpoint.py run_pipeline.py tests/
git commit -m "feat(records): record the device and mode, and refuse to mix them"
```

### Task 4: The entry point's flags

**Files:**
- Modify: `run_pipeline.py` (`build_parser`, `check_flags`, `run`, and the module docstring)
- Test: `tests/test_run_pipeline.py`

**Interfaces:**
- Consumes: `device.require_cuda`, `device.configure_deterministic` (Task 1), the samplers' `deterministic` keyword (Task 2), `run_identity`'s device fields (Task 3).
- Produces: `args.device` (`"cpu"` or `"cuda"`) and `args.deterministic` (bool).

- [ ] **Step 1: Write the failing tests**

```python
def test_device_defaults_to_cpu():
    args = build_parser().parse_args([])
    assert (args.device, args.deterministic) == ("cpu", False)

def test_deterministic_needs_cuda():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--deterministic"])
    # stderr names "--device cuda" and says a CPU run is already deterministic

def test_cuda_without_a_gpu_stops_before_the_model_loads(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="--device cuda"):
        run(parse(["--device", "cuda", ...]), load_model=forbidden_loader, sampler=canned)

def test_deterministic_configures_before_the_model_loads(monkeypatch):
    # configure_deterministic and require_cuda are patched to record calls;
    # the loader records when it runs; the order is configure, require, load

def test_deterministic_binds_the_samplers(monkeypatch):
    # with --device cuda --deterministic and CUDA patched available, the
    # sampler evaluate receives is called with deterministic=True

def test_cpu_run_calls_the_loader_with_the_name_only():
    # the CPU path still calls load_model(args.model) and never moves the model
```

- [ ] **Step 2: Run them and watch them fail**

Run: `uv run --group dev pytest tests/test_run_pipeline.py -q`
Expected: the new tests FAIL, and the existing ones pass.

- [ ] **Step 3: Implement**

`build_parser` adds the two flags. `check_flags` refuses `--deterministic` without `--device cuda` through `parser.error`. In `run`, before `load_model` is called and after the existing checks that can refuse a run: `configure_deterministic()` if `args.deterministic`, then `require_cuda()` if `args.device == "cuda"`. After loading, `model = model.to("cuda")` on the CUDA path only. When `args.deterministic`, `sampler` and `rollout` are wrapped with `functools.partial(..., deterministic=True)`. The module docstring gains a paragraph on the two flags.

The spec has `_load_model` gain the device. This plan moves the model in `run` instead, so that `run` keeps calling `load_model(args.model)` with one argument, as every test's injected loader expects. The effect is the same: the model loads in float32 on CPU and moves to CUDA only on the CUDA path.

- [ ] **Step 4: Run the whole suite and watch it pass**

Run: `uv run --group dev pytest -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add run_pipeline.py tests/test_run_pipeline.py
git commit -m "feat(pipeline): add --device cuda and --deterministic"
```

### Task 5: README, and the CPU run that proves nothing moved

**Files:**
- Modify: `README.md` (a new section after "Checkpoints and stages", and the Provenance section)

- [ ] **Step 1: Write the README section**

As the spec's "README" section lists: how to run on a GPU and switch on deterministic mode, that CPU and CUDA records never match, that deterministic runs reproduce only on the same GPU type, driver, CUDA and PyTorch, and that fast mode's measured gap and both modes' L4 timings are "to come, from kueue-training-lab's runs on an L4". It must not state a number that has not been measured. Add `device.py` and `tests/test_device.py` to Provenance's AI-written list.

- [ ] **Step 2: Run a full CPU pipeline on the M4 Pro, unattended, about 3 hours**

Run: `uv run python run_pipeline.py --out-dir results/cuda-support-cpu-check`
Then compare it with the committed run:
`diff <(jq -S 'del(.seed)' docs/runs/2026-09-28-dynamic-sampling/training.json) <(jq -S 'del(.seed)' results/cuda-support-cpu-check/training.json)`, and `cmp` the other three files against the committed run's.
Expected: no differences, apart from `training.json`'s `seed` field, which the committed run predates. Any other difference means the CPU path moved, and that is fixed before going on.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs(readme): document opt-in CUDA support"
```

### Task 6: Hand the branch to the lab

- [ ] **Step 1: Push the branch and open a draft PR**

```bash
git push -u origin cuda-support
gh pr create --draft --title "feat: opt-in CUDA support with a deterministic mode" --body "<summary, the spec link, and: Do not merge until kueue-training-lab's GKE smoke run has passed on this branch's head, per the spec's 'Verification on a GPU'.>"
```

The body ends with no session links and no "Generated with" line.

- [ ] **Step 2: Record the branch's head commit for the lab**

Run: `git rev-parse HEAD`
Expected: a 40-character SHA, which kueue-training-lab pins as `PIPELINE_SHA` for its amd64 image and its smoke run. The PR stays a draft until that smoke run passes.
