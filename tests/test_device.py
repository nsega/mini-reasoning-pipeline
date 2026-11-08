"""Contract tests for the device module: the CUDA check, the deterministic
switches, and the nucleus step that replaces transformers' in deterministic
mode.

No machine running these has CUDA. The process-wide switches are patched or
restored, and the sampling chain is checked against transformers' own
warpers on CPU, where both must give the same scores.

Written with Claude (AI-written); see the README's Provenance section.
"""
import os
from types import SimpleNamespace

import pytest
import torch
from transformers import (
    GenerationConfig, GenerationMixin, LogitsProcessorList, Qwen3Config, Qwen3ForCausalLM,
    TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper,
)

from mini_reasoning import device


@pytest.fixture(autouse=True)
def restore_cudnn():
    """Puts back the cuDNN switches `configure_deterministic` flips."""
    deterministic = torch.backends.cudnn.deterministic
    benchmark = torch.backends.cudnn.benchmark
    yield
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = benchmark


def fake_model(top_k):
    """Returns a stand-in for a model, with only the top-k its config holds."""
    return SimpleNamespace(generation_config=SimpleNamespace(top_k=top_k))


def test_require_cuda_raises_without_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="--device cuda"):
        device.require_cuda()


def test_require_cuda_passes_with_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert device.require_cuda() is None


def test_configure_deterministic_sets_every_switch(monkeypatch):
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    calls = []
    monkeypatch.setattr(torch, "use_deterministic_algorithms", calls.append)
    device.configure_deterministic()
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert calls == [True]
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False


@pytest.mark.parametrize(
    "existing, kept",
    [(":16:8", ":16:8"), (":4096:8", ":4096:8"), (":0:0", ":4096:8")],
)
def test_configure_deterministic_keeps_only_deterministic_values(
        monkeypatch, existing, kept):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", existing)
    monkeypatch.setattr(torch, "use_deterministic_algorithms", lambda mode: None)
    device.configure_deterministic()
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == kept


def test_deterministic_top_p_matches_transformers_on_cpu():
    torch.manual_seed(0)
    scores = torch.randn(4, 1000)
    expected = TopPLogitsWarper(top_p=0.95)(None, scores.clone())
    assert torch.equal(device.DeterministicTopP(0.95)(None, scores.clone()), expected)


def test_deterministic_top_p_honours_min_tokens_to_keep():
    torch.manual_seed(1)
    scores = torch.randn(3, 50)
    expected = TopPLogitsWarper(top_p=0.1, min_tokens_to_keep=3)(None, scores.clone())
    got = device.DeterministicTopP(0.1, min_tokens_to_keep=3)(None, scores.clone())
    assert torch.equal(got, expected)
    assert (got > -float("inf")).sum(dim=-1).min() >= 3


def test_chain_matches_transformers_order_on_cpu():
    torch.manual_seed(2)
    scores = torch.randn(4, 1000)
    expected = scores.clone()
    for warper in (
        TemperatureLogitsWarper(0.7), TopKLogitsWarper(20),
        TopPLogitsWarper(0.95),
    ):
        expected = warper(None, expected)
    kwargs = device.deterministic_sampling_kwargs(fake_model(top_k=20), 0.7, 0.95)
    got = kwargs["logits_processor"](None, scores.clone())
    assert torch.equal(got, expected)


def test_kwargs_switch_off_transformers_own_warpers():
    kwargs = device.deterministic_sampling_kwargs(fake_model(top_k=20), 0.7, 0.95)
    assert (kwargs["temperature"], kwargs["top_k"], kwargs["top_p"]) == (1.0, 0, 1.0)
    assert isinstance(kwargs["logits_processor"], LogitsProcessorList)
    assert [type(p) for p in kwargs["logits_processor"]] == [
        TemperatureLogitsWarper, TopKLogitsWarper, device.DeterministicTopP,
    ]


def test_unset_top_k_takes_transformers_global_default():
    default = GenerationConfig._get_default_generation_params()["top_k"]
    kwargs = device.deterministic_sampling_kwargs(fake_model(top_k=None), 0.7, 0.95)
    chain = list(kwargs["logits_processor"])
    assert [type(p) for p in chain] == [
        TemperatureLogitsWarper, TopKLogitsWarper, device.DeterministicTopP,
    ]
    assert chain[1].top_k == default


def test_zero_top_k_leaves_out_the_top_k_step():
    kwargs = device.deterministic_sampling_kwargs(fake_model(top_k=0), 0.7, 0.95)
    assert [type(p) for p in kwargs["logits_processor"]] == [
        TemperatureLogitsWarper, device.DeterministicTopP,
    ]


@pytest.mark.parametrize("top_k", [20, None])
def test_the_chain_reproduces_generates_own_sampling(monkeypatch, top_k):
    """A real generate, with and without the chain, draws the same tokens.

    The tiny Qwen3 is randomly initialised from a config, so nothing is
    downloaded. top_k=20 exercises an explicit top-k step. top_k=None is
    shaped like the real checkpoint's config (top_k unset, do_sample false,
    sampling switched on by the call), so generate falls back to
    transformers' global default of 50, and the vocabulary is larger than
    that so top-k actually removes tokens.
    A spy on GenerationMixin._get_logits_processor records the processor
    list generate actually uses, so the test also checks that each step
    appears once: transformers must not add its own on top of the chain.
    """
    used = []
    original = GenerationMixin._get_logits_processor

    def spy(self, *args, **kwargs):
        processors = original(self, *args, **kwargs)
        used.append([type(p) for p in processors])
        return processors

    monkeypatch.setattr(GenerationMixin, "_get_logits_processor", spy)
    torch.manual_seed(0)
    config = Qwen3Config(
        vocab_size=256, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
        head_dim=8, max_position_embeddings=64)
    model = Qwen3ForCausalLM(config).eval()
    model.generation_config.top_k = top_k
    model.generation_config.do_sample = False
    prompt = torch.tensor([[3, 7, 11]])

    torch.manual_seed(1)
    plain = model.generate(
        prompt, do_sample=True, temperature=0.7, top_p=0.95,
        max_new_tokens=8, pad_token_id=0)
    torch.manual_seed(1)
    chained = model.generate(
        prompt, do_sample=True, max_new_tokens=8, pad_token_id=0,
        **device.deterministic_sampling_kwargs(model, 0.7, 0.95))

    assert torch.equal(plain, chained)

    wanted = (TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper,
              device.DeterministicTopP)
    plain_used, chained_used = (
        [t for t in types if t in wanted] for types in used)
    assert plain_used == [
        TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper]
    assert chained_used == [
        TemperatureLogitsWarper, TopKLogitsWarper, device.DeterministicTopP]


def test_shown_field_names_cpu_and_fast_mode_for_missing_fields():
    assert device.shown_field("device", None) == "cpu"
    assert device.shown_field("deterministic", None) is False
    assert device.shown_field("seed", None) is None
    assert device.shown_field("device", "cuda") == "cuda"
    assert device.shown_field("deterministic", True) is True
