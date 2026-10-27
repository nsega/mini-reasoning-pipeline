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
    LogitsProcessorList, TemperatureLogitsWarper, TopKLogitsWarper,
    TopPLogitsWarper,
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


@pytest.mark.parametrize("top_k", [None, 0])
def test_chain_leaves_out_top_k_when_the_model_has_none(top_k):
    kwargs = device.deterministic_sampling_kwargs(fake_model(top_k=top_k), 0.7, 0.95)
    assert [type(p) for p in kwargs["logits_processor"]] == [
        TemperatureLogitsWarper, device.DeterministicTopP,
    ]
