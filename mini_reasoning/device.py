"""Choosing CUDA and making it deterministic, for runs that ask for it.

The pipeline runs on CPU unless a run passes `--device cuda`. This module
holds what that needs that does not belong in the model loader:

- require_cuda stops a CUDA run on a machine with no GPU. There is no
  fallback to CPU, which would write CPU records under a GPU run's name.
- configure_deterministic applies the four switches that make CUDA
  reproduce itself: the cuBLAS workspace setting, PyTorch's deterministic
  algorithms, and cuDNN's deterministic and non-benchmark modes.
- DeterministicTopP and deterministic_sampling_kwargs replace the nucleus
  (top-p) step of sampling. transformers' TopPLogitsWarper takes a
  cumulative sum of a floating-point tensor, and
  torch.use_deterministic_algorithms(True) raises on torch.cumsum of such
  a tensor on CUDA, so deterministic mode would fail on its first sample.
  DeterministicTopP computes the same nucleus with the cumulative sum on
  CPU, where it is deterministic.

transformers merges the processors a caller passes to generate in before
it appends its own sampling warpers (GenerationMixin._get_logits_processor).
So deterministic_sampling_kwargs switches off transformers' temperature,
top-k and top-p steps and passes the same three as a logits processor
list, and the chain runs where transformers' own would.

CPU runs and CUDA fast mode never call into this chain, so their records
stay as they were.

Written with Claude (AI-written); see the README's Provenance section.
"""
import os

import torch
from transformers import (
    GenerationConfig, LogitsProcessor, LogitsProcessorList, TemperatureLogitsWarper,
    TopKLogitsWarper,
)

CUBLAS_ENV = "CUBLAS_WORKSPACE_CONFIG"
CUBLAS_DEFAULT = ":4096:8"
# The two workspace settings cuBLAS documents as deterministic.
CUBLAS_DETERMINISTIC = (":4096:8", ":16:8")


def require_cuda() -> None:
    """Stops a CUDA run on a machine that has no usable GPU.

    Raises:
        RuntimeError: If torch.cuda.is_available() is false.
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            "--device cuda needs a CUDA GPU, and torch.cuda.is_available() "
            "is false on this machine. There is no fallback to CPU: it "
            "would write CPU records under a GPU run's name.")


def configure_deterministic() -> None:
    """Switches PyTorch's CUDA work to deterministic kernels.

    Sets CUBLAS_WORKSPACE_CONFIG to ":4096:8" unless it already holds one
    of cuBLAS's deterministic values, turns on
    torch.use_deterministic_algorithms, sets cudnn.deterministic and
    clears cudnn.benchmark. cuBLAS reads its variable when it
    initializes, so call this before the first CUDA call. These are
    process-wide settings.

    Once the switch is on, an operation with no deterministic CUDA kernel
    raises RuntimeError when it runs. That is a finding, not something to
    turn the switch off for.
    """
    if os.environ.get(CUBLAS_ENV) not in CUBLAS_DETERMINISTIC:
        os.environ[CUBLAS_ENV] = CUBLAS_DEFAULT
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class DeterministicTopP(LogitsProcessor):
    """transformers' nucleus step with a cumulative sum that CUDA allows.

    Mirrors transformers.TopPLogitsWarper.__call__ line for line, with one
    change: the cumulative sum runs on CPU, where it is deterministic, and
    the result returns to the scores' device. On CPU it gives the same
    scores as the warper.
    """

    def __init__(self, top_p: float, min_tokens_to_keep: int = 1):
        """Builds the step.

        Args:
            top_p: The cumulative probability mass to keep, from 0 to 1.
            min_tokens_to_keep: The fewest tokens to leave unmasked.

        Raises:
            ValueError: If top_p is outside 0 to 1, or min_tokens_to_keep
                is not a positive integer.
        """
        top_p = float(top_p)
        if top_p < 0 or top_p > 1.0:
            raise ValueError(
                f"`top_p` has to be a float between 0 and 1, but is {top_p}")
        if not isinstance(min_tokens_to_keep, int) or min_tokens_to_keep < 1:
            raise ValueError(
                "`min_tokens_to_keep` has to be a positive integer, but is "
                f"{min_tokens_to_keep}")
        self.top_p = top_p
        self.min_tokens_to_keep = min_tokens_to_keep

    def __call__(self, input_ids: torch.LongTensor,
                 scores: torch.FloatTensor) -> torch.FloatTensor:
        """Masks the tokens outside the top-p nucleus to -inf.

        Args:
            input_ids: The tokens so far. Unused, kept for the interface.
            scores: Logits of shape (batch, vocab).

        Returns:
            The scores with every token outside the nucleus set to -inf.
        """
        sorted_logits, sorted_indices = torch.sort(scores, descending=False)
        cumulative_probs = (
            sorted_logits.softmax(dim=-1).cpu().cumsum(dim=-1).to(scores.device))
        sorted_indices_to_remove = cumulative_probs <= (1 - self.top_p)
        sorted_indices_to_remove[..., -self.min_tokens_to_keep:] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove)
        return scores.masked_fill(indices_to_remove, -float("inf"))


def shown_field(field: str, value):
    """Renders a stamp or identity field for a refusal message.

    A CPU run adds no device or deterministic field, so a missing one means
    CPU and the fast mode. The message names that, not None. Stored records
    are unchanged.

    Args:
        field: The field's name.
        value: The stored value, None when the field is missing.

    Returns:
        The value to show: 'cpu' for a missing device, False for a missing
        deterministic, otherwise value itself.
    """
    if value is None:
        return {"device": "cpu", "deterministic": False}.get(field, value)
    return value


def deterministic_sampling_kwargs(model, temperature: float,
                                  top_p: float) -> dict:
    """Builds the generate() arguments for sampling in deterministic mode.

    Args:
        model: The model to sample from. Its generation_config.top_k sets
            the chain's top-k step. An unset top_k takes transformers' global
            default, as generate does, and 0 means no top-k step.
        temperature: The sampling temperature.
        top_p: The nucleus mass.

    Returns:
        Keyword arguments for model.generate. temperature=1.0, top_k=0 and
        top_p=1.0 keep transformers from adding its own steps, and
        logits_processor holds the temperature, top-k (when the model has
        one) and DeterministicTopP steps, in transformers' order.
    """
    processors = LogitsProcessorList([TemperatureLogitsWarper(temperature)])
    top_k = model.generation_config.top_k
    if top_k is None:
        top_k = GenerationConfig._get_default_generation_params()["top_k"]
    if top_k != 0:
        processors.append(TopKLogitsWarper(top_k))
    processors.append(DeterministicTopP(top_p))
    return {
        "temperature": 1.0,
        "top_k": 0,
        "top_p": 1.0,
        "logits_processor": processors,
    }
