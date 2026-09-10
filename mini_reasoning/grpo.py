"""GRPO loss with group-relative advantage. Role 2 wiring (reward).

The trainer harness calls the verifier itself and hands this module a
tensor of scalar rewards, one per rollout, grouped along the last axis:
`grpo.py` never sees text or extraction. Its two lessons are local:

- group_relative_advantages(rewards): per-group mean/std with epsilon;
  flat groups, where every reward equals the first, return exact zeros
  (this is what lets updates run on all-same-reward groups safely). A
  group of one raises, because torch.std's n-1 divisor makes it NaN,
  not zero.
- grpo_loss(logprobs, advantages): advantages detached inside; scalar
  for gradient DESCENT (descent raises positive-advantage logprobs).

Trainer-level lessons that live in the harness, not here: clip grad to
norm 1.0 (pre-clip norms ran 260-400x the bound), accumulate backward
per rollout (six live graphs OOM'd 18.7GB on MPS) and scale each such
call by 1/G, since grpo_loss averages only what it is handed, build the
per-rollout sequence logprob handed here (a token sum or a token mean
sets a further 1/T on the gradient scale, and this module takes the
scalar as given), generation under no_grad never inference_mode, and
NO-KL DEFAULT: the signed-logratio penalty is the sole gradient on
zero-advantage steps and collapsed the policy in 40 steps at this
scale.
"""
import torch

_EPS = 1e-8


def group_relative_advantages(rewards: torch.Tensor) -> torch.Tensor:
    """Normalise rewards within each group along the last axis.

    Args:
        rewards: Tensor of shape (..., group_size), one reward per
            rollout, with the group along the last axis. Any dtype is
            accepted and normalised in float32: a bool or integer tensor
            is what a harness builds straight from the verifier's
            verdicts, and float16 cannot represent the epsilon.

    Returns:
        Float32 tensor of the same shape: per group, (r - mean) / (std +
        eps) with the unbiased std, or exact zeros where every reward in
        the group equals the first. Flatness is judged by exact equality
        on the rewards, not by the std being zero: seven copies of 0.1
        have a nonzero float32 std, and the epsilon would turn that
        rounding noise into an advantage of -0.41 on every rollout.

    Raises:
        ValueError: if the group axis holds fewer than two rollouts, if
            the tensor is 0-dim or holds no elements, or if any reward
            is NaN or infinite.
    """
    if rewards.dim() == 0 or rewards.shape[-1] < 2:
        size = None if rewards.dim() == 0 else rewards.shape[-1]
        raise ValueError(f"group_size must be >= 2, got {size}")
    if rewards.numel() == 0:
        raise ValueError(
            f"rewards holds no groups, got shape {tuple(rewards.shape)}")
    if not torch.isfinite(rewards).all():
        raise ValueError("rewards must be finite, got NaN or inf")
    rewards = rewards.float()
    flat = (rewards == rewards[..., :1]).all(dim=-1, keepdim=True)
    mean = rewards.mean(dim=-1, keepdim=True)
    std = rewards.std(dim=-1, keepdim=True)
    advantages = (rewards - mean) / (std + _EPS)
    return advantages.masked_fill(flat, 0.0)


def grpo_loss(logprobs: torch.Tensor,
              advantages: torch.Tensor) -> torch.Tensor:
    """Return the scalar GRPO policy loss for gradient descent.

    Args:
        logprobs: Tensor of per-rollout sequence log-probabilities, with
            a graph back to the policy.
        advantages: Tensor of the same shape from
            group_relative_advantages. Detached here, so gradient flows
            only through logprobs.

    Returns:
        A 0-dim tensor, the mean of -advantage * logprob over every
        element handed in. Descending it raises the log-probability of
        positive-advantage rollouts. The mean sees only this call: a
        harness that accumulates backward one rollout at a time gets a
        sum over the group unless it scales each call by 1/G.

    Raises:
        ValueError: if the two shapes differ, or if they hold no
            elements. Broadcasting a (G, 1) column against a (G,) row
            would form an outer product and hand every rollout the mean
            advantage's gradient, silently; a mean over nothing is NaN.
    """
    if logprobs.shape != advantages.shape:
        raise ValueError(
            "logprobs and advantages must have the same shape, got "
            f"{tuple(logprobs.shape)} and {tuple(advantages.shape)}")
    if logprobs.numel() == 0:
        raise ValueError(
            f"logprobs holds no rollouts, got shape {tuple(logprobs.shape)}")
    return -(advantages.detach() * logprobs).mean()
