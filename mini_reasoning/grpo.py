"""GRPO loss with group-relative advantage. Role 2 wiring (reward).

The trainer harness calls the verifier itself and hands this module a
tensor of scalar rewards, one per rollout, grouped along the last axis:
`grpo.py` never sees text or extraction. Its two lessons are local:

- group_relative_advantages(rewards): per-group mean/std with epsilon;
  zero-variance groups return exact zeros (this is what lets updates
  run on all-same-reward groups safely). A group of one raises, because
  torch.std's n-1 divisor makes it NaN, not zero.
- grpo_loss(logprobs, advantages): advantages detached inside; scalar
  for gradient DESCENT (descent raises positive-advantage logprobs).

Trainer-level lessons that live in the harness, not here: clip grad to
norm 1.0 (pre-clip norms ran 260-400x the bound), accumulate backward
per rollout (six live graphs OOM'd 18.7GB on MPS), generation under
no_grad never inference_mode, and NO-KL DEFAULT: the signed-logratio
penalty is the sole gradient on zero-advantage steps and collapsed the
policy in 40 steps at this scale.
"""
import torch

_EPS = 1e-8


def group_relative_advantages(rewards):
    """Normalise rewards within each group along the last axis.

    Args:
        rewards: Tensor of shape (..., group_size), one reward per
            rollout, with the group along the last axis. A bool or
            integer tensor, which is what a harness builds straight from
            the verifier's verdicts, is cast to float first.

    Returns:
        Float tensor of the same shape: per group, (r - mean) / (std +
        eps) with the unbiased std, or exact zeros where the group has
        no variance.

    Raises:
        ValueError: if the group axis holds fewer than two rollouts.
    """
    if rewards.shape[-1] < 2:
        raise ValueError(
            f"group_size must be >= 2, got {rewards.shape[-1]}")
    # Flatness is judged on the rewards, not on the std: seven copies of
    # 0.1 have a float32 std near 1e-8, which the epsilon would then turn
    # into O(1) advantages instead of the exact zeros the trainer relies on.
    if not rewards.is_floating_point():
        rewards = rewards.float()
    flat = (rewards == rewards[..., :1]).all(dim=-1, keepdim=True)
    mean = rewards.mean(dim=-1, keepdim=True)
    std = rewards.std(dim=-1, keepdim=True)
    advantages = (rewards - mean) / (std + _EPS)
    return torch.where(flat, torch.zeros_like(advantages), advantages)


def grpo_loss(logprobs, advantages):
    """Return the scalar GRPO policy loss for gradient descent.

    Args:
        logprobs: Tensor of per-rollout sequence log-probabilities, with
            a graph back to the policy.
        advantages: Tensor of the same shape from
            group_relative_advantages. Detached here, so gradient flows
            only through logprobs.

    Returns:
        A 0-dim tensor, the mean of -advantage * logprob. Descending it
        raises the log-probability of positive-advantage rollouts.
    """
    return -(advantages.detach() * logprobs).mean()
