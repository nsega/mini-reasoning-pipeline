"""Contract tests for GRPO core (scaffold).

Week 1 pass criteria plus the group_size >= 2 guard added after the
NaN forensics (torch.std's n-1 divisor).
"""
import pytest
import torch

from mini_reasoning.grpo import group_relative_advantages, grpo_loss


class TestAdvantages:
    def test_zero_mean_unit_std(self):
        adv = group_relative_advantages(torch.tensor([1.0, 2.0, 3.0, 4.0]))
        assert adv.mean().item() == pytest.approx(0.0, abs=1e-5)
        assert adv.std().item() == pytest.approx(1.0, abs=1e-2)

    def test_per_group_normalization(self):
        rewards = torch.tensor([[1.0, 2.0, 3.0, 4.0], [100.0, 200.0, 300.0, 400.0]])
        adv = group_relative_advantages(rewards)
        assert torch.allclose(adv[0], adv[1], atol=1e-4)

    def test_zero_variance_is_exact_zeros(self):
        # Stronger than Week 1's isfinite: the trainer steps through
        # zero-variance groups, so their advantages must be exactly zero.
        adv = group_relative_advantages(torch.tensor([1.0, 1.0, 1.0]))
        assert torch.equal(adv, torch.zeros(3))

    def test_group_of_one_raises(self):
        with pytest.raises(ValueError):
            group_relative_advantages(torch.tensor([2.0]))


class TestLoss:
    def test_descent_direction(self):
        logprobs = torch.tensor([-1.0, -1.0], requires_grad=True)
        grpo_loss(logprobs, torch.tensor([1.0, -1.0])).backward()
        assert logprobs.grad[0].item() < 0
        assert logprobs.grad[1].item() > 0

    def test_advantages_detached(self):
        logprobs = torch.tensor([-1.0, -2.0], requires_grad=True)
        advantages = torch.tensor([1.0, -1.0], requires_grad=True)
        grpo_loss(logprobs, advantages).backward()
        assert advantages.grad is None or torch.all(advantages.grad == 0)
