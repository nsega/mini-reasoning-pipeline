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
        rewards = torch.tensor([[1.0, 2.0, 3.0, 4.0],
                                [100.0, 200.0, 300.0, 400.0]])
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


class TestGroupAxis:
    def test_shape_is_preserved(self):
        rewards = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        assert group_relative_advantages(rewards).shape == rewards.shape

    def test_each_row_normalises_against_its_own_row(self):
        # Hand-derived: each row has mean 0.5 and unbiased std sqrt(1/3),
        # so both rows land on the same +-0.866 pattern. A global mean or a
        # global std would put row 1 far from it.
        rewards = torch.tensor([[0.0, 1.0, 1.0, 0.0],
                                [100.0, 101.0, 101.0, 100.0]])
        want = torch.tensor([[-0.866, 0.866, 0.866, -0.866]] * 2)
        assert torch.allclose(group_relative_advantages(rewards), want,
                              atol=1e-3)

    def test_group_of_one_raises_in_a_batch_too(self):
        # The guard is on the group axis, not on the element count: two
        # groups of one is still a group of one.
        with pytest.raises(ValueError):
            group_relative_advantages(torch.tensor([[2.0], [3.0]]))


class TestZeroVariance:
    def test_flat_group_does_not_poison_its_batch(self):
        # The trainer steps through zero-variance groups inside a batch: the
        # flat row must be exact zeros while the live row normalises as if
        # it were alone ([0, 1, 2] has mean 1 and unbiased std 1).
        rewards = torch.tensor([[1.0, 1.0, 1.0], [0.0, 1.0, 2.0]])
        adv = group_relative_advantages(rewards)
        assert torch.equal(adv[0], torch.zeros(3))
        assert torch.allclose(adv[1], torch.tensor([-1.0, 0.0, 1.0]),
                              atol=1e-5)

    def test_flat_group_of_inexact_floats_is_still_exact_zeros(self):
        # Seven copies of 0.1 do not sum to 0.7 in float32, so their std is
        # ~1e-8 rather than 0. Flatness must be judged on the rewards
        # themselves, not on a std that rounding can lift off zero.
        adv = group_relative_advantages(torch.full((7,), 0.1))
        assert torch.equal(adv, torch.zeros(7))

    def test_flat_group_yields_no_update(self):
        # End to end, rewards to gradient: an all-same-reward group must
        # move the policy by exactly nothing. With no KL term this is the
        # only thing that keeps zero-advantage steps inert.
        logprobs = torch.tensor([-0.5, -1.0, -1.5], requires_grad=True)
        adv = group_relative_advantages(torch.tensor([1.0, 1.0, 1.0]))
        grpo_loss(logprobs, adv).backward()
        assert torch.equal(logprobs.grad, torch.zeros(3))


class TestLossValue:
    def test_loss_is_a_scalar(self):
        loss = grpo_loss(torch.tensor([-1.0, -2.0]), torch.tensor([1.0, -1.0]))
        assert loss.dim() == 0

    def test_batched_rollouts_reduce_to_one_scalar(self):
        loss = grpo_loss(torch.full((2, 3), -1.0), torch.zeros(2, 3))
        assert loss.dim() == 0

    def test_loss_is_the_mean_of_negative_weighted_logprobs(self):
        # Mean, not sum: -((1)(-1) + (-1)(-2)) / 2 = -0.5. A sum would give
        # -1.0 and make the step size grow with the group.
        loss = grpo_loss(torch.tensor([-1.0, -2.0]), torch.tensor([1.0, -1.0]))
        assert loss.item() == pytest.approx(-0.5)

    def test_gradient_is_minus_advantage_over_n(self):
        # d(loss)/d(logprob_i) = -A_i / n, independent of the logprob value.
        logprobs = torch.tensor([-1.0, -1.0, -1.0, -1.0], requires_grad=True)
        grpo_loss(logprobs, torch.tensor([2.0, -2.0, 1.0, -1.0])).backward()
        assert torch.allclose(logprobs.grad,
                              torch.tensor([-0.5, 0.5, -0.25, 0.25]))
