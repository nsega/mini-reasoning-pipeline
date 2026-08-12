"""Scorer interface tests: land together with Naoki's interface design.

The interface is the capstone's hard constraint and is deliberately
not pre-designed by scaffolding (see mini_reasoning/scorers.py for the
design questions). These placeholders keep the obligation visible in
every test run without inventing the contract.
"""
import pytest


@pytest.mark.skip(reason="scorer interface is Naoki's design; tests land with it")
def test_scorers_share_one_call_contract():
    ...


@pytest.mark.skip(reason="scorer interface is Naoki's design; tests land with it")
def test_parseability_gate_precedes_ranking():
    ...


@pytest.mark.skip(reason="scorer interface is Naoki's design; tests land with it")
def test_same_verifier_serves_reward_and_validation_roles():
    ...
