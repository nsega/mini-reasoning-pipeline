"""Before/after evaluation on the frozen subset. Roles 1 and 3.

The same function runs the baseline eval before training and the
validation eval after it, on the same frozen subset, with the same
scorer and the same extraction fallback. Role 3 grades with the verifier
that trained the model in role 2: that circularity is the point, and the
README discusses it. The full record is in docs/design-notes.md.

- evaluate builds the material bundle the scorer protocol expects (text
  plus a logprob summary per sample) and stores it in the per-problem
  record, so a composite can be re-run offline over the record. Every
  lab discovery came from re-analysing raw records.
- Sampling is the one model-bound step and sits behind the sampler
  argument, so everything else is tested against real scorers and the
  real verifier. The default sampler wraps Hugging Face generation.
- Selection is the caller's argmax, checked for the all-rejected case
  first: a composite that rejects everything returns all -inf and says
  nothing more, and a bare argmax would silently pick index 0.
- Accuracy is reported raw and reweighted to MATH-500's level mix. The
  seed-42 subset skews hard (mean level 3.88 against 3.44 for the full
  set), so the raw number understates the model.
"""
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict

import torch

from mini_reasoning import verifier
from mini_reasoning.scorers import REJECTED, Candidate, Scorer

MATH500_LEVEL_COUNTS = {1: 43, 2: 90, 3: 105, 4: 128, 5: 134}

Sampler = Callable[[object, object, str, int], Sequence[Candidate]]

_PROMPT = ("Problem: {problem}\n"
           "Solve the problem. Put the final answer in \\boxed{{}}.\n"
           "Solution:")


def sample_solutions(model, tokenizer, problem: str, n: int,
                     max_new_tokens: int = 512, temperature: float = 0.7,
                     top_p: float = 0.95,
                     logprob_batch: int = 1) -> list[Candidate]:
    """Samples n solutions and summarises each by its mean token logprob.

    The prompt is plain text rather than the tokenizer's chat template:
    the pipeline's model is a base model, and on Qwen3-0.6B-Base the
    chat template made it regurgitate prompts (0 of 3 usable) where the
    plain prompt at temperature 0.7 boxed 3 of 3.

    The summary is taken from the raw logits rather than from the
    processed scores. The scores have already been divided by the
    temperature and truncated to the nucleus, so summarising them would
    rank candidates by the sampling distribution instead of by the
    model: on one 8-sample probe the two orderings differed. The logits
    are recomputed in a second forward pass rather than retained by
    generation, which would hold 2 GB at the defaults before scoring
    copies it.

    Args:
        model: A causal LM with generate and compute_transition_scores.
        tokenizer: Its tokenizer.
        problem: The problem statement.
        n: Number of independent samples to draw.
        max_new_tokens: Generation cap per sample.
        temperature: Sampling temperature; 1.0 boxed 1 of 3 on the same
            probe where 0.7 boxed 3 of 3.
        top_p: Nucleus sampling cutoff.
        logprob_batch: Rows scored per forward pass. One keeps the peak
            at a single sequence's logits.

    Returns:
        One Candidate per sample: the decoded text, the mean logprob the
        model assigned to its generated tokens with padding excluded,
        and whether it stopped rather than hitting the cap.
    """
    eos_ids = _eos_ids(model, tokenizer)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = eos_ids[0]
    inputs = tokenizer(_PROMPT.format(problem=problem), return_tensors="pt")
    inputs = inputs.to(model.device)
    prompt_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        sequences = model.generate(
            **inputs, do_sample=True, temperature=temperature, top_p=top_p,
            num_return_sequences=n, max_new_tokens=max_new_tokens,
            pad_token_id=pad_id)
        token_logprobs = _generated_token_logprobs(
            model, sequences, prompt_len, logprob_batch)
    generated = sequences[:, prompt_len:]
    summaries = sequence_logprob_summary(token_logprobs, generated, eos_ids)
    stops = torch.tensor(eos_ids, device=generated.device)
    finished = torch.isin(generated, stops).any(-1)
    texts = tokenizer.batch_decode(generated, skip_special_tokens=True)
    return [Candidate(text=text, logprob_summary=float(summary),
                      finished=bool(done))
            for text, summary, done in zip(texts, summaries, finished)]


def token_logprobs_from_logits(logits: torch.Tensor,
                               token_ids: torch.Tensor) -> torch.Tensor:
    """Logprob of each realised token, without a full log_softmax copy.

    log_softmax allocates a second tensor the size of the logits, which
    are the largest thing the sampler holds. Subtracting the logsumexp
    from the gathered logit is the same quantity and reduces over the
    vocabulary rather than materialising it.

    Args:
        logits: Shape (rows, steps, vocab).
        token_ids: Shape (rows, steps), the token realised at each step.

    Returns:
        Shape (rows, steps).
    """
    chosen = logits.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)
    return chosen - logits.logsumexp(-1)


def _generated_token_logprobs(model, sequences: torch.Tensor, prompt_len: int,
                              batch_size: int) -> torch.Tensor:
    """Scores the generated tail a few rows at a time.

    Asking generation to return its logits retains one vocabulary-wide
    tensor per step: 2 GB at the default cap and sample count, before
    the copies scoring them makes, which is enough to end a 50-problem
    eval partway. A second forward pass spends compute instead and
    bounds the peak at one chunk of rows.
    """
    width = max(batch_size, 1)
    chunks = []
    for start in range(0, sequences.shape[0], width):
        rows = sequences[start:start + width]
        logits = model(rows).logits[:, prompt_len - 1:-1, :].float()
        chunks.append(token_logprobs_from_logits(logits,
                                                 rows[:, prompt_len:]))
        del logits
    return torch.cat(chunks)


def _eos_ids(model, tokenizer) -> tuple[int, ...]:
    """The ids generation actually stops on, as a tuple.

    Taken from the generation config rather than the tokenizer: a chat
    checkpoint often stops on several ids, and one the tokenizer does
    not name would leave an early-finished row looking unfinished.
    """
    eos = getattr(model.generation_config, "eos_token_id", None)
    if eos is None:
        eos = getattr(tokenizer, "eos_token_id", None)
    ids = (eos,) if isinstance(eos, int) else tuple(eos or ())
    if not ids:
        raise ValueError(
            "no stop token: neither the generation config nor the "
            "tokenizer names an eos_token_id, so a sample that finished "
            "cannot be told from one that hit the cap")
    return ids


def sequence_logprob_summary(token_logprobs: torch.Tensor,
                             generated: torch.Tensor,
                             eos_ids: int | Iterable[int]) -> torch.Tensor:
    """Mean token logprob per row, up to and including the first EOS.

    Positions after the first stop token are padding, and are excluded
    by selection rather than by a multiplicative mask. The mask came
    first from the processed scores, which carry -inf there, and -inf
    times zero is NaN; the caller now passes raw logits, where those
    positions hold ordinary values, but selection stays because it is
    exact either way. A row that never stopped averages every token.

    Args:
        token_logprobs: Shape (rows, steps), one logprob per generated
            token.
        generated: Shape (rows, steps), the generated token ids.
        eos_ids: The id generation stops on, or every such id.

    Returns:
        Shape (rows,), the per-row mean.
    """
    if isinstance(eos_ids, int):
        eos_ids = (eos_ids,)
    stops = torch.tensor(tuple(eos_ids), device=generated.device)
    is_eos = torch.isin(generated, stops)
    finished_before = is_eos.cumsum(-1) - is_eos.int()
    keep = finished_before == 0
    kept = token_logprobs.masked_fill(~keep, 0.0)
    return kept.sum(-1) / keep.sum(-1).clamp(min=1)


_REQUIRED_FIELDS = ("unique_id", "problem", "answer", "level")


def _validate(problems: Sequence[Mapping]) -> None:
    """Checks every problem before the sampling loop runs.

    The reweighting looks each level up in MATH-500's counts, so an
    unknown level would otherwise raise after a whole run of generation
    and return nothing.
    """
    for position, problem in enumerate(problems):
        missing = [f for f in _REQUIRED_FIELDS if f not in problem]
        if missing:
            raise ValueError(
                f"problem {position} is missing {', '.join(missing)}")
        if problem["level"] not in MATH500_LEVEL_COUNTS:
            raise ValueError(
                f"problem {position} has level {problem['level']!r}, not "
                f"one of {sorted(MATH500_LEVEL_COUNTS)}")


def _select(scores: Sequence[float]) -> int | None:
    """Index of the first maximum, or None when every score is REJECTED."""
    if all(score == REJECTED for score in scores):
        return None
    return max(range(len(scores)), key=scores.__getitem__)


def _level_reweighted_accuracy(records: Sequence[Mapping]) -> float:
    """Accuracy under MATH-500's level shares, renormalised over the
    levels present so a level missing from the subset drops out rather
    than zeroing the total."""
    correct: dict[int, int] = {}
    total: dict[int, int] = {}
    for record in records:
        level = record["level"]
        total[level] = total.get(level, 0) + 1
        correct[level] = correct.get(level, 0) + int(record["correct"])
    weight = sum(MATH500_LEVEL_COUNTS[level] for level in total)
    return sum(MATH500_LEVEL_COUNTS[level] / weight * correct[level]
               / total[level] for level in total)


def evaluate(model, tokenizer, problems: Sequence[Mapping], scorer: Scorer,
             *, n_samples: int = 7, fallback: str | None = None,
             sampler: Sampler = sample_solutions, seed: int | None = None,
             on_record: Callable[[dict], None] | None = None) -> dict:
    """Samples, selects and grades every problem, and stores the record.

    Args:
        model: Passed through to the sampler.
        tokenizer: Passed through to the sampler.
        problems: Mappings with "unique_id", "problem", "answer" and
            "level", as in the frozen MATH-500 subset.
        scorer: Any Scorer; selection is argmax over its scores.
        n_samples: Samples drawn per problem, the candidate-set size.
        fallback: Extraction fallback for grading. None is boxed-only,
            the strict default; both eval call sites pass "number".
        sampler: Returns the material bundle for one problem. It is
            called with exactly (model, tokenizer, problem, n), so the
            default sampler's own knobs (the token cap, temperature and
            top-p) are set by partial application, not through here.
        seed: Seeds the sampling per problem, so a run over the frozen
            subset repeats. None leaves the global RNG alone, and a
            before/after delta then carries sampling noise.
        on_record: Called with each record as it is finished. The
            returned list is lost if the loop raises, so a caller that
            must not lose a long run persists from here.

    Returns:
        A dict with "accuracy", "level_reweighted_accuracy" and
        "records", one record per problem in input order holding the
        samples, extracted candidates, scores, selected index and
        correctness.

    Raises:
        ValueError: if problems is empty, if n_samples is below one, if
            any problem lacks a required field or carries a level
            outside MATH-500's five, or if the sampler returns other
            than n_samples samples.
    """
    if not problems:
        raise ValueError("evaluate needs at least one problem")
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}")
    _validate(problems)
    records = []
    for position, problem in enumerate(problems):
        if seed is not None:
            torch.manual_seed(seed + position)
        samples = list(sampler(model, tokenizer, problem["problem"],
                               n_samples))
        if len(samples) != n_samples:
            raise ValueError(
                f"sampler returned {len(samples)} samples for problem "
                f"{problem['unique_id']!r}, expected {n_samples}")
        candidates = [verifier.extract_final_candidate(s.text, fallback)
                      for s in samples]
        scores = list(scorer.score(samples))
        selected = _select(scores)
        correct = selected is not None and verifier.verify_answer(
            candidates[selected], problem["answer"])
        record = {
            "unique_id": problem["unique_id"],
            "level": problem["level"],
            "answer": problem["answer"],
            "samples": [asdict(s) for s in samples],
            "candidates": candidates,
            "scores": scores,
            "selected": selected,
            "correct": bool(correct),
        }
        records.append(record)
        if on_record is not None:
            on_record(record)
    accuracy = sum(r["correct"] for r in records) / len(records)
    return {
        "accuracy": accuracy,
        "level_reweighted_accuracy": _level_reweighted_accuracy(records),
        "records": records,
    }
