# Design notes: the scorer interface

> Status: draft — fills in as the interface design lands. This is the one
> design document that ships with the public repo. The README's Design
> rationale gives one bullet per decision; this file holds the full version:
> what was decided, what was rejected, and the evidence for each.
>
> Provenance: all decisions recorded here are self-written (the scorer
> interface is the capstone's closed-book constraint). The section skeleton
> was scaffolded; the content is Naoki's.

<!--
HOW TO FILL THIS FILE (delete this comment before merge)

Source: your filled answers in docs/scorer-interface-worksheet.md (branch-
only), rewritten in your own voice. Each section follows the same rubric:
decision / rejected option + the concrete failure it causes / evidence (a
lab measurement or a forcing call site) / cost accepted / the test that
pins it. Prefer stable references (function names) over file:line numbers,
which rot as implementations land. Worksheet question mapping is noted per
section as an HTML comment.
-->

## One interface or two

<!-- worksheet Q3.0: selection (no ground truth) vs verification (requires it) -->

**Decision.** _TODO_

**Rejected.** _TODO_

**Evidence.** _TODO_

**Cost accepted.** _TODO_

**Pinned by.** _TODO_

## What a scorer consumes and returns

<!-- worksheet Q1: signature, output type, and where the model asymmetry lives -->

**Decision.** _TODO_

**Rejected.** _TODO_

**Evidence.** _TODO_

**Cost accepted.** _TODO_

**Pinned by.** _TODO_

## Composition and the veto

<!-- worksheet Q2: is Composite a scorer; what "reject" means on each path -->

**Decision.** _TODO_

**Rejected.** _TODO_

**Evidence.** _TODO_

**Cost accepted.** _TODO_

**Pinned by.** _TODO_

## Three roles, one verifier

<!-- worksheet Q3: evaluation / reward / validation; where strictness binds -->

**Decision.** _TODO_

**Rejected.** _TODO_

**Evidence.** _TODO_

**Cost accepted.** _TODO_

**Pinned by.** _TODO_

## Forensics: obligation or convention

<!-- worksheet Q4: metadata from inside the scorer, or storage by the harness -->

**Decision.** _TODO_

**Rejected.** _TODO_

**Evidence.** _TODO_

**Cost accepted.** _TODO_

**Pinned by.** _TODO_

## The contract tests

<!-- worksheet Q5: 2-3 properties that must hold for ANY implementation -->

_TODO — one line per property, each naming the test in `tests/test_scorers.py`
that enforces it._
