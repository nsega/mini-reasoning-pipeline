"""The swappable scorer interface: the capstone's hard design constraint.

THIS INTERFACE IS NAOKI'S DESIGN. This stub deliberately contains no
interface, only the design questions the lab's ablations left behind.
Tests land together with the design (tests/test_scorers.py holds
skipped placeholders until then).

Questions the design must answer:

1. Signature: what does a scorer consume (sample texts? extracted
   candidates? per-token logprobs? all three?) and what does it return
   (per-sample scores? a selected index? a distribution?). Note the
   logprob scorer needs model access while the heuristic does not:
   where does that asymmetry live?
2. Composition: the lab's verdict was "compose signals: parseability
   gate -> rank -> vote agreement". Is composition itself a scorer
   (Composite(list_of_scorers))? How are stages ordered and vetoed?
3. Three roles, one object: the same verifier must serve evaluation
   (grading), reward (binary signal inside GRPO), and validation
   (before/after eval). What is shared, what differs per role, and
   where does the strict boxed-only reward mode plug in?
4. Failure honesty: every scorer must expose enough bookkeeping to
   reanalyze offline (the lab's selected-None-rate discovery was only
   possible because raw candidates were stored).
"""
