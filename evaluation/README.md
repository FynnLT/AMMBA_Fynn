# Evaluation harness

**`evaluation/` is not part of the artifact.** It is the simulation harness for
the thesis evaluation: it drives the three services in process, records what
they answer, and writes figures. Nothing here is deployed, and nothing in the
three services depends on it.

## Frozen scripts: `harness/legacy/`

`eval_r0_r1.py`, `eval_r2.py`, `eval_r3_r4.py`, `eval_r5.py`, `blockd.py`,
`blockd2.py` and `blocke.py` are frozen at the 02.09.2026 test run and run
against `git checkout fe52d63`, not against the current runner signature — it
now requires an explicit community and slot, and these scripts call `run_slot`
positionally. They produced the 02.09. test-run figures; no Chapter 5 result
depends on them, so they are kept as a record rather than adapted.
