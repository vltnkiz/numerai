# 1. The live entrypoint runs one named strategy

## Status

Accepted

## Context

`scripts/run_pipeline.py` is the only entrypoint that submits: every
invocation that finds an unfinished round uploads to `SUBMISSION_MODEL_SLOT`,
overwriting the round's real submission. Before this change it took a
`--model` flag choosing among `MODEL_NAMES`, and set the per-model fields on
`PipelineConfig` from that flag.

The scheduled task (`scheduling/Invoke-ZemirLiveRun.ps1`) has been passing
`--model ensemble` on every invocation, unconditionally, while
`zemir/config.py`'s `PRODUCTION_BASELINE` comment still claimed production
was "linear only, neutralized at 0.5". The flag and the config had drifted
apart, unnoticed, for months — see issue #74's "Out of scope" section, which
records this as a defect worse than originally found. A flag on the live
entrypoint is a *second* place the production decision can be made, and the
two can silently disagree; that drift is exactly how this one happened.

`zemir/strategy.py` (issue #76) and `zemir/config.py`'s `STRATEGIES`
registry (issue #77) replaced the three tables the old `--model` flag keyed
into with named, versioned `Strategy` instances, and
`PRODUCTION_STRATEGY = STRATEGIES["zemir_01"]` — one line in `config.py` —
now names what production ships.

## Decision

`scripts/run_pipeline.py` takes no strategy argument. It always runs
`zemir.config.PRODUCTION_STRATEGY`. Changing what production ships means
editing that one line in `zemir/config.py`, not passing a flag at invocation
time.

`scripts/run_experiment.py` and `scripts/fit_harness.py` — the two
entrypoints that measure without submitting — keep a `--strategy` flag,
since trying a strategy other than the production one is their whole job.

## Alternative considered

Keep a `--strategy` flag on `run_pipeline.py` too, defaulting to
`PRODUCTION_STRATEGY`.

Rejected: a flag that merely *defaults* to the production strategy is still
a second place the production decision can be expressed, and the whole
failure mode this ADR exists to close is exactly that shape — a
value passed at invocation time silently taking precedence over `config.py`'s
own claim about what ships. A future reader who doesn't know this history
will likely see the missing flag on `run_pipeline.py` as an oversight —
`run_experiment.py` and `fit_harness.py` both keep theirs — and helpfully add
it back, recreating the two-decision-sites drift this document exists to
prevent.

## Consequences

Trying a different strategy in production requires editing
`PRODUCTION_STRATEGY` in `zemir/config.py`, not passing a flag. That edit is
a code change — reviewed, committed, and visible in `git log` — rather than
a value chosen by whatever last invoked the scheduled task.

See issue #78.
