# prod/

Each subdirectory is one pipeline: `prod/<pipeline>/`.

- Same input: the live Numerai Classic dataset.
- Same output: predictions submitted for the current round (daily/weekly cadence per pipeline).
- Free choice inside that: model(s), ensembling, feature set, whatever the pipeline needs.

Currently: `zemir_0.1`.
