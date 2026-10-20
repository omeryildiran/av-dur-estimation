# data_solid

Copies of the **complete sessions of real participants on the lab display**, made automatically when a session ends, plus each participant's confirmed bias files. The analysis notebooks read this folder.

Not copied here:
- practice runs;
- sessions stopped before the end;
- test-display profiles (`office_test`, `laptop_test`);
- autotests;
- simulated runs.

`data/` keeps every real run exactly as recorded, including stopped sessions.

To exclude a session from the analysis (for example, a participant who did not follow the instructions), delete its `.csv` and `.json` here and note why. The original stays in `data/`.

The layout is the same as `data/`: `<PID>/<task>/<PID>_<task>[_<block>]_<date-time>.csv` and `.json`, and `<PID>/bias/<PID>_bias_<date-time>.json`.
