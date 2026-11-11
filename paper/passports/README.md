# Material Passport

`cuekeep_passport.yaml` (ARS Schema 9) and its run ledger, created by the ARS pipeline on first run. Do not hand-edit the ledger.

- `experiment_provenance.generated.yaml`: output of `python scripts/export_provenance.py`, for the scholar to supply at the experiment-intake question.
- `literature_corpus.generated.yaml`: output of `bash scripts/build_corpus.sh` (Obsidian adapter over `research/literature/papers/`).

The `experiment_intake_declaration` is answered by you, in your own words, when ARS asks. It is never pre-filled.
