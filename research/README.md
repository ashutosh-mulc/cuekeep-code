# Research record

This folder is the project's memory. If something was decided, learned, or ruled out, it is written
down here. The repository history is the audit trail.

| File / folder | Purpose | When to update |
|---|---|---|
| [overview.md](overview.md) | Problem, hypotheses, claimed contributions | When the pitch changes |
| [method.md](method.md) | The method, precisely: math, data structures, budgets | When the algorithm changes |
| [related_work.md](related_work.md) | Closest work, the novelty argument, what we must cite and contrast | When a new close paper appears |
| [roadmap.md](roadmap.md) | Milestones toward the deadline, with status | Weekly |
| [open_questions.md](open_questions.md) | Risks and unknowns, each linked to the experiment that resolves it | As questions open and close |
| [literature/](literature/) | Survey notes and reading notes, one file per source or survey, date-prefixed | After reading |
| [decisions/](decisions/) | Decision records (`NNNN-slug.md`): context, options, decision, consequences | When a choice would be costly to reverse |
| [notebook/](notebook/) | Lab notebook, one file per working day (`YYYY-MM-DD.md`) | Every working session |

## Logging rules

1. **Notebook first.** Each working session gets an entry in `notebook/YYYY-MM-DD.md` covering what
   was tried, what happened, and what comes next. Link the runs (`runs/E###/<ts>`) and experiments involved.
2. **Experiments are pre-registered.** Write the hypothesis and the success criterion in
   `experiments/E###/README.md` *before* running. Do not edit them afterwards. Add to the
   Results and Conclusion sections instead.
3. **No orphan numbers.** Every number in a table, plot or paper draft must trace back to a run
   directory, which records its config, git SHA and environment.
4. **Negative results count.** Log them with the same care as positive ones, and mark the hypothesis
   ✗ in the experiment index.
5. **Decisions get a record.** Examples: choosing a base codebase, changing the primary model, or
   dropping a baseline. Use [decisions/_template.md](decisions/_template.md).
6. **Dates are absolute** (`2026-11-11`), never "last week".
