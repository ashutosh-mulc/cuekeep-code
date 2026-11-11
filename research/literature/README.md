# Literature

| Folder | Contents |
|---|---|
| `surveys/` | Landscape surveys and codebase scans, date-prefixed (`YYYY-MM-DD-slug.md`) |
| `papers/` | **One reading note per paper**, named `<citekey>.md`. This folder is the ARS literature corpus: `bash scripts/build_corpus.sh` runs the ARS Obsidian adapter over it to produce `literature_corpus[]` for the Material Passport |
| [reading_queue.md](reading_queue.md) | Papers to read, with priority |

## Paper-note rules (ARS Obsidian adapter, Convention A)

- Copy [_paper_note_template.md](_paper_note_template.md) to `papers/<citekey>.md`.
- The frontmatter **must** have `citekey`, `title`, `authors`, `year`. Copy these from the paper
  itself (arXiv or the venue page), never from memory or a summary. The ARS integrity gates check
  them against the source.
- Keep only paper notes in `papers/`. Any other `.md` file there would be ingested as a corpus entry.
- Citekey style: `familyYYYYword`, e.g. `chen2024shadowkv`.
