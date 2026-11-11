#!/usr/bin/env bash
# Build the ARS literature_corpus[] from research/literature/papers/*.md (Obsidian adapter, Convention A).
# ARS_HOME defaults to the newest installed academic-research-skills plugin.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ARS_HOME="${ARS_HOME:-$(ls -d "$HOME"/.claude/plugins/cache/academic-research-skills/academic-research-skills/*/ 2>/dev/null | sort -V | tail -1)}"
if [[ -z "${ARS_HOME}" || ! -f "${ARS_HOME}/scripts/adapters/obsidian.py" ]]; then
  echo "ARS not found; set ARS_HOME to the academic-research-skills checkout" >&2; exit 1
fi
OUT="$ROOT/paper/passports"
mkdir -p "$OUT"
python3 "$ARS_HOME/scripts/adapters/obsidian.py" \
  --input "$ROOT/research/literature/papers" \
  --passport "$OUT/literature_corpus.generated.yaml" \
  --rejection-log "$OUT/literature_corpus.rejections.yaml"
python3 "$ARS_HOME/scripts/check_literature_corpus_schema.py" --passport "$OUT/literature_corpus.generated.yaml" || {
  echo "schema check failed; see $OUT/literature_corpus.rejections.yaml" >&2; exit 1; }
echo "corpus → paper/passports/literature_corpus.generated.yaml"
