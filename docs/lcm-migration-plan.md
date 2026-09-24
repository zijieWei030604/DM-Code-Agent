# LCM migration contract

Status: LCM is wired as the default backend when context compression is enabled.

## Current implementation boundary

The new `dm_agent/memory/lcm/` package contains SQLite branch-prefix storage,
summary provenance, explicit call/result boundaries, bounded positive-gain
compaction batches, and three read-only tool factories. `core/lcm_memory.py` and
`core/lcm_context_window.py` connect these to Agent. The old heuristic evidence
memory observer is removed; Planner and evidence-gate decisions are unchanged.
Legacy compressor/window helpers remain for standalone legacy tests, not the
active Agent path. The prior atomic-memory checkpoint schema is explicitly rejected.

## Runtime behavior

- Every normalized Runtime decision and its observation have a stable pairing ID.
  These are Runtime IDs, not a claim that provider-native call IDs are preserved.
  Normal and failed calls use the same grouping rule; no text matching is involved.
- The four built-in provider clients make bounded tool-free summary calls directly,
  bypassing request hooks. Output limit is 1024 tokens, transport timeout is 60 s,
  at most two attempts per batch and four batches per request. SDK retries are
  disabled for summary requests. A custom provider must implement `complete_summary`.
- Summary usage is recorded separately in `lcm_summary_call`; estimated input and
  output totals are reported in result metadata. Estimates use the existing chars/4
  estimator, not a provider tokenizer. Failed calls may have unknown actual usage.
- SQLite databases live under `~/.dm_agent/lcm/`. Normal messages are retained,
  redacted with the existing Trace rules. They are local sensitive data, not encrypted.
- On truncated output, the redacted full body is durably saved as a Trace artifact
  before its SQLite reference is committed. Without a Trace sink the full body stays
  in SQLite. Search never reads Trace; expand resolves only references associated
  with a visible record and verifies its event type and digest. Missing artifacts
  return an error rather than invented content.
- Checkpoints store database path, branch, cutoff, history IDs and context frontier.
  Resume/fork opens a child branch at that cutoff; later parent records and siblings
  cannot be searched or expanded. Keep the database and referenced Trace files when
  moving a checkpoint. This is not a self-contained checkpoint export.
- Zero context budget disables automatic compaction but retains storage and query
  tools. Disabling compression entirely disables the LCM backend and its tools.
- Summary request input is capped at 16000 estimated tokens per call and 64000
  accumulated estimated input tokens per context-building request. Historical
  usage remains in the checkpoint for accounting; it does not permanently disable
  future compaction after successful previous batches. Oversized protected
  groups or exhausted budgets stop safely if the active context cannot fit.

Live-model quality and cost must be measured again; no old benchmark scores are
claimed for this backend.

## Provenance

Architectural reference: `stephenschoettler/hermes-lcm` at
`8d1b1e6d3d63f5fc7b209e8d7ec1dc9b814f2e54`. The files currently added here are
independent implementations for this repository, not unmodified vendored upstream
files. No upstream source is copied. Further direct source reuse requires reading
and preserving that revision's applicable license and notices first.

## Scope

- Replace heuristic atomic-memory compaction with hierarchical LLM summaries.
- Preserve complete recent tool-call/result groups.
- Store ordinary messages in SQLite and expose bounded FTS5 search.
- Expose only lcm_grep, lcm_describe, and lcm_expand.
- Persist oversized, redacted tool output before truncation: use Trace when enabled,
  otherwise retain the full payload in SQLite.
- Search SQLite only; expand authorized Trace references on demand.
- Limit visibility to the current branch and its visible ancestors.
- Use direct, tool-free summary model calls with separate usage accounting.
- Commit each successful compaction batch independently; retain failed batches.
- Support bounded dynamic chunking and bounded multi-batch compaction.
- Stop on unrecoverable context overflow; do not silently discard history.
- Do not migrate legacy compressor checkpoints; support new-state resume and fork.
- Do not enable automatic focus extraction, ignored-reply propagation,
  cross-session retrieval, embeddings, or temporal rollups.

## Integration boundaries

Pin and attribute the upstream hermes-lcm revision before adapting code. Preserve
reusable algorithms and tests; do not emulate the Hermes runtime. Adapt storage,
summary DAG, fresh-tail boundaries, compaction orchestration, and tool entry points
to existing model clients, tool results, Trace, and checkpoint interfaces.

Keep Planner behavior, evidence acceptance, and the ReAct decision loop unchanged.
Remove obsolete atomic-memory dependencies only after replacing their callers.

## Verification

Integration verification on 2026-09-24:

- Complete offline pytest suite: 772 passed, 1 skipped (one existing Starlette warning).
- Python compileall passed; deterministic `full/direct_finish` eval passed.
- Maintenance benchmark listing loaded successfully; no live benchmark was run.
- Ruff and Black passed for all changed Python files. Focused mypy passed for 14
  LCM/provider/Trace modules.
- Whole-repository static checks still report pre-existing issues in unmodified
  modules (one Ruff finding, formatting differences, and 12 mypy errors). They
  were not folded into this migration.
- Tests include summary request isolation, provider request limits, source pairing,
  branch visibility, redacted Trace expansion, missing artifacts, overflow stopping,
  persisted summary reuse, and retry rollback. No real model API was called.

Use offline model doubles to test pairing, source expansion, branch isolation,
idempotent ingestion, partial-batch failure, checkpoint recovery, redaction, and
Trace-disabled storage. Run repository regression checks before live benchmarks.
Measure summary cost and total task cost separately. Do not reuse old benchmark
or compression metrics as measurements of the new implementation.

Do not commit or push these changes.
