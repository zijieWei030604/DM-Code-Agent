# Semantic Workspace Impact Benchmark

This benchmark evaluates change-impact predictions without an LLM. Each case contains a small
Python repository, one or more changed files, and manually labeled affected source files and tests.

Run it with:

```powershell
dm-agent-impact-eval `
  --manifest benchmarks/semantic_workspace/impact_cases.json `
  --output bench_reports/semantic-impact-baseline.json `
  --markdown bench_reports/semantic-impact-baseline.md
```

The report separates affected production files from related tests and records false positives and
false negatives per case. These controlled cases are the first evaluation tier; repository-derived
cases should be added as a separate tier before claiming real-world accuracy.

The repository-derived tier evaluates 20 manually checked relationships in this project. It mixes
production impact cases with focused test-selection cases; an axis is scored only when its gold set
can be labeled exhaustively:

```powershell
dm-agent-impact-eval `
  --manifest benchmarks/semantic_workspace/repository_impact_cases.json `
  --output bench_reports/semantic-impact-repository.json `
  --markdown bench_reports/semantic-impact-repository.md
```

Each case can disable an axis when its gold set cannot be exhaustively labeled. Disabled axes are
excluded from aggregate metrics instead of treating unknown relationships as false positives.
Related tests are returned in dependency-distance order, with exact filename companions promoted,
and are capped at eight files to keep downstream verification work bounded.
