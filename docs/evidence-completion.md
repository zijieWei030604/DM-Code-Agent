# Evidence completion checks

The optional evidence capability records facts through lifecycle hooks. It does
not infer correctness from a precise pytest node ID or a model's `direct` label.

## Blocking evidence

- A `run_tests` assertion failure may block when a prior successful check has
  the same arguments, test/helper/configuration fingerprints and observed runtime
  identity, but a different workspace version. The verification node records
  `baseline_node_id` and a `compared_with` edge.
- Test context and interpreter/package fingerprints are collected before and
  after verification. Missing or changing context cannot establish a baseline.
  Directory targets that include source code may conservatively prevent comparison.
- An error mentioning Python syntax triggers a bounded syntax-only probe of
  changed Python files. The local interpreter is used for local runs; SWE-bench
  injects its existing container validation runner. The probe compiles source in
  memory without executing it or writing bytecode. Unavailable probes do not block.
- Other failures remain available to the model and in the graph. `direct` describes
  target precision, not issue relevance or causation. Custom shell checks do not
  establish regression baselines in this implementation.

All original verification records remain in the graph/Trace. An unavailable
retry cannot replace an effective conclusion for the same check and workspace
version. A successful comparable retry can clear a contradiction. Changing the
workspace invalidates the old version's conclusions for completion purposes.
Runtime fingerprints describe observable interpreter/packages/configuration;
they do not guarantee determinism or stable external services, clocks or data.

## Command mutations and intervention

Shell/Python calls are bracketed by source/config fingerprints using the workspace
scanner's existing exclusions. Actual additions, edits and deletions create change
nodes even if the command fails. Each call increments the change revision once.
Binary outputs and ignored paths are outside this scanner's scope. A check that
changes the workspace cannot verify its own resulting version.

The existing one-time unverified-completion reminder and repeated-contradiction
policy remain. The latter can end a run as `critic_rejected` without deleting the
patch. No baseline tests are automatically run and no Runtime refactor is required.

Focused verification:

```console
python -m pytest tests/test_evidence_regression.py tests/test_evidence_graph.py tests/test_runtime_contracts.py tests/test_swebench_verified.py
```

Benchmark comparisons require a fresh run; historical scores do not measure this policy.
