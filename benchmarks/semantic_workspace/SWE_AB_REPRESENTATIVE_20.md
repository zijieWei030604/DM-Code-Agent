# Semantic Workspace SWE-bench A/B Set

This frozen set is drawn from the latter 30 instances of the project's established 50-instance
SWE-bench Verified selection. It contains two instances from each of ten repositories.

Selection deliberately includes all outcome categories observed in the earlier run:

- baseline-only resolved;
- Semantic Workspace-only resolved;
- both resolved;
- both unresolved or empty;
- long and max-step trajectories.

The set must not be changed in response to new A/B results. Baseline and experiment runs use the
same manifest, model, temperature, step limit, timeout, and workspace configuration. The only
experimental difference is `--enable-semantic-workspace`.
