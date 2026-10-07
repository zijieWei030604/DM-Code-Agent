# LCM lexical search ranking

`lcm_grep` accepts `sort=recency|relevance|hybrid`; default recency keeps the
existing descending sequence order. Relevance and hybrid adapt Hermes-LCM
revision `8d1b1e6d3d63f5fc7b209e8d7ec1dc9b814f2e54` (MIT), specifically
`tools.py::_combined_result_sort_key`, the hybrid summary override, and
`store.py` / `dag.py` directness adjustments and LIKE scoring.

Source: https://github.com/stephenschoettler/hermes-lcm/tree/8d1b1e6d3d63f5fc7b209e8d7ec1dc9b814f2e54
License: ../THIRD_PARTY_NOTICES/hermes-lcm-LICENSE.txt

## Preserved formulas

- FTS5 BM25 rank, lower is better. For precise queries (one term, or one
  phrase with at most two terms), subtract `max(directness, 0) * 3e-7`
  for messages, `* 2e-7` for summaries.
- Directness uses the existing upstream-derived `compute_directness_score`.
  JSON-looking tool-role messages receive the upstream -4 penalty.
- LIKE rank is negative term occurrence count, collapsed to one per term
  for risky ASCII queries. This is lexical fallback scoring, not BM25.
- Hybrid: `age_hours=max(0,(now-timestamp)/3600)`;
  `blended=rank/(1+age_hours*0.001)` ascending.
- Summary override: directness >= highest message directness + 8.
- Combined relevance key: tier, rank, negative effective directness, role,
  negative timestamp, type. Hybrid prepends negative override and substitutes
  blended for rank. Effective summary directness is multiplied by 0.8.
- Tuple tie-breaks require equality, not approximately similar scores.

## Local integration boundaries

This ports ranking rules, not the full upstream storage/retrieval engine.
Local FTS remains a unified record index (upstream separates messages and
summaries), so corpus-dependent BM25 values need not equal upstream values.
All matching visible records are ranked before limiting; upstream has bounded
per-source candidate fetches. This can cost more on very large histories.
Branch visibility and record-type filtering apply before final ranking.
Artifacts here are SQLite history records, not upstream externalized-file
search results. Embeddings, RRF and cross-session recall are not included.

New records have persisted creation times. Additive migration leaves old
timestamps NULL (unknown); the upstream missing-time fallback is zero. No
historical date is invented. Summary latest_at tracks source time when known.
One fixed clock reading is used per ranking operation for stable comparisons.

Example: `lcm_grep({"query":"timeout","sort":"hybrid","limit":10})`.
