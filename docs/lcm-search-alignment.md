# LCM query alignment

Query normalization and fallback detection use hermes-lcm `search_query.py`,
revision `8d1b1e6d3d63f5fc7b209e8d7ec1dc9b814f2e54` (MIT; notice beside the module).
Only formatting and typing modernization have been applied to that helper.

Agent input is literal text, not trusted FTS syntax. Balanced quoted phrases are
preserved; punctuation is sanitized and bare uppercase boolean operators are
neutralized. FTS uses the sanitized query directly, not an OR of regex tokens.
CJK, emoji, unsupported queries and FTS operational errors use escaped LIKE terms.
An empty FTS result does not trigger broader substring matching.

The database adapter retains local branch visibility and record-type filtering.
Results use descending sequence order as local recency. This is not a wholesale
copy of the upstream timestamp/role-based ranking, hybrid sorting or session schema.
Original records and summary sources remain unchanged.
