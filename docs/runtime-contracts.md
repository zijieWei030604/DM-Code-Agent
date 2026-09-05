# Runtime contracts and recovery

The runtime retains the ReAct loop and existing extension registration API.
These contracts distinguish an operation's result, its side effects, and the
evidence supporting a completion claim.

## Tool results

`ToolResult` carries status, exit code, changed files and check scope. Built-in
execution tools and ordinary file writes register structured result runners.
Their public text functions remain available for existing callers. Third-party,
MCP and container tools returning text still use the legacy observation adapter;
they have not all been migrated to structured outcomes.

A failed syntax check can accompany a real file modification. `has_effect`
ensures the edit ledger, semantic index and verified-edit snapshots still see
that modification. Unavailable checks do not become passing evidence.

## Hooks

`EventBus.on(..., kind="observer")` receives an isolated copy of the event.
Its return value is ignored; exceptions are reported. This isolates runtime
state, but is not a security sandbox for arbitrary plugin Python code.

`kind="policy"` is supported on before-tool and before-finish hooks. Exceptions
and invalid return values block the decision. The built-in read-before-edit
guard is a policy. Existing unclassified middleware remains compatible.
Final tool arguments are checked to be an object, and tool implementations
perform their existing field validation. This is not a new full JSON Schema
validation system.

## Versioned evidence

Verification records contain a workspace fingerprint, command identity and
scope. Source/config extensions in `core/workspace_version.py` define the
fingerprint boundary; ignored directories and other file types are not covered.
This is not a fingerprint of the interpreter, installed dependencies or remote
services. Checks that modify the fingerprinted workspace cannot certify it.

Completion recomputes the fingerprint. Previous-version checks remain in the
graph but cannot verify the current version. Repeated checks with the same
identity and scope use the newest result; earlier failures remain in the audit.
Links inferred from nearby reads are explicitly marked inferred. A passing
test remains evidence for its recorded scope, not proof of every requirement.
This does not automatically schedule tests or introduce another completion gate.

## Interrupted writes and calls

Local text writes persist an intent before replacement, including the target's
before hash and expected after hash. Intent and completion records are flushed
and fsynced. File data is fsynced before atomic replacement; failed replacement
does not fall back to a non-atomic overwrite.

The write journal is located in `tempfile.gettempdir()/dm_agent_write_journal`.
Set TEMP/TMP before starting the process to choose another disk. Keep these
journals while recovering interrupted work. They contain paths/hashes, not file
contents. At the next write to a target, pending intents are reconciled with its
actual bytes. Unexpected content blocks the write without overwriting it.
Malformed journal records require inspection and are not silently discarded.

Runs with checkpoints also create `<checkpoint>.calls.jsonl`. Resume refuses
to blindly replay potentially mutating calls that are still pending or newer
than the selected checkpoint, even if their result was recorded. Pure reads
can be retried. The call journal must travel with the checkpoint. A run without
a checkpoint does not have this run-level recovery barrier.

Recovery deliberately pauses ambiguous side effects for inspection; automatic
reconstruction of missing model turns or arbitrary shell effects is not
implemented. Local file intents and checkpoint call barriers provide different
guarantees. They assume one writer per workspace/checkpoint. Neither fsync nor
these journals provide backups against disk loss or exactly-once execution of
external services. Review a blocked run and start a new task after reconciliation;
do not delete its journal to bypass the check.

## Verification

`tests/test_runtime_contracts.py` covers observer isolation, policy failure,
version invalidation, repeated checks, interrupted file writes and checkpoint
gaps. Existing tool, capability, trace and checkpoint suites cover compatibility.

Custom tools known to have no side effects can declare `Tool(..., read_only=True)`
to permit retry during recovery. Unclassified custom tools remain conservative.
