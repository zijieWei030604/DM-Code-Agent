# Structured multi-tool turns

DeepSeek Chat/SSE and OpenAI Responses retain native call ids, normalize calls
for dispatch, and project canonical assistant/tool history back to the provider
wire protocol. DeepSeek receives assistant.tool_calls plus role=tool messages;
OpenAI receives function_call/function_call_output items. Prompt-only providers
receive a text projection without unsupported tool roles. This is an extension
of the existing providers, not native tool support for every provider.

## Structured history and pairing

Every native result carries tool_call_id, tool_name, status, is_error and synthetic.
Structured ToolResult details (error code, exit code, changed files, check scope,
metadata) are retained in the private history and checkpoint. Provider adapters
validate pairing before issuing a model request. Repeated provider ids are repaired
without dropping siblings. Original normalized response content remains available
under the existing trace/checkpoint privacy policy.

LCM persists native fields alongside message bodies and protects the whole
assistant-call/result group from partial compaction. Native messages retain the
model_response/tool_result recall categories. JSONL message_fields allow session
reconstruction to retain ids and outcomes. Public trace hides assistant argument
payloads unless full LLM capture is enabled; checkpoint remains private.

## Scheduling and policies

Tools declare concurrency="shared" or "exclusive", optionally via a function of
prepared arguments. Undeclared tools default to exclusive. Shared execution also
requires read_only=True; writes never become concurrent just from a shared label.
The five audited builtin file lookup tools are shared. Stateful tools such as
skill loading, memory queries, LSP sessions, MCP wrappers, plan updates and task
management remain exclusive unless deliberately configured otherwise.

Dispatch uses bounded groups of at most four shared calls. Exclusive tools wait
for all prior calls and lifecycle commits; subsequent calls wait for the exclusive
tool. Argument-dependent policy exceptions fall back to exclusive execution.
New skill tools are looked up when their slot is prepared.

ToolInvoker has three phases: prepare (policies, guards and intent), execute
(tool body), commit (output bounds, execution facts, post-hooks and journal end).
Only bodies run in worker threads. Hooks, evidence, shared metadata and ordered
history commits stay on the coordinator. Per-call start/end trace events reflect
settlement; model-visible results and checkpoints stay in source call order.

## Errors, budgets and completion

Ordinary failures, unknown tools, invalid arguments and policy blocks produce
individual error results. Independent siblings continue. A tool exception does
not erase successful siblings or imply rollback. Dependent actions whose arguments
or necessity depend on an earlier result belong in the next model turn.

A turn accepts 1-32 calls; each committed call consumes one execution step.
finish/task_complete/completes_task tools must be submitted alone and still pass
the normal finish gate. SSE calls execute only after the full turn terminates.
Malformed argument calls receive error results; a truncated/rejected model turn
never authorizes tool execution, even if an argument prefix happens to parse.

## Cancellation and recovery

ReactAgent.cancel() requests cooperative cancellation. Tools may expose
cancel_runner(arguments, event). Queued calls stop; active synchronous calls must
return or honor the signal. Workers drain before the parent advances. Completed
results are preserved, and never-started siblings receive paired synthetic
cancelled results. KeyboardInterrupt and worker control exceptions preserve the
same evidence before propagating. This is not forced thread termination or an
automatic rollback of an arbitrary external side effect.

RunCheckpoint.pending_tool_calls retains calls outside the remaining step budget.
Snapshots before dispatch and after ordered commits allow resume to drain only
the unfinished tail before requesting the model again. Already-prefetched reads
may be re-read after a crash if their results were not checkpointed. Uncertain
writes remain blocked by the existing call journal. Cancelled placeholders close
the transcript; retrying those calls requires a new explicit model decision.

## Deterministic checks

The tool batch, scheduling, provider, lifecycle, memory, session and subagent tests
use fake transports and require no API keys. Tests cover shared overlap, exclusive
barriers, coordinator-only hooks, ordering, real provider request projection,
per-call errors, cancellation/draining, native finish gating, checkpoint details,
LCM safe boundaries, replay, SSE interleaving and truncated argument rejection.
