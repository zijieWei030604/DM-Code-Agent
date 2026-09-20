"""失败后的重规划：失败签名去重、预算护栏与新计划生成。

失败签名与重规划强耦合——「这次失败和上次是不是同一个」直接决定 adaptive 策略
走哪条分支，所以两者放在同一个模块里，而不是拆到观察判定那边。

``try_replan`` 不直接改对话历史：新计划生效时它把要追加的恢复提示放进
``ReplanOutcome.history_note``，由 agent 自己追加。原因是
``conversation_history`` 会被 ``run()`` / resume / ``reset_conversation``
整体替换，协作者持有它的引用迟早会指向旧列表。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .planner import PlanStep, ReplanSignal, TaskPlanner

# 失败签名里保留的观察文本长度。刻意短于截断阈值，
# 让"同一个失败"在观察被截断前后得到同一个签名。
FAILURE_SIGNATURE_OBSERVATION_CHARS = 160


@dataclass(frozen=True)
class FailureContext:
    """触发重规划的那次失败。"""

    observation: str
    action: str = ""
    step_number: int | None = None
    error_kind: str | None = None
    tool_status: str = ""
    verification: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class ReplanOutcome:
    """重规划的结果。

    ``plan`` 要么是新计划，要么是原样退回的旧计划；``history_note`` 非空时，
    调用方需要把它作为一条 user 消息追加进对话历史。
    """

    plan: list[PlanStep] = field(default_factory=list)
    history_note: str | None = None


def failure_signature(failure: FailureContext) -> str:
    """把一次失败压成可比对的签名：动作 | 类别 | 观察摘要。"""
    compact_observation = " ".join(str(failure.observation or "").split())[
        :FAILURE_SIGNATURE_OBSERVATION_CHARS
    ]
    return "|".join(
        [
            str(failure.action or ""),
            str(failure.error_kind or "unknown"),
            compact_observation,
        ]
    )


class ReplanCoordinator:
    """决定是否重规划、生成新计划，并把过程记进 metadata 与 trace。"""

    def __init__(
        self,
        *,
        planner: TaskPlanner | None,
        policy: Any,
        trace_writer: Any | None = None,
        adaptive: bool = False,
        max_replans: int = -1,
    ) -> None:
        self.planner = planner
        self.policy = policy
        self.trace_writer = trace_writer
        self.adaptive = adaptive
        self.max_replans = max_replans

    def try_replan(
        self,
        task: str,
        plan: list[PlanStep],
        failure: FailureContext,
        metadata: dict[str, Any],
        *,
        default_budget: int,
    ) -> ReplanOutcome:
        """按当前策略决定是否重规划，返回新计划（或原计划）与要追加的历史提示。"""
        disposition = failure_disposition(failure)
        metadata.setdefault("failure_disposition_counts", {})[disposition] = int(
            metadata.setdefault("failure_disposition_counts", {}).get(disposition, 0)
        ) + 1
        repeated_failure, repeated_payload = self.record_failure_signature(failure, metadata)
        # Default mode is deliberately conservative: the ReAct loop handles local tool
        # recovery, while the planner is reserved for direct evidence that the current
        # patch is wrong. Repeated generic failures become an adaptive-policy concern.
        if not self.adaptive and disposition != "contradicted":
            metadata["replan_skipped_count"] += 1
            metadata["replan_suppressed_count"] = int(
                metadata.get("replan_suppressed_count", 0)
            ) + 1
            self._record_trigger(
                failure,
                disposition=disposition,
                should_replan=False,
                repeated_failure=repeated_failure,
                repeated_payload=repeated_payload,
            )
            return ReplanOutcome(plan=plan)

        completed_steps = list(plan)
        signal: ReplanSignal | None = None
        decision = None
        if self.adaptive:
            signal, decision = self._decide_adaptive(
                failure, metadata, repeated_failure=repeated_failure,
                repeated_failure_payload=repeated_payload,
            )
            if not decision.should_replan:
                metadata["replan_skipped_count"] += 1
                if decision.strategy == "replan_budget_exhausted":
                    metadata["replan_maxed_count"] += 1
                return ReplanOutcome(plan=plan)
        elif self._default_budget_exhausted(failure, metadata, default_budget=1):
            return ReplanOutcome(plan=plan)

        try:
            new_plan = (
                self.planner.replan(
                    task,
                    completed_steps,
                    failure.observation,
                    error_signal=signal,
                    disposition=disposition,
                )
                if self.planner
                else []
            )
        except Exception as exc:
            metadata["failure_reason"] = f"Replan failed: {exc}"
            return ReplanOutcome(plan=plan)

        if not new_plan:
            return ReplanOutcome(plan=plan)

        metadata["replan_count"] += 1
        metadata["plan_revision_count"] = int(metadata.get("plan_revision_count", 0)) + 1
        self._record_trigger(
            failure,
            disposition=disposition,
            should_replan=True,
            repeated_failure=repeated_failure,
            repeated_payload=repeated_payload,
        )
        if self.trace_writer:
            self.trace_writer.record_replan(
                reason=failure.observation,
                steps=new_plan,
                strategy=decision.strategy if decision else "",
                signal=signal.to_dict() if signal else None,
            )
        return ReplanOutcome(
            plan=new_plan,
            history_note=(
                "Recovery: execution plan was regenerated after failure.\n"
                f"Failure classification: {disposition}.\n"
                f"Failure observation: {failure.observation}\n"
                "Updated remaining plan:\n"
                + "\n".join(
                    f"- {step.phase}: {step.goal} "
                    f"(evidence: {step.completion_evidence})"
                    for step in new_plan
                    if step.status != "satisfied"
                )
            ),
        )

    def _decide_adaptive(
        self,
        failure: FailureContext,
        metadata: dict[str, Any],
        *,
        repeated_failure: bool,
        repeated_failure_payload: dict[str, Any] | None,
    ) -> tuple[ReplanSignal, Any]:
        """adaptive 路径：分类失败信号、查重复失败、落决策与 trace。"""
        signal = self.policy.classify(
            failure.observation,
            action=failure.action,
            step_number=failure.step_number,
            error_kind=failure.error_kind,
        )
        decision = self.policy.decide(
            signal,
            replan_count=int(metadata.get("replan_count", 0)),
            max_replans=self.max_replans,
            repeated_failure=repeated_failure,
        )
        metadata["replan_decision_count"] += 1
        metadata["replan_strategy"] = decision.strategy
        strategy_counts = metadata.setdefault("replan_strategy_counts", {})
        strategy_counts[decision.strategy] = strategy_counts.get(decision.strategy, 0) + 1
        metadata.setdefault("replan_signals", []).append(decision.to_dict())
        if self.trace_writer:
            payload = {
                "step_number": failure.step_number,
                "action": failure.action,
                "repeated_failure": repeated_failure,
                **decision.to_dict(),
            }
            if repeated_failure_payload:
                payload["repeated_failure_details"] = repeated_failure_payload
            self.trace_writer.record("replan_decision", payload)
        return signal, decision

    def _record_trigger(
        self,
        failure: FailureContext,
        *,
        disposition: str,
        should_replan: bool,
        repeated_failure: bool,
        repeated_payload: dict[str, Any] | None,
    ) -> None:
        if not self.trace_writer:
            return
        payload: dict[str, Any] = {
            "step_number": failure.step_number,
            "action": failure.action,
            "disposition": disposition,
            "should_replan": should_replan,
            "scope": "phase_update" if should_replan else "none",
            "repeated_failure": repeated_failure,
        }
        if repeated_payload:
            payload["repeated_failure_details"] = repeated_payload
        self.trace_writer.record("replan_trigger", payload)

    def _default_budget_exhausted(
        self, failure: FailureContext, metadata: dict[str, Any], *, default_budget: int
    ) -> bool:
        """非 adaptive 默认路径的成本护栏。

        这条路径此前没有任何预算：每个失败观察都会触发一次完整的 planner LLM 调用。
        """
        budget = self.max_replans if self.max_replans >= 0 else default_budget
        if int(metadata.get("replan_count", 0)) < budget:
            return False
        metadata["replan_skipped_count"] += 1
        metadata["replan_budget_exhausted_count"] = (
            int(metadata.get("replan_budget_exhausted_count", 0)) + 1
        )
        if self.trace_writer:
            self.trace_writer.record(
                "replan_decision",
                {
                    "step_number": failure.step_number,
                    "action": failure.action,
                    "should_replan": False,
                    "strategy": "replan_budget_exhausted",
                    "reason": (
                        "Default replan budget exhausted "
                        f"({metadata.get('replan_count', 0)}/{budget})."
                    ),
                },
            )
        return True

    @staticmethod
    def record_failure_signature(
        failure: FailureContext, metadata: dict[str, Any]
    ) -> tuple[bool, dict[str, Any] | None]:
        """记录本次失败签名，返回 (是否与上次重复, 重复失败详情)。"""
        signature = failure_signature(failure)
        previous = str(metadata.get("last_failure_signature") or "")
        metadata["last_failure_signature"] = signature
        if not signature or signature != previous:
            return False, None

        payload = {
            "step_number": failure.step_number,
            "action": failure.action,
            "kind": failure.error_kind or "unknown",
            "signature": signature,
        }
        metadata["repeated_failure_count"] = int(metadata.get("repeated_failure_count", 0)) + 1
        metadata.setdefault("repeated_failures", []).append(payload)
        return True, payload


def failure_disposition(failure: FailureContext) -> str:
    """Classify whether a failure disproves the patch or only blocks verification."""
    verification = failure.verification
    if isinstance(verification, Mapping):
        execution = str(verification.get("execution_status") or "")
        outcome = str(verification.get("outcome") or "")
        kind = str(verification.get("failure_kind") or failure.error_kind or "")
        scope = tuple(str(item) for item in verification.get("scope") or ())
        if execution in {"invalid", "unavailable"} or kind in {
            "invalid_target",
            "unknown_result",
            "command_unavailable",
            "dependency_unavailable",
            "timeout",
        }:
            return "unavailable"
        if outcome in {"failed", "error"}:
            if kind == "assertion_failure" and _targeted_scope(scope):
                return "contradicted"
            return "inconclusive"
    if failure.error_kind in {"invalid_arguments", "parse_error", "unknown_tool"}:
        return "tool_failure"
    return "tool_failure"


def _targeted_scope(scope: tuple[str, ...]) -> bool:
    if not scope:
        return False
    return all("::" in item or item.replace("\\", "/").endswith(".py") for item in scope)
