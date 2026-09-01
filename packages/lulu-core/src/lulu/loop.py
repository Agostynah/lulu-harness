"""loop.py: the agent turn loop.

Ties ModelClient (reasons), ToolRegistry (acts), PermissionChecker (gates
every side-effecting act), and -- when configured -- MemoryStore (recalls
and records) into the actual reason -> act -> observe -> repeat cycle.
Permission checks happen HERE, not inside ToolRegistry.dispatch() --
dispatch never decides whether a call is allowed, only how to execute one
that already cleared permissions.

Memory is optional (memory=None is a valid, fully supported mode -- e.g.
a one-shot task with no need for recall) but when configured this is the
one piece of code that actually calls MemoryStore.search()/.write()
during a real run. Without it, ContextAssembler and MemoryRouter would be
tested, working components nothing ever calls outside evals/dbpedia.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

from lulu.attention import Decision
from lulu.llm.client import Message, ModelClient, ToolCall, ToolResult, Usage
from lulu.permissions import PermissionChecker
from lulu.tools.registry import ToolRegistry

if TYPE_CHECKING:
    from lulu_router.trace import RoutingTrace

    from lulu.memory import MemoryStore

DEFAULT_MAX_ITERATIONS = 50

# (tool_name, arguments, reason) -> approved?
AskHuman = Callable[[str, dict, str], bool]


@dataclass
class TurnResult:
    messages: list[Message]
    iterations: int
    stopped_reason: str  # "final_text" | "max_iterations"
    # One Usage per model call this turn made, in order -- lines up 1:1
    # with the assistant-role messages appended to `messages` during this
    # turn, since each iteration appends exactly one. session.py uses this
    # to attribute real token counts per assistant message, not just per
    # turn, without ModelResponse.usage having to live inside Message
    # itself (which stays a pure provider-agnostic conversation shape).
    usages: list[Usage]
    # What the memory router did this turn -- None when no MemoryStore was
    # configured, or when it was but retrieved nothing. This is what
    # /trace, /cost, and the inspector UI (day 5) actually render.
    trace: "RoutingTrace | None" = None


def _last_assistant_text(messages: list[Message]) -> str:
    for message in reversed(messages):
        if message.role == "assistant" and message.content:
            return message.content
    return ""


class AgentLoop:
    def __init__(
        self,
        model: ModelClient,
        tools: ToolRegistry,
        permissions: PermissionChecker,
        ask_human: AskHuman,
        system: str = "",
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        memory: "MemoryStore | None" = None,
        memory_scope: str | None = None,
    ) -> None:
        self.model = model
        self.tools = tools
        self.permissions = permissions
        self.ask_human = ask_human
        self.system = system
        self.max_iterations = max_iterations
        self.memory = memory
        self.memory_scope = memory_scope

    def run_turn(self, history: list[Message], user_input: str) -> TurnResult:
        messages = [*history, Message(role="user", content=user_input)]
        tool_specs = self.tools.specs()
        usages: list[Usage] = []

        trace: "RoutingTrace | None" = None
        # Kept separate from self.system, not concatenated into it: memory
        # results differ on every turn, so folding them into `system`
        # would put volatile content inside what AnthropicClient treats as
        # the cacheable prefix, breaking the cache on every single turn
        # instead of reusing it. See ModelClient's docstring.
        context = ""
        if self.memory is not None:
            assembled = self.memory.search(user_input, scope=self.memory_scope)
            trace = assembled.trace
            context = assembled.text

        result: TurnResult | None = None
        for i in range(self.max_iterations):
            response = self.model.complete(messages, tool_specs, system=self.system, context=context)
            messages.append(response.message)
            usages.append(response.usage)

            if not response.message.tool_calls:
                result = TurnResult(
                    messages=messages, iterations=i + 1, stopped_reason="final_text", usages=usages, trace=trace
                )
                break

            tool_results = [self._execute(call) for call in response.message.tool_calls]
            messages.append(Message(role="user", tool_results=tool_results))
        else:
            # A model that never stops calling tools (buggy, adversarial
            # prompt, or a tool that keeps looking like it needs a
            # follow-up) must not be able to hang the harness forever.
            result = TurnResult(
                messages=messages,
                iterations=self.max_iterations,
                stopped_reason="max_iterations",
                usages=usages,
                trace=trace,
            )

        if self.memory is not None:
            final_text = _last_assistant_text(messages)
            if final_text:
                self.memory.write(
                    f"User: {user_input}\nLulu: {final_text}",
                    shard="episodic",
                    scope=self.memory_scope,
                )

        return result

    def _execute(self, call: ToolCall) -> ToolResult:
        permission = self.permissions.check(call.name, call.arguments)

        if permission.decision == Decision.DENY:
            self.permissions.log(call.name, call.arguments, permission, outcome="auto_denied")
            return ToolResult(
                tool_call_id=call.id, content=f"denied by policy: {permission.reason}", is_error=True
            )

        if permission.decision == Decision.ASK:
            approved = self.ask_human(call.name, call.arguments, permission.reason)
            self.permissions.log(
                call.name, call.arguments, permission, outcome="approved" if approved else "denied"
            )
            if not approved:
                return ToolResult(tool_call_id=call.id, content="denied by user", is_error=True)
            return self.tools.dispatch(call)

        # ALLOW
        self.permissions.log(call.name, call.arguments, permission, outcome="auto_allowed")
        return self.tools.dispatch(call)
