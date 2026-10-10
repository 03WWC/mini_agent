"""连接模型、工具和会话，运行一次用户对话。"""

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List
from uuid import uuid4

from .parser import ParseError, parse_response
from .session import SessionManager
from .tools import ToolRegistry


@dataclass
class ToolTrace:
    """记录一次工具调用，方便查看 Agent 做过什么。"""

    step: int
    name: str
    arguments: Dict[str, Any]
    status: str
    result: Any


@dataclass
class RunResult:
    """一次对话的回答、会话 ID 和工具执行记录。"""

    answer: str
    session_id: str
    trace: List[ToolTrace] = field(default_factory=list)


class AgentRunError(RuntimeError):
    """Agent 无法在限制内完成本次对话。"""

    # 报错时仍提供会话 ID 和已经执行的工具记录。
    def __init__(self, message: str, session_id: str, trace: List[ToolTrace]):
        super().__init__(message)
        self.session_id = session_id
        self.trace = trace


class AgentRuntime:
    """每次 run 处理一条用户消息，按需反复调用模型和工具。"""

    # 保存模型、工具和会话管理器，并限制每次对话的模型调用次数。
    def __init__(self, client: Any, tools: ToolRegistry,
                 sessions: SessionManager, max_steps: int = 8):
        if max_steps < 1:
            raise ValueError("max_steps 至少为 1")
        self.client = client
        self.tools = tools
        self.sessions = sessions
        self.max_steps = max_steps

    # 单独请求模型概括旧对话；不把模型的思考文本放进摘要请求。
    def _summarize(self, old_summary: str, old_messages: List[Dict[str, Any]],
                   limit: int) -> str:
        messages = [
            {key: value for key, value in message.items() if key != "reasoning_content"}
            for message in old_messages
        ]
        prompt = (
            f"请把旧对话压缩成不超过 {limit} 字的中文摘要，只输出摘要。"
            "保留用户目标、已确认的事实、工具结果和未完成事项；不要编造。\n"
            f"已有摘要：{old_summary or '无'}\n"
            f"旧对话：{json.dumps(messages, ensure_ascii=False)}"
        )
        response = self.client.complete([{"role": "user", "content": prompt}], None)
        parsed = parse_response(response)
        if parsed.kind != "final":
            raise ValueError("大模型没有返回对话摘要")
        return parsed.final_answer

    # 恢复中断前尚未回填的工具结果；未知结果只做标记，不自动重放写操作。
    def _recover_tools(self, session) -> None:
        pending = {}
        for message in session.context.messages:
            for call in message.get("tool_calls", []):
                pending[call["id"]] = call
            if message["role"] == "tool":
                pending.pop(message["tool_call_id"], None)
        if not pending:
            return
        records = {
            item["call_id"]: item
            for item in self.sessions.store.tool_history(session.user_id, session.session_id)
            if item["run_id"] == session.metadata.get("run_id")
        }
        for call_id in pending:
            record = records.get(call_id)
            if record and record["status"] in {"ok", "error", "unknown"}:
                result = record["result"]
            elif record:
                result = "上次执行中断，结果未知。不要自动重复写入，请先查询当前业务状态。"
                self.sessions.store.finish_tool(record["id"], "unknown", result)
            else:
                result = "上次执行中断，此调用尚未开始执行。"
            session.context.add_tool_result(call_id, result)

    # 处理一条用户消息；模型可直接回答，也可调用工具后继续思考。
    def run(self, user_id: str, text: str,
            session_id: str | None = None) -> RunResult:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("用户消息不能为空")

        session = (self.sessions.open(user_id, session_id) if session_id is not None
                   else self.sessions.create(user_id))
        tools = self.tools.for_session(self.sessions.store, user_id, session.session_id)
        self._recover_tools(session)
        context = session.context
        # Step one：接收用户输入，加入当前会话的上下文。
        context.add_user(text)
        trace: List[ToolTrace] = []
        run_id = uuid4().hex
        session.metadata.update(status="running", current_task=text, run_id=run_id,
                                step=0, last_error=None)
        self.sessions.save(session)

        try:
            for step in range(1, self.max_steps + 1):
                session.metadata["step"] = step
                # Step two：把上下文和工具说明交给模型，判断直接回复还是调用工具。
                response = self.client.complete(
                    context.for_llm(self._summarize), tools.all_schemas()
                )
                try:
                    parsed = parse_response(response)
                except ParseError as exc:
                    raise AgentRunError(f"模型回复无法解析: {exc}",
                                        session.session_id, trace) from exc

                if parsed.kind == "final":
                    context.add_assistant(parsed.final_answer,
                                          reasoning_content=parsed.thought)
                    session.metadata["status"] = "completed"
                    return RunResult(parsed.final_answer, session.session_id, trace)

                if parsed.kind == "thought":
                    context.add_assistant(reasoning_content=parsed.thought)
                    continue

                context.add_assistant(parsed.content, parsed.tool_calls,
                                      reasoning_content=parsed.thought)
                self.sessions.save(session)
                # Step three：执行模型选择的工具，记录结果和错误。
                for call in parsed.tool_calls:
                    execution_id = self.sessions.store.start_tool(
                        user_id, session.session_id, run_id, step, call.id, call.name, call.arguments)
                    try:
                        result = tools.execute(call.name, call.arguments)
                        json.dumps(result, ensure_ascii=False, allow_nan=False)
                        status = "ok"
                    except Exception as exc:
                        result = f"工具执行失败: {exc}"
                        status = "error"

                    trace.append(ToolTrace(step, call.name, call.arguments,
                                           status, result))
                    self.sessions.store.finish_tool(execution_id, status, result)
                    context.add_tool_result(call.id, result)
                    self.sessions.save(session)

                # Step four：带着工具结果继续循环，由模型决定继续调用工具还是返回答案。

            raise AgentRunError(f"达到最大模型调用次数: {self.max_steps}",
                                session.session_id, trace)
        except Exception as exc:
            session.metadata.update(status="failed", last_error=str(exc))
            if isinstance(exc, AgentRunError):
                raise
            raise AgentRunError(str(exc), session.session_id, trace) from exc
        finally:
            # 即使模型或工具失败，也保存本次已经产生的对话。
            self.sessions.save(session)
