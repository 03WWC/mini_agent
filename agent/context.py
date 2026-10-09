"""保存单个会话的消息，并把较早的轮次压缩成简短摘要。"""

import copy
import json
from typing import Any, Callable, Dict, List

from .parser import ToolCall


Summarizer = Callable[[str, List[Dict[str, Any]], int], str]


class Context:
    """一份 Context 只属于一个会话；不同会话应使用不同实例。"""
    # 设置最多保留多少轮原始对话，以及触发压缩的大致字符数。
    def __init__(self, max_rounds: int = 10, max_chars: int = 12000):
        if max_rounds < 1 or max_chars < 100:
            raise ValueError("max_rounds 至少为 1，max_chars 至少为 100")
        self.max_rounds = max_rounds   #保留多轮对话
        self.max_chars = max_chars
        self.messages: List[Dict[str, Any]] = []
        self.summary = ""

    # 记录用户发起的新一轮对话。
    def add_user(self, content: str) -> None:
        if not isinstance(content, str) or not content.strip():
            raise ValueError("用户消息不能为空")
        self.messages.append({"role": "user", "content": content})

    # 记录助手的回答、工具调用和模型返回的思考文本。
    def add_assistant(self, content: str | None = None,
                      tool_calls: List[ToolCall] | None = None,
                      reasoning_content: str | None = None) -> None:
        if not any(message["role"] == "user" for message in self.messages):
            raise ValueError("请先添加用户消息")
        if reasoning_content is not None and not isinstance(reasoning_content, str):
            raise ValueError("思考内容必须是文本")
        if not content and not tool_calls and not reasoning_content:
            raise ValueError("助手消息不能同时缺少文本、工具调用和思考内容")

        message: Dict[str, Any] = {"role": "assistant", "content": content}
        if reasoning_content:
            message["reasoning_content"] = reasoning_content
        if tool_calls:
            # 保留调用 id。LLM 接口要求后续工具结果通过同一个 id 对应回来。
            message["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(call.arguments, ensure_ascii=False),
                    },
                }
                for call in tool_calls
            ]
        self.messages.append(message)

    # 记录工具返回值，并检查它确实对应一个尚未收到结果的调用。
    def add_tool_result(self, call_id: str, result: Any) -> None:
        pending_ids = set()
        for message in self.messages:
            if message["role"] == "assistant":
                for call in message.get("tool_calls", []):
                    # 记录已发起，还没有返回结果的工具调用 id。
                    pending_ids.add(call["id"])
            elif message["role"] == "tool":
                # 看到工具结果时， 把对应 ID 移出集合
                pending_ids.discard(message["tool_call_id"])

        # 避免给一个不存在的调用塞结果，或给同一次调用重复塞结果。
        if call_id not in pending_ids:
            raise ValueError(f"没有等待结果的工具调用: {call_id}")

        content = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
        self.messages.append({"role": "tool", "tool_call_id": call_id, "content": content})

    # 按用户消息划分轮次，找到每一轮在消息列表中的起点。
    def _round_starts(self) -> List[int]:
        return [index for index, message in enumerate(self.messages)
                if message["role"] == "user"]

    # 超过轮次或字符限制时，让模型概括旧轮次，保留最新一轮原文。
    def compact(self, summarizer: Summarizer | None = None) -> None:
        starts = self._round_starts()
        cutoff = 0
        while len(starts) > 1:
            size = len(json.dumps(self.messages[cutoff:], ensure_ascii=False)) + len(self.summary)
            if len(starts) <= self.max_rounds and size <= self.max_chars:
                break
            cutoff = starts[1]
            starts = starts[1:]

        if cutoff == 0:
            return
        if summarizer is None:
            raise ValueError("上下文需要压缩，请提供大模型摘要函数")

        # 先得到摘要，再替换旧消息；模型失败时不丢失原始对话。
        limit = self.max_chars // 2
        summary = summarizer(self.summary, copy.deepcopy(self.messages[:cutoff]), limit)
        if not isinstance(summary, str) or not summary.strip():
            raise ValueError("大模型没有返回有效摘要")
        self.summary = summary.strip()[-limit:]
        self.messages = self.messages[cutoff:]

    # 返回可直接交给 LLM 的消息副本；摘要放在原始消息之前。
    def for_llm(self, summarizer: Summarizer | None = None) -> List[Dict[str, Any]]:
        self.compact(summarizer)
        messages = copy.deepcopy(self.messages)
        if self.summary:
            return [{"role": "system", "content": "先前对话摘要：\n" + self.summary}] + messages
        return messages

    # 导出可序列化的数据，供后续的 store.py 保存。
    def to_dict(self) -> Dict[str, Any]:
        return {"messages": copy.deepcopy(self.messages), "summary": self.summary}

    # 从 store.py 读出的数据恢复一份独立的 Context。
    @classmethod
    def from_dict(cls, data: Dict[str, Any], max_rounds: int = 10,
                  max_chars: int = 12000) -> "Context":
        if not isinstance(data, dict):
            raise ValueError("Context 数据必须是对象")
        messages = data.get("messages")
        summary = data.get("summary")
        if not isinstance(messages, list) or not isinstance(summary, str):
            raise ValueError("Context 数据格式错误")
        if any(not isinstance(message, dict) or
               message.get("role") not in {"user", "assistant", "tool"}
               for message in messages):
            raise ValueError("Context 消息格式错误")

        context = cls(max_rounds=max_rounds, max_chars=max_chars)
        context.messages = copy.deepcopy(messages)
        context.summary = summary
        return context
