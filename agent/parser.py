import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal


class ParseError(ValueError):
    """模型返回的数据缺少必要字段，或字段格式不正确。"""


@dataclass
class ToolCall:
    """一次工具调用；id 用于稍后把工具结果回传给模型。"""

    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass
class ParsedResponse:
    """解析结果：思考、工具调用、最终回答三种状态之一。"""

    kind: Literal["thought", "tool_call", "final"]
    thought: str | None = None
    tool_calls: List[ToolCall] = field(default_factory=list)
    final_answer: str | None = None
    content: str | None = None

"""
{
  "role": "assistant",
  "content": null,
  "tool_calls": [
    {
      "id": "call_001",
      "type": "function",
      "function": {
        "name": "calculator",
        "arguments": "{\"expression\":\"(1+2)*5\"}"
      }
    }
  ]
}

"""


# 解析单条工具调用，并把 JSON 参数字符串转成字典。
def _parse_tool_call(data: Any) -> ToolCall:
    if not isinstance(data, dict) or data.get("type") != "function":
        raise ParseError("工具调用必须是 function 类型")

    call_id = data.get("id")
    function = data.get("function")
    if not isinstance(call_id, str) or not call_id:
        raise ParseError("工具调用缺少 id")
    if not isinstance(function, dict):
        raise ParseError("工具调用缺少 function")
    # 获取工具名
    name = function.get("name")
    if not isinstance(name, str) or not name:
        raise ParseError("工具调用缺少名称")
    # 获取参数
    arguments = function.get("arguments")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError as exc:
            raise ParseError(f"工具 {name} 的参数不是有效 JSON") from exc
    if not isinstance(arguments, dict):
        raise ParseError(f"工具 {name} 的参数必须是 JSON 对象")
    try:
        json.dumps(arguments, allow_nan=False)
    except (ValueError, TypeError) as exc:
        raise ParseError(f"工具 {name} 的参数包含无效 JSON 值") from exc

    return ToolCall(id=call_id, name=name, arguments=arguments)


# 读取模型的第一条回复，判断它是工具调用、最终回答还是只有思考文本。
def parse_response(response: Dict[str, Any]) -> ParsedResponse:
    if not isinstance(response, dict):
        raise ParseError("模型返回值必须是对象")

    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ParseError("模型返回值缺少 choices")
    if not isinstance(choices[0], dict):
        raise ParseError("choices[0] 格式错误")
    if choices[0].get("finish_reason") in {"length", "content_filter"}:
        raise ParseError("模型回复被截断或拦截，不能作为完整结果执行")
    # 获取 message第一条消息，里面可能包含思考、工具调用或最终回答。
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise ParseError("模型返回值缺少 message")

    content = message.get("content")
    if content is not None and not isinstance(content, str):
        raise ParseError("message.content 必须是文本或 null")

    # 只有模型显式提供思考字段时才保留；普通回答不会被当作思考。
    thought = message.get("reasoning_content") or message.get("reasoning")
    if thought is not None and not isinstance(thought, str):
        raise ParseError("思考字段必须是文本")
    if isinstance(thought, str):
        thought = thought if thought.strip() else None

    raw_calls = message.get("tool_calls")
    if raw_calls is not None:
        if not isinstance(raw_calls, list):
            raise ParseError("message.tool_calls 必须是列表")
        if raw_calls:
            calls = [_parse_tool_call(item) for item in raw_calls]
            if len({call.id for call in calls}) != len(calls):
                raise ParseError("同一条回复中工具调用 ID 不能重复")
            return ParsedResponse(kind="tool_call", thought=thought,
                                  tool_calls=calls, content=content)

    if content and content.strip():
        return ParsedResponse(kind="final", thought=thought,
                              final_answer=content, content=content)
    if thought:
        return ParsedResponse(kind="thought", thought=thought)
    raise ParseError("模型没有返回工具调用、回答或思考文本")
