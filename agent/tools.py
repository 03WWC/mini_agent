"""工具定义、注册和四个内置工具。"""

import ast
import json
import operator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping


@dataclass
class Tool:
    """一个工具需要告诉模型怎么调用，也需要提供实际执行的函数。"""

    name: str
    description: str
    parameters: Dict[str, Any]  # 参数的 JSON Schema
    func: Callable[..., Any]

    # 把工具信息整理成模型能够识别的格式。
    def to_schema(self) -> Dict[str, Any]:
        """转换成常见的 LLM function-calling 格式。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    """统一保存工具，并负责按名称调用。"""

    # 新建注册表，起初没有任何工具。
    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    # 注册工具，并防止同名工具覆盖已有工具。
    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"工具重复注册: {tool.name}")
        if tool.parameters.get("type") != "object":
            raise ValueError("工具参数的顶层 Schema 必须是 object")
        self._tools[tool.name] = tool

    # 根据名称取出工具，找不到时给出明确错误。
    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise KeyError(f"未找到工具: {name}")
        return self._tools[name]

    # 列出所有工具的 Schema，供 LLM 选择工具。
    def all_schemas(self) -> List[Dict[str, Any]]:
        return [tool.to_schema() for tool in self._tools.values()]

    # 列出已经注册的工具名称。
    def names(self) -> List[str]:
        return list(self._tools.keys())

    # 检查参数后，调用对应的工具函数。
    def execute(self, name: str, arguments: Dict[str, Any]) -> Any:
        """先检查参数，再执行工具函数。"""
        tool = self.get(name)
        _check_arguments(tool.parameters, arguments)
        try:
            return tool.func(**arguments)
        except TypeError as exc:
            raise ValueError(f"工具参数不匹配: {exc}") from exc

    # 将模型返回的 JSON 参数字符串解析后执行工具。
    def execute_json(self, name: str, arguments_json: str) -> Any:
        """有些模型会把工具参数作为 JSON 字符串返回。"""
        try:
            arguments = json.loads(arguments_json)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("工具参数不是有效 JSON") from exc
        return self.execute(name, arguments)


# 按工具的 Schema 检查必填项、额外参数、类型和字符串长度。
def _check_arguments(schema: Dict[str, Any], arguments: Any) -> None:
    """检查本项目用到的基础 JSON Schema 规则。"""
    if not isinstance(arguments, dict):
        raise ValueError("工具参数必须是 JSON 对象")

    properties = schema.get("properties", {})
    for name in schema.get("required", []):
        if name not in arguments:
            raise ValueError(f"缺少必需参数: {name}")

    if schema.get("additionalProperties") is False:
        for name in arguments:
            if name not in properties:
                raise ValueError(f"未知参数: {name}")

    # JSON 的 number 不包含布尔值；Python 中 bool 是 int 的子类，所以要单独判断。
    python_types = {
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
        "array": list,
        "object": dict,
    }
    for name, value in arguments.items():
        rule = properties.get(name, {})
        expected = rule.get("type")
        if expected == "null":
            valid_type = value is None
        elif expected in python_types:
            valid_type = isinstance(value, python_types[expected])
            if expected in ("integer", "number") and isinstance(value, bool):
                valid_type = False
        else:
            valid_type = True
        if not valid_type:
            raise ValueError(f"参数 {name} 应为 {expected}")

        if isinstance(value, str):
            if len(value) < rule.get("minLength", 0):
                raise ValueError(f"参数 {name} 太短")
            if "maxLength" in rule and len(value) > rule["maxLength"]:
                raise ValueError(f"参数 {name} 太长")


# 只允许这些算术运算。其他 AST 节点（函数调用、变量、属性访问等）会被拒绝。
_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPERATORS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


# 解析并计算算术表达式，不执行表达式里的 Python 代码。
def calculator(expression: str) -> int | float:
    """用 AST 计算算式，不使用 eval。"""
    if not isinstance(expression, str) or not expression.strip() or len(expression) > 100:
        raise ValueError("算式必须是长度为 1～100 的字符串")

    # 递归处理 AST 中允许的数字和运算节点。
    def calculate_node(node: ast.AST, depth: int = 0) -> int | float:
        if depth > 16:
            raise ValueError("算式过于复杂")

        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            if abs(node.value) > 1e100:
                raise ValueError("数字过大")
            return node.value

        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
            value = calculate_node(node.operand, depth + 1)
            return _UNARY_OPERATORS[type(node.op)](value)

        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
            left = calculate_node(node.left, depth + 1)
            right = calculate_node(node.right, depth + 1)
            if isinstance(node.op, ast.Pow) and (abs(left) > 1e6 or abs(right) > 100):
                raise ValueError("幂运算参数过大")
            try:
                result = _BINARY_OPERATORS[type(node.op)](left, right)
            except (ZeroDivisionError, OverflowError) as exc:
                raise ValueError(f"算式无法计算: {exc}") from exc
            if isinstance(result, complex) or abs(result) > 1e100:
                raise ValueError("结果过大或不是实数")
            return result

        raise ValueError("只支持数字和基本算术运算")

    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError("算式语法错误") from exc
    return calculate_node(tree.body)


# 建立默认工具表，并把文档目录和模拟搜索数据交给对应工具。
def create_default_registry(
    docs_root: str | Path,
    search_index: Mapping[str, str] | None = None,
) -> ToolRegistry:
    """创建包含 calculator、search、read_docs 和 weather 的工具注册表。"""
    root = Path(docs_root).resolve()

    # search 先用本地字典模拟。之后接入真实搜索时，只需要替换这个函数。
    index = dict(search_index) if search_index is not None else {
        "Python": "Python 是一种编程语言。",
        "Agent": "Agent 可以调用工具，并根据结果继续处理任务。",
    }

    # 在本地模拟索引中查找关键词。
    def search(query: str) -> List[Dict[str, str]]:
        keyword = query.strip().casefold()
        if not keyword:
            raise ValueError("搜索关键词不能为空")

        results = []
        for title, snippet in index.items():
            if keyword in title.casefold() or keyword in snippet.casefold():
                results.append({"title": title, "snippet": snippet})
        return results[:5]

    # 读取文档目录内的 UTF-8 文本，并限制文件大小。
    def read_docs(path: str) -> str:
        # resolve 会处理 .. 和符号链接，防止读取目录外的文件。
        target = (root / path).resolve()
        if not target.is_relative_to(root):
            raise ValueError("只能读取文档目录内的文件")
        if not target.is_file():
            raise ValueError("文档不存在")
        if target.stat().st_size > 65536:
            raise ValueError("文档不能超过 64 KiB")
        try:
            return target.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("文档必须是 UTF-8 文本") from exc

    # 这些天气是演示用固定数据，不代表实时天气。
    weather_data = {
        "北京": {"condition": "晴", "temperature_c": 22},
        "上海": {"condition": "多云", "temperature_c": 25},
        "深圳": {"condition": "小雨", "temperature_c": 28},
    }

    # 根据城市返回一条模拟天气；没有数据时明确报错。
    def weather(city: str) -> Dict[str, Any]:
        city = city.strip()
        if city not in weather_data:
            raise ValueError(f"暂无 {city} 的模拟天气数据")
        return {"city": city, **weather_data[city], "source": "mock"}


    registry = ToolRegistry()
    # 注册 calculator 工具。
    registry.register(Tool(
        name="calculator",
        description="计算算术表达式，支持 +、-、*、/、//、%、** 和括号。",
        parameters={
            "type": "object",
            "properties": {
                "expression": {"type": "string", "minLength": 1, "maxLength": 100,
                               "description": "需要计算的算式"},
            },
            "required": ["expression"],
            "additionalProperties": False,
        },
        func=calculator,
    ))
    # 注册 search、read_docs 和 weather 工具。
    registry.register(Tool(
        name="search",
        description="搜索本地模拟数据，返回标题和摘要，不访问互联网。",
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string", "minLength": 1,
                                     "description": "搜索关键词"}},
            "required": ["query"],
            "additionalProperties": False,
        },
        func=search,
    ))
    registry.register(Tool(
        name="read_docs",
        description="读取指定目录内的 UTF-8 文本文件，最大 64 KiB。",
        parameters={
            "type": "object",
            "properties": {"path": {"type": "string", "minLength": 1,
                                    "description": "相对文档目录的路径"}},
            "required": ["path"],
            "additionalProperties": False,
        },
        func=read_docs,
    ))
    registry.register(Tool(
        name="weather",
        description="查询演示用的模拟天气，只支持北京、上海、深圳；不代表实时天气。",
        parameters={
            "type": "object",
            "properties": {"city": {"type": "string", "minLength": 1,
                                    "description": "要查询的城市名称"}},
            "required": ["city"],
            "additionalProperties": False,
        },
        func=weather,
    ))
    return registry
