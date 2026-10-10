import tempfile
import unittest
from pathlib import Path

from agent.tools import Tool, ToolRegistry, create_default_registry


class ToolTests(unittest.TestCase):
    # Schema 的嵌套字段、枚举、数值范围和格式约束必须真正执行。
    def test_nested_schema_and_format_validation(self):
        registry = ToolRegistry()
        registry.register(Tool("check", "校验", {
            "type": "object", "required": ["data"], "additionalProperties": False,
            "properties": {"data": {"type": "object", "required": ["count", "kind"],
                "additionalProperties": False, "properties": {
                    "count": {"type": "integer", "minimum": 1, "maximum": 3},
                    "kind": {"type": "string", "enum": ["a", "b"]},
                }}},
        }, lambda data: data))
        self.assertEqual(registry.execute("check", {"data": {"count": 2, "kind": "a"}})["count"], 2)
        for data in ({"count": True, "kind": "a"}, {"count": 4, "kind": "a"},
                     {"count": 1, "kind": "x"}, {"count": 1},
                     {"count": 1, "kind": "a", "unexpected": 2}):
            with self.subTest(data=data), self.assertRaises(ValueError):
                registry.execute("check", {"data": data})

    def test_registry_exposes_function_calling_schema(self):
        registry = ToolRegistry()
        registry.register(Tool(
            name="echo", description="Echo text",
            parameters={"type": "object", "properties": {"text": {"type": "string"}},
                        "required": ["text"]},
            func=lambda text: text,
        ))
        self.assertEqual(registry.names(), ["echo"])
        self.assertEqual(registry.all_schemas(), [{
            "type": "function",
            "function": {"name": "echo", "description": "Echo text",
                         "parameters": {"type": "object", "properties": {"text": {"type": "string"}},
                                        "required": ["text"]}},
        }])
        self.assertEqual(registry.execute("echo", {"text": "hello"}), "hello")

    def test_duplicate_and_unknown_tools_are_rejected(self):
        registry = ToolRegistry()
        tool = Tool("echo", "Echo", {"type": "object", "properties": {}}, lambda: "ok")
        registry.register(tool)
        with self.assertRaises(ValueError):
            registry.register(tool)
        with self.assertRaises(KeyError):
            registry.get("missing")

    def test_calculator_evaluates_arithmetic_without_executing_code(self):
        with tempfile.TemporaryDirectory() as directory:
            registry = create_default_registry(directory)
            self.assertEqual(registry.execute("calculator", {"expression": "(2 + 3) * 4"}), 20)
            with self.assertRaises(ValueError):
                registry.execute("calculator", {"expression": "__import__('os').system('echo bad')"})

    def test_arguments_follow_declared_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            registry = create_default_registry(directory)
            for arguments in ({}, {"expression": 3}, {"expression": "1+1", "extra": True}):
                with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                    registry.execute("calculator", arguments)

    def test_search_uses_local_mock_index(self):
        with tempfile.TemporaryDirectory() as directory:
            registry = create_default_registry(directory, search_index={
                "weather": "Weather data is unavailable offline",
                "weekly report": "Weekly report template",
            })
            self.assertEqual(registry.execute("search", {"query": "weather"}),
                             [{"title": "weather", "snippet": "Weather data is unavailable offline"}])
            self.assertEqual(registry.execute("search", {"query": "unknown"}), [])

    def test_search_rejects_whitespace_only_query(self):
        with tempfile.TemporaryDirectory() as directory:
            registry = create_default_registry(directory)
            with self.assertRaises(ValueError):
                registry.execute("search", {"query": "   "})

    # 文档目录外的文件即使真实存在，也不能被 read_docs 读取。
    def test_read_docs_reads_only_workspace_text(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            docs = root / "docs"
            docs.mkdir()
            (docs / "guide.txt").write_text("Agent guide", encoding="utf-8")
            (root / "outside.txt").write_text("Outside data", encoding="utf-8")
            registry = create_default_registry(docs)
            self.assertEqual(registry.execute("read_docs", {"path": "guide.txt"}), "Agent guide")
            with self.assertRaisesRegex(ValueError, "只能读取文档目录内"):
                registry.execute("read_docs", {"path": "../outside.txt"})

    # 检查天气工具返回明确标记的模拟数据，并拒绝没有数据的城市。
    def test_weather_returns_mock_data_for_known_city(self):
        with tempfile.TemporaryDirectory() as directory:
            registry = create_default_registry(directory)
            self.assertEqual(registry.execute("weather", {"city": "北京"}), {
                "city": "北京", "condition": "晴", "temperature_c": 22, "source": "mock",
            })
            with self.assertRaises(ValueError):
                registry.execute("weather", {"city": "未知城市"})


if __name__ == "__main__":
    unittest.main()
