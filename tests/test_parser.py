import unittest

from agent.parser import ParseError, parse_response


class ParserTests(unittest.TestCase):
    # 检查普通文本会被识别为最终回答。
    def test_final_answer(self):
        response = {"choices": [{"message": {"role": "assistant", "content": "答案是 42"}}]}
        result = parse_response(response)
        self.assertEqual(result.kind, "final")
        self.assertEqual(result.final_answer, "答案是 42")
        self.assertEqual(result.tool_calls, [])

    # 检查一个消息中的多个工具调用都能被解析，参数字符串会转成字典。
    def test_multiple_tool_calls(self):
        response = {"choices": [{"message": {
            "role": "assistant", "content": None,
            "tool_calls": [
                {"id": "call_1", "type": "function", "function": {
                    "name": "calculator", "arguments": '{"expression":"2+2"}'}},
                {"id": "call_2", "type": "function", "function": {
                    "name": "weather", "arguments": '{"city":"北京"}'}},
            ],
        }}]}
        result = parse_response(response)
        self.assertEqual(result.kind, "tool_call")
        self.assertEqual([(call.id, call.name, call.arguments) for call in result.tool_calls], [
            ("call_1", "calculator", {"expression": "2+2"}),
            ("call_2", "weather", {"city": "北京"}),
        ])

    # 检查显式提供的思考文本可以保留，但不把它当作最终答案。
    def test_thought_only(self):
        response = {"choices": [{"message": {
            "role": "assistant", "content": None, "reasoning_content": "先查天气",
        }}]}
        result = parse_response(response)
        self.assertEqual(result.kind, "thought")
        self.assertEqual(result.thought, "先查天气")
        self.assertIsNone(result.final_answer)

    # 工具调用后的后续请求需要原样带回 reasoning_content。
    def test_reasoning_text_is_not_trimmed(self):
        response = {"choices": [{"message": {
            "role": "assistant", "content": None,
            "reasoning_content": "  先计算，再回答。\n",
            "tool_calls": [{"id": "call_1", "type": "function", "function": {
                "name": "calculator", "arguments": '{"expression":"2+2"}'}}],
        }}]}
        self.assertEqual(parse_response(response).thought, "  先计算，再回答。\n")

    # 检查工具参数不是 JSON 对象时给出解析错误。
    def test_invalid_tool_arguments(self):
        response = {"choices": [{"message": {"tool_calls": [{
            "id": "call_1", "type": "function", "function": {
                "name": "calculator", "arguments": '[1, 2]'},
        }]}}]}
        with self.assertRaises(ParseError):
            parse_response(response)

    # 检查没有选择项或没有有效内容时不能伪造一个回答。
    def test_empty_response(self):
        for response in ({"choices": []}, {"choices": [{"message": {"content": "  "}}]}):
            with self.subTest(response=response), self.assertRaises(ParseError):
                parse_response(response)


if __name__ == "__main__":
    unittest.main()
