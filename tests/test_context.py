import unittest
from unittest.mock import Mock

from agent.context import Context
from agent.parser import ToolCall


class ContextTests(unittest.TestCase):
    # 检查工具调用和结果使用同一个 call_id，追问时旧消息仍在上下文中。
    def test_tool_result_and_follow_up(self):
        context = Context(max_rounds=4, max_chars=2000)
        context.add_user("2+2 等于多少？")
        context.add_assistant(tool_calls=[ToolCall("call_1", "calculator", {"expression": "2+2"})])
        context.add_tool_result("call_1", 4)
        context.add_assistant("等于 4")
        context.add_user("再加 1 呢？")

        messages = context.for_llm()
        self.assertEqual(messages[1]["tool_calls"][0]["function"]["name"], "calculator")
        self.assertEqual(messages[2], {"role": "tool", "tool_call_id": "call_1", "content": "4"})
        self.assertEqual(messages[-1], {"role": "user", "content": "再加 1 呢？"})

    # 超过轮次限制时，较早的完整对话进入摘要，最近两轮仍保持原样。
    def test_max_rounds_compacts_oldest_complete_round(self):
        context = Context(max_rounds=2, max_chars=2000)
        for number in range(1, 4):
            context.add_user(f"问题 {number}")
            context.add_assistant(f"答案 {number}")

        summarize = Mock(return_value="问题 1 的摘要")
        messages = context.for_llm(summarize)
        self.assertIn("问题 1", messages[0]["content"])
        self.assertEqual([item["content"] for item in messages if item["role"] == "user"],
                         ["问题 2", "问题 3"])
        self.assertEqual([item["content"] for item in summarize.call_args.args[1]],
                         ["问题 1", "答案 1"])

    # 字符限制会触发基础压缩，但不删掉正在进行的最新一轮。
    def test_max_chars_keeps_latest_round(self):
        context = Context(max_rounds=10, max_chars=250)
        for number in range(3):
            context.add_user(f"问题{number} " + "甲" * 70)
            context.add_assistant(f"答案{number} " + "乙" * 70)

        summarize = Mock(return_value="问题0 和问题1 的摘要")
        messages = context.for_llm(summarize)
        self.assertIn("问题0", messages[0]["content"])
        self.assertEqual(messages[-2]["content"], "问题2 " + "甲" * 70)
        self.assertEqual(messages[-1]["content"], "答案2 " + "乙" * 70)
        summarize.assert_called_once()

    # 摘要装满后保留最近压缩的轮次，不让旧摘要一直占住空间。
    def test_full_summary_keeps_recent_compacted_round(self):
        context = Context(max_rounds=1, max_chars=120)
        for number in range(8):
            context.add_user(f"question-{number}")
            context.add_assistant(f"answer-{number}")

        # 用替身模拟大模型，只检查 Context 是否采用最近的旧轮次摘要。
        def summarize(old_summary, old_messages, limit):
            return "最近压缩：" + old_messages[-2]["content"]

        messages = context.for_llm(summarize)
        self.assertIn("question-6", context.summary)
        self.assertNotIn("question-0", context.summary)
        self.assertLessEqual(len(context.summary), 60)
        self.assertEqual(messages[-2]["content"], "question-7")
        self.assertEqual(messages[-1]["content"], "answer-7")

    # 再次压缩时，要把旧摘要交给模型，避免丢掉更早的有效信息。
    def test_next_summary_receives_previous_summary(self):
        context = Context(max_rounds=1, max_chars=200)
        context.add_user("我喜欢蓝色")
        context.add_assistant("记住了")
        context.add_user("我喜欢猫")
        context.add_assistant("也记住了")
        context.for_llm(lambda old, messages, limit: "用户喜欢蓝色")

        context.add_user("总结我的喜好")
        summarize = Mock(return_value="用户喜欢蓝色和猫")
        context.for_llm(summarize)

        self.assertEqual(summarize.call_args.args[0], "用户喜欢蓝色")
        self.assertEqual(summarize.call_args.args[1][0]["content"], "我喜欢猫")
        self.assertEqual(context.summary, "用户喜欢蓝色和猫")

    # 模型摘要失败时，不能先删除原始对话。
    def test_failed_summary_keeps_original_messages(self):
        context = Context(max_rounds=1)
        context.add_user("第一轮")
        context.add_assistant("回答一")
        context.add_user("第二轮")

        with self.assertRaises(ValueError):
            context.for_llm(lambda old, messages, limit: "")

        self.assertEqual([item["content"] for item in context.messages],
                         ["第一轮", "回答一", "第二轮"])
        self.assertEqual(context.summary, "")

    # 导出后再恢复，应得到同样的对话数据，且不会共享可变对象。
    def test_snapshot_round_trip(self):
        context = Context()
        context.add_user("你好")
        context.add_assistant("你好！")
        restored = Context.from_dict(context.to_dict())
        self.assertEqual(restored.for_llm(), context.for_llm())
        restored.add_user("另一个问题")
        self.assertEqual(len(context.for_llm()), 2)

    # 没有对应工具调用的结果不能加入上下文。
    def test_rejects_unknown_tool_result(self):
        context = Context()
        context.add_user("你好")
        with self.assertRaises(ValueError):
            context.add_tool_result("missing", "结果")

    # DeepSeek 的工具调用消息需要保留完整的 reasoning_content。
    def test_assistant_reasoning_is_kept_for_follow_up_request(self):
        context = Context()
        context.add_user("算 2+2")
        context.add_assistant(tool_calls=[ToolCall("call_1", "calculator", {"expression": "2+2"})],
                              reasoning_content="  需要计算。\n")
        context.add_tool_result("call_1", 4)
        self.assertEqual(context.for_llm()[1]["reasoning_content"], "  需要计算。\n")
        self.assertEqual(Context.from_dict(context.to_dict()).for_llm()[1]["reasoning_content"],
                         "  需要计算。\n")


if __name__ == "__main__":
    unittest.main()
