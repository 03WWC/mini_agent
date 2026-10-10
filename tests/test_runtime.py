import copy
import tempfile
import unittest
from pathlib import Path

from agent.runtime import AgentRunError, AgentRuntime
from agent.session import SessionManager
from agent.store import SQLiteStore
from agent.tools import create_default_registry


class FakeLLM:
    # 依次返回预设的模型消息，并记录真正收到的上下文。
    def __init__(self, messages):
        self.messages = list(messages)
        self.requests = []

    # 模拟模型返回的 Chat Completions 结构。
    def complete(self, messages, tools):
        self.requests.append((copy.deepcopy(messages), copy.deepcopy(tools)))
        message = self.messages.pop(0)
        if isinstance(message, Exception):
            raise message
        return {"choices": [{"message": message}]}


class RuntimeTests(unittest.TestCase):
    # 每个用例都使用真实的 SQLite、SessionManager 和工具注册表。
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        root = Path(self.directory.name)
        self.sessions = SessionManager(SQLiteStore(root / "sessions.db"))
        self.tools = create_default_registry(root)

    # 直接回答应保存到新会话，供同一窗口追问。
    def test_direct_answer_is_saved_and_follow_up_uses_history(self):
        llm = FakeLLM([
            {"role": "assistant", "content": "你好"},
            {"role": "assistant", "content": "刚才我说了你好"},
        ])
        runtime = AgentRuntime(llm, self.tools, self.sessions)

        first = runtime.run("user-A", "你好")
        second = runtime.run("user-A", "刚才你说了什么？", first.session_id)

        self.assertEqual(first.answer, "你好")
        self.assertEqual(second.answer, "刚才我说了你好")
        self.assertEqual(first.trace, [])
        self.assertEqual([item["content"] for item in llm.requests[1][0] if item["role"] == "user"],
                         ["你好", "刚才你说了什么？"])
        self.assertEqual(self.sessions.open("user-A", first.session_id).context.messages[-1]["content"],
                         "刚才我说了你好")

    # 工具结果和完整思考文本应回传给模型，并留下可查看的执行记录。
    def test_tool_call_continues_with_result_and_reasoning(self):
        llm = FakeLLM([
            {"role": "assistant", "content": None, "reasoning_content": "  先算。\n",
             "tool_calls": [{"id": "call_1", "type": "function", "function": {
                 "name": "calculator", "arguments": '{"expression":"2+2"}'}}]},
            {"role": "assistant", "content": "结果是 4"},
        ])
        result = AgentRuntime(llm, self.tools, self.sessions).run("user-A", "2+2 等于几？")

        self.assertEqual(result.answer, "结果是 4")
        self.assertEqual(len(result.trace), 1)
        self.assertEqual(result.trace[0].name, "calculator")
        self.assertEqual(result.trace[0].status, "ok")
        self.assertEqual(result.trace[0].result, 4)
        self.assertEqual(llm.requests[1][0][-2]["reasoning_content"], "  先算。\n")
        self.assertEqual(llm.requests[1][0][-1],
                         {"role": "tool", "tool_call_id": "call_1", "content": "4"})
        self.assertIn("calculator", [item["function"]["name"] for item in llm.requests[0][1]])

    # 工具失败时把错误作为工具结果送回模型，让模型决定如何回答。
    def test_tool_error_is_recorded_and_model_can_recover(self):
        llm = FakeLLM([
            {"role": "assistant", "content": None,
             "tool_calls": [{"id": "call_2", "type": "function", "function": {
                 "name": "calculator", "arguments": '{"expression":"1/0"}'}}]},
            {"role": "assistant", "content": "这个算式不能除以零"},
        ])
        result = AgentRuntime(llm, self.tools, self.sessions).run("user-A", "算 1/0")

        self.assertEqual(result.answer, "这个算式不能除以零")
        self.assertEqual(result.trace[0].status, "error")
        self.assertIn("division by zero", llm.requests[1][0][-1]["content"])
        self.assertEqual(self.sessions.open("user-A", result.session_id).context.messages[-1]["content"],
                         "这个算式不能除以零")

    # 达到最大模型调用次数后明确报错，并保留已完成的工具执行记录。
    def test_max_steps_stops_loop_and_saves_session(self):
        llm = FakeLLM([
            {"role": "assistant", "content": None,
             "tool_calls": [{"id": "call_3", "type": "function", "function": {
                 "name": "calculator", "arguments": '{"expression":"2+2"}'}}]},
        ])
        runtime = AgentRuntime(llm, self.tools, self.sessions, max_steps=1)

        with self.assertRaises(AgentRunError) as caught:
            runtime.run("user-A", "一直计算")

        self.assertEqual(len(llm.requests), 1)
        self.assertEqual(caught.exception.trace[0].result, 4)
        saved = self.sessions.open("user-A", caught.exception.session_id)
        self.assertEqual(saved.context.messages[-1]["content"], "4")

    # 同一用户的两个窗口应有独立历史。
    def test_two_windows_do_not_share_history(self):
        llm = FakeLLM([
            {"role": "assistant", "content": "北京晴"},
            {"role": "assistant", "content": "周报已记"},
            {"role": "assistant", "content": "天气仍晴"},
        ])
        runtime = AgentRuntime(llm, self.tools, self.sessions)
        weather = runtime.run("user-A", "查天气")
        report = runtime.run("user-A", "写周报")
        runtime.run("user-A", "明天呢？", weather.session_id)

        self.assertNotEqual(weather.session_id, report.session_id)
        self.assertNotIn("写周报", str(llm.requests[2][0]))
        self.assertEqual(self.sessions.open("user-A", report.session_id).context.messages[-1]["content"],
                         "周报已记")

    # 旧对话超出轮次限制时，先让模型生成摘要，再带摘要回答追问。
    def test_old_rounds_are_summarized_by_llm(self):
        self.sessions = SessionManager(self.sessions.store, max_rounds=1)
        llm = FakeLLM([
            {"role": "assistant", "content": "北京晴", "reasoning_content": "内部思考"},
            {"role": "assistant", "content": "之前查过北京天气"},
            {"role": "assistant", "content": "明天暂无数据"},
        ])
        runtime = AgentRuntime(llm, self.tools, self.sessions)
        first = runtime.run("user-A", "北京天气怎么样？")
        second = runtime.run("user-A", "明天呢？", first.session_id)

        self.assertEqual(second.answer, "明天暂无数据")
        self.assertEqual(len(llm.requests), 3)
        self.assertIsNone(llm.requests[1][1])
        self.assertIn("北京天气怎么样？", llm.requests[1][0][0]["content"])
        self.assertNotIn("内部思考", llm.requests[1][0][0]["content"])
        self.assertIn("之前查过北京天气", llm.requests[2][0][0]["content"])
        self.assertEqual(self.sessions.open("user-A", first.session_id).context.summary,
                         "之前查过北京天气")

    # 一个回复中的多个工具都执行；重启后的工具追问能查到自己的待办。
    def test_weather_and_todo_follow_up_after_restart(self):
        llm = FakeLLM([
            {"content": None, "tool_calls": [
                {"id": "weather-1", "type": "function", "function": {
                    "name": "weather", "arguments": '{"city":"北京"}'}},
                {"id": "todo-1", "type": "function", "function": {
                    "name": "todo", "arguments": '{"action":"add","title":"带伞"}'}},
            ]}, {"content": "已查询天气并记录带伞"},
        ])
        result = AgentRuntime(llm, self.tools, self.sessions).run("A", "查北京天气并记待办带伞")
        self.assertEqual([item.name for item in result.trace], ["weather", "todo"])
        self.assertEqual([item["tool_call_id"] for item in llm.requests[1][0] if item["role"] == "tool"],
                         ["weather-1", "todo-1"])
        reopened = SessionManager(SQLiteStore(self.sessions.store.database_path))
        follow_up = FakeLLM([
            {"content": None, "tool_calls": [{"id": "list-1", "type": "function", "function": {
                "name": "todo", "arguments": '{"action":"list"}'}}]},
            {"content": "你的待办是带伞"},
        ])
        answer = AgentRuntime(follow_up, self.tools, reopened).run("A", "还有什么待办？", result.session_id)
        self.assertEqual(answer.trace[0].result["todos"][0]["title"], "带伞")
        saved = reopened.open("A", result.session_id)
        self.assertEqual(saved.metadata["status"], "completed")
        self.assertTrue(saved.metadata["created_at"])
        self.assertEqual(saved.metadata["current_task"], "还有什么待办？")
        logs = reopened.store.tool_history("A", result.session_id)
        self.assertEqual(len(logs), 3)
        self.assertTrue(all(item["status"] == "ok" for item in logs))
        self.assertEqual(reopened.store.tool_history("B", result.session_id), [])

    # 写工具成功后模型超时，业务数据、错误状态和执行日志仍然保存。
    def test_failure_preserves_task_and_tool_history(self):
        llm = FakeLLM([
            {"content": None, "tool_calls": [{"id": "todo-1", "type": "function", "function": {
                "name": "todo", "arguments": '{"action":"add","title":"交周报"}'}}]},
            TimeoutError("请求超时"),
        ])
        with self.assertRaises(AgentRunError) as caught:
            AgentRuntime(llm, self.tools, self.sessions).run("A", "记待办交周报")
        saved = self.sessions.open("A", caught.exception.session_id)
        self.assertEqual(saved.metadata["status"], "failed")
        self.assertIn("请求超时", saved.metadata["last_error"])
        self.assertEqual(self.sessions.store.tool_history("A", saved.session_id)[0]["status"], "ok")
        bound = self.tools.for_session(self.sessions.store, "A", saved.session_id)
        self.assertEqual(len(bound.execute("todo", {"action": "list"})["todos"]), 1)

    # 工具结果已写日志但还没回填消息时中断，恢复时直接使用已保存的结果。
    def test_recover_pending_call_from_execution_log(self):
        from agent.parser import ToolCall

        session = self.sessions.create("A")
        session.metadata.update(status="running", run_id="run-1")
        session.context.add_user("2+2")
        session.context.add_assistant(tool_calls=[ToolCall("call-1", "calculator", {"expression": "2+2"})])
        self.sessions.save(session)
        execution_id = self.sessions.store.start_tool("A", session.session_id, "run-1", 1,
                                                       "call-1", "calculator", {"expression": "2+2"})
        self.sessions.store.finish_tool(execution_id, "ok", 4)
        llm = FakeLLM([{"content": "之前算出 4"}])
        result = AgentRuntime(llm, self.tools, self.sessions).run("A", "继续", session.session_id)
        self.assertEqual(result.trace, [])
        self.assertIn({"role": "tool", "tool_call_id": "call-1", "content": "4"}, llm.requests[0][0])
        self.assertEqual(len(self.sessions.store.tool_history("A", session.session_id)), 1)

    # 中断时未确认的写操作只标记未知，不重新执行。
    def test_unknown_write_is_not_replayed(self):
        from agent.parser import ToolCall

        session = self.sessions.create("A")
        session.metadata.update(status="running", run_id="run-unknown")
        session.context.add_user("记待办")
        arguments = {"action": "add", "title": "带伞"}
        session.context.add_assistant(tool_calls=[ToolCall("todo-unknown", "todo", arguments)])
        self.sessions.save(session)
        self.sessions.store.start_tool("A", session.session_id, "run-unknown", 1,
                                       "todo-unknown", "todo", arguments)
        llm = FakeLLM([{"content": "请先确认上次执行结果"}])
        result = AgentRuntime(llm, self.tools, self.sessions).run("A", "继续", session.session_id)
        self.assertEqual(result.trace, [])
        history = self.sessions.store.tool_history("A", session.session_id)
        self.assertEqual(history[0]["status"], "unknown")
        self.assertIn("结果未知", llm.requests[0][0][2]["content"])
        bound = self.tools.for_session(self.sessions.store, "A", session.session_id)
        self.assertEqual(bound.execute("todo", {"action": "list"})["todos"], [])


if __name__ == "__main__":
    unittest.main()
