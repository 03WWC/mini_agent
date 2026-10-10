import io
import os
import re
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import main
from agent.store import SQLiteStore


class MainTests(unittest.TestCase):
    # 达到步数上限时，终端仍显示已经执行的工具记录。
    def test_limit_error_displays_trace(self):
        with tempfile.TemporaryDirectory() as directory:
            response = {"choices": [{"message": {"tool_calls": [{
                "id": "call-1", "type": "function", "function": {
                    "name": "calculator", "arguments": '{"expression":"2+2"}'},
            }]}}]}
            output = io.StringIO()
            with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key"}), \
                 patch("agent.llm.DeepSeekClient.complete", return_value=response), \
                 patch("builtins.input", side_effect=["算一下", "/exit"]), redirect_stdout(output):
                code = main.main(["--user-id", "A", "--db", str(Path(directory) / "test.db"),
                                  "--max-steps", "1"])
            self.assertEqual(code, 0)
            self.assertIn("calculator", output.getvalue())
            self.assertIn("达到最大", output.getvalue())

    # 首次聊天显示会话 ID 和工具记录，重启后可以用同一 ID 继续追问。
    def test_chat_can_resume_saved_session(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "sessions.db"
            responses = [
                {"choices": [{"message": {"role": "assistant", "content": None,
                    "tool_calls": [{"id": "call_1", "type": "function", "function": {
                        "name": "calculator", "arguments": '{"expression":"2+2"}'}}]}}]},
                {"choices": [{"message": {"role": "assistant", "content": "结果是 4"}}]},
                {"choices": [{"message": {"role": "assistant", "content": "刚才算出了 4"}}]},
            ]
            output = io.StringIO()
            with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key"}), \
                 patch("agent.llm.DeepSeekClient.complete", side_effect=responses) as complete, \
                 patch("builtins.input", side_effect=["2+2 是多少？", "/exit"]), \
                 redirect_stdout(output):
                code = main.main(["--user-id", "user-A", "--db", str(database)])

            self.assertEqual(code, 0)
            session_id = re.search(r"[0-9a-f]{32}", output.getvalue()).group()
            self.assertIn("calculator", output.getvalue())
            self.assertIn("结果是 4", output.getvalue())
            self.assertEqual(SQLiteStore(database).load("user-A", session_id)["messages"][-1]["content"],
                             "结果是 4")

            output = io.StringIO()
            with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key"}), \
                 patch("agent.llm.DeepSeekClient.complete", side_effect=responses[2:]) as follow_up, \
                 patch("builtins.input", side_effect=["刚才结果是多少？", "/exit"]), \
                 redirect_stdout(output):
                code = main.main(["--user-id", "user-A", "--session-id", session_id,
                                  "--db", str(database)])

            self.assertEqual(code, 0)
            self.assertIn("刚才算出了 4", output.getvalue())
            sent_messages = follow_up.call_args.args[0]
            self.assertEqual([message["content"] for message in sent_messages
                              if message["role"] == "user"],
                             ["2+2 是多少？", "刚才结果是多少？"])

    # 用户指定不存在的窗口时给出错误，不新建另一份会话。
    def test_unknown_session_reports_error(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "sessions.db"
            output = io.StringIO()
            with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key"}), \
                 redirect_stdout(output):
                code = main.main(["--user-id", "user-A", "--session-id", "missing",
                                  "--db", str(database)])

            self.assertEqual(code, 1)
            self.assertIn("未找到会话", output.getvalue())
            self.assertIsNone(SQLiteStore(database).load("user-A", "missing"))


if __name__ == "__main__":
    unittest.main()
