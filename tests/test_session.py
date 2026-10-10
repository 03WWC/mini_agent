import tempfile
import unittest
from pathlib import Path

from agent.session import SessionManager
from agent.store import SQLiteStore


class SessionManagerTests(unittest.TestCase):
    # 保存时未提供元数据则使用默认状态，恢复后消息和摘要保持不变。
    def test_session_gets_default_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SQLiteStore(Path(directory) / "sessions.db")
            store.save("A", "old", {"messages": [{"role": "user", "content": "旧问题"}], "summary": "旧摘要"})
            manager = SessionManager(store)
            session = manager.open("A", "old")
            self.assertEqual(session.metadata["status"], "idle")
            manager.save(session)
            loaded = store.load("A", "old")
            self.assertTrue(loaded["metadata"]["created_at"])
            self.assertEqual(loaded["summary"], "旧摘要")
            self.assertEqual(loaded["messages"][0]["content"], "旧问题")

    # 同一用户的两个窗口有不同 ID，保存后可以分别恢复和继续聊天。
    def test_two_windows_are_independent_and_resumable(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "sessions.db"
            manager = SessionManager(SQLiteStore(database))
            weather = manager.create("user-A")
            report = manager.create("user-A")
            self.assertNotEqual(weather.session_id, report.session_id)

            weather.context.add_user("查天气")
            weather.context.add_assistant("北京是晴天")
            manager.save(weather)
            report.context.add_user("写周报")
            report.context.add_assistant("本周完成了测试")
            manager.save(report)

            reopened = SessionManager(SQLiteStore(database))
            weather_again = reopened.open("user-A", weather.session_id)
            report_again = reopened.open("user-A", report.session_id)
            self.assertEqual(weather_again.context.messages[0]["content"], "查天气")
            self.assertEqual(report_again.context.messages[0]["content"], "写周报")

            weather_again.context.add_user("明天呢？")
            reopened.save(weather_again)
            self.assertEqual(len(reopened.open("user-A", report.session_id).context.messages), 2)
            self.assertEqual(reopened.open("user-A", weather.session_id).context.messages[-1]["content"],
                             "明天呢？")

    # 即使知道别人的 session ID，也不能按自己的 user_id 打开它。
    def test_other_user_cannot_open_session(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = SessionManager(SQLiteStore(Path(directory) / "sessions.db"))
            session = manager.create("user-A")
            with self.assertRaises(KeyError):
                manager.open("user-B", session.session_id)

    # 不存在的 session ID 应明确报错，不能悄悄创建空对话。
    def test_unknown_session_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = SessionManager(SQLiteStore(Path(directory) / "sessions.db"))
            with self.assertRaises(KeyError):
                manager.open("user-A", "missing")

    # 创建会话时仍要遵守存储层的用户 ID 校验。
    def test_blank_user_id_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = SessionManager(SQLiteStore(Path(directory) / "sessions.db"))
            with self.assertRaises(ValueError):
                manager.create(" ")


if __name__ == "__main__":
    unittest.main()
