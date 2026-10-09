import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from agent.context import Context
from agent.store import SQLiteStore


class SQLiteStoreTests(unittest.TestCase):
    # 两个窗口使用不同的 session ID，保存和恢复时不能串数据。
    def test_save_and_load_independent_sessions(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "sessions.db"
            store = SQLiteStore(database)
            first = Context()
            first.add_user("查天气")
            second = Context()
            second.add_user("写周报")

            store.save("user-A", "window-1", first.to_dict())
            store.save("user-A", "window-2", second.to_dict())

            reopened = SQLiteStore(database)
            self.assertEqual(Context.from_dict(reopened.load("user-A", "window-1")).messages[0]["content"], "查天气")
            self.assertEqual(Context.from_dict(reopened.load("user-A", "window-2")).messages[0]["content"], "写周报")
            self.assertTrue(database.read_bytes().startswith(b"SQLite format 3"))

    # 不同用户即使使用相同的 session ID，也不能读到对方的记录。
    def test_same_session_id_is_isolated_by_user(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SQLiteStore(Path(directory) / "sessions.db")
            store.save("user-A", "window-1", {"messages": [], "summary": "A 的数据"})
            store.save("user-B", "window-1", {"messages": [], "summary": "B 的数据"})
            self.assertEqual(store.load("user-A", "window-1")["summary"], "A 的数据")
            self.assertEqual(store.load("user-B", "window-1")["summary"], "B 的数据")
            self.assertIsNone(store.load("user-C", "window-1"))

    # 未创建过的会话应返回 None，交给 session.py 决定是否新建。
    def test_missing_session_returns_none(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertIsNone(SQLiteStore(Path(directory) / "sessions.db").load("user-A", "new-window"))

    # session ID 不能包含路径分隔符或上级目录符号。
    def test_invalid_session_id_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SQLiteStore(Path(directory) / "sessions.db")
            for session_id in ("../outside", "a/b", "", "."):
                with self.subTest(session_id=session_id), self.assertRaises(ValueError):
                    store.load("user-A", session_id)
            with self.assertRaises(ValueError):
                store.load(" ", "window-1")

    # 数据库中的会话 JSON 损坏时给出错误，不把坏数据当作空会话。
    def test_corrupt_record_raises_error(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "sessions.db"
            store = SQLiteStore(database)
            with closing(sqlite3.connect(database)) as connection:
                with connection:
                    connection.execute(
                        "INSERT INTO sessions (user_id, session_id, data) VALUES (?, ?, ?)",
                        ("user-A", "broken", "{bad json"),
                    )
            with self.assertRaises(ValueError):
                store.load("user-A", "broken")

    # 保存失败不能覆盖原有会话。
    def test_failed_save_keeps_previous_data(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SQLiteStore(Path(directory) / "sessions.db")
            store.save("user-A", "window-1", {"messages": [], "summary": "原数据"})
            with self.assertRaises(ValueError):
                store.save("user-A", "window-1", {"messages": [], "summary": {1, 2}})
            self.assertEqual(store.load("user-A", "window-1")["summary"], "原数据")

    # 遇到旧版单主键表时，应明确提示结构不兼容，不静默覆盖旧会话。
    def test_legacy_table_is_rejected_clearly(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "sessions.db"
            with closing(sqlite3.connect(database)) as connection:
                with connection:
                    connection.execute(
                        "CREATE TABLE sessions (session_id TEXT PRIMARY KEY, data TEXT NOT NULL)"
                    )
            with self.assertRaisesRegex(ValueError, "旧版"):
                SQLiteStore(database)


if __name__ == "__main__":
    unittest.main()
