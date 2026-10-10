import tempfile
import unittest
from pathlib import Path

from agent.session import SessionManager
from agent.store import SQLiteStore
from agent.tools import create_default_registry


class TodoTests(unittest.TestCase):
    # 使用真实数据库和不同窗口，验证业务数据隔离。
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.store = SQLiteStore(self.root / "sessions.db")
        self.sessions = SessionManager(self.store)
        self.first = self.sessions.create("A")
        self.second = self.sessions.create("A")
        self.other = self.sessions.create("B")
        self.base = create_default_registry(self.root)
        self.tools = self.base.for_session(self.store, "A", self.first.session_id)

    # 新增、修改、完成、过滤和删除均写入数据库，重建对象后可以恢复。
    def test_crud_survives_restart(self):
        added = self.tools.execute("todo", {"action": "add", "title": "  交周报  ",
                                          "due_at": "2026-10-12T18:00:00+08:00"})["todo"]
        self.assertEqual(added["title"], "交周报")
        restored = self.base.for_session(SQLiteStore(self.store.database_path), "A", self.first.session_id)
        self.assertEqual(restored.execute("todo", {"action": "list"})["todos"][0]["id"], added["id"])
        changed = restored.execute("todo", {"action": "update", "todo_id": added["id"],
                                            "title": "交正式周报", "due_at": None})["todo"]
        self.assertEqual(changed["title"], "交正式周报")
        self.assertIsNone(changed["due_at"])
        restored.execute("todo", {"action": "complete", "todo_id": added["id"]})
        self.assertEqual(restored.execute("todo", {"action": "list", "status": "pending"})["todos"], [])
        self.assertEqual(len(restored.execute("todo", {"action": "list", "status": "completed"})["todos"]), 1)
        restored.execute("todo", {"action": "delete", "todo_id": added["id"]})
        self.assertEqual(restored.execute("todo", {"action": "list"})["todos"], [])

    # 知道待办 ID 也不能跨窗口或跨用户查看、完成或删除。
    def test_session_and_user_isolation(self):
        todo_id = self.tools.execute("todo", {"action": "add", "title": "私人待办"})["todo"]["id"]
        for user, session in (("A", self.second), ("B", self.other)):
            tools = self.base.for_session(self.store, user, session.session_id)
            self.assertEqual(tools.execute("todo", {"action": "list"})["todos"], [])
            for action in ("complete", "delete"):
                with self.subTest(user=user, action=action), self.assertRaises(ValueError):
                    tools.execute("todo", {"action": action, "todo_id": todo_id})
        self.assertEqual(len(self.tools.execute("todo", {"action": "list"})["todos"]), 1)

    # 格式、必填项和业务参数不正确时，必须在写入前拒绝。
    def test_invalid_arguments_do_not_write(self):
        cases = [
            {"action": "wrong"}, {"action": "add"}, {"action": "add", "title": "   "},
            {"action": "add", "title": 123}, {"action": "add", "title": "x" * 201},
            {"action": "add", "title": "会议", "due_at": "明天下午"},
            {"action": "add", "title": "会议", "due_at": "2026-02-30T12:00:00+08:00"},
            {"action": "add", "title": "会议", "due_at": "2026-10-12T12:00:00"},
            {"action": "complete"}, {"action": "delete", "todo_id": "bad"},
            {"action": "list", "status": "wrong"}, {"action": "list", "title": "误传"},
            {"action": "add", "title": "会议", "user_id": "B"},
        ]
        for arguments in cases:
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                self.tools.execute("todo", arguments)
        self.assertEqual(self.tools.execute("todo", {"action": "list"})["todos"], [])

    # 压缩聊天不会影响单独持久化的待办。
    def test_todo_survives_context_compression(self):
        self.tools.execute("todo", {"action": "add", "title": "明天带伞"})
        context = self.first.context
        context.max_rounds = 1
        context.add_user("记待办")
        context.add_assistant("已记录")
        context.add_user("新问题")
        context.for_llm(lambda old, messages, limit: "旧聊天已压缩")
        self.sessions.save(self.first)
        self.assertEqual(self.tools.execute("todo", {"action": "list"})["todos"][0]["title"], "明天带伞")

    # 不存在的待办不能修改，已完成的待办不能再编辑，重复完成不新增数据。
    def test_state_and_existence_validation(self):
        with self.assertRaises(ValueError):
            self.tools.execute("todo", {"action": "complete", "todo_id": "0" * 32})
        todo_id = self.tools.execute("todo", {"action": "add", "title": "交周报"})["todo"]["id"]
        with self.assertRaises(ValueError):
            self.tools.execute("todo", {"action": "update", "todo_id": todo_id})
        for _ in range(2):
            self.tools.execute("todo", {"action": "complete", "todo_id": todo_id})
        with self.assertRaises(ValueError):
            self.tools.execute("todo", {"action": "update", "todo_id": todo_id, "title": "另一个"})
        self.assertEqual(len(self.tools.execute("todo", {"action": "list"})["todos"]), 1)


if __name__ == "__main__":
    unittest.main()
