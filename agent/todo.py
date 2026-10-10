"""按用户和窗口保存待办，独立于可压缩的聊天记录。"""

import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from uuid import uuid4

from .store import SQLiteStore
from .tools import Tool


class TodoService:
    # 验证当前会话，并建立独立的待办表；身份只来自 Runtime。
    def __init__(self, store: SQLiteStore, user_id: str, session_id: str):
        if store.load(user_id, session_id) is None:
            raise ValueError("待办所属会话不存在")
        self.store = store
        self.user_id = user_id
        self.session_id = session_id
        with closing(sqlite3.connect(store.database_path)) as connection, connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS todos ("
                "id TEXT PRIMARY KEY, user_id TEXT NOT NULL, session_id TEXT NOT NULL, "
                "title TEXT NOT NULL, due_at TEXT, status TEXT NOT NULL, "
                "created_at TEXT NOT NULL, updated_at TEXT NOT NULL)"
            )
            connection.execute("CREATE INDEX IF NOT EXISTS todos_owner ON todos(user_id, session_id)")

    # 按操作校验业务参数，再在事务中读写当前窗口的待办。
    def execute(self, action: str, **fields):
        allowed = {
            "add": {"title", "due_at"}, "list": {"status"},
            "update": {"todo_id", "title", "due_at"},
            "complete": {"todo_id"}, "delete": {"todo_id"},
        }
        if action not in allowed or set(fields) - allowed[action]:
            raise ValueError("操作类型或该操作的参数不正确")
        if action == "add" and "title" not in fields:
            raise ValueError("新增待办必须提供 title")
        if "title" in fields:
            fields["title"] = fields["title"].strip()
            if not fields["title"]:
                raise ValueError("待办标题不能为空白")
        if action in {"update", "complete", "delete"} and not fields.get("todo_id"):
            raise ValueError("此操作必须提供 todo_id")
        if action == "update" and not ({"title", "due_at"} & fields.keys()):
            raise ValueError("修改待办必须提供 title 或 due_at")

        now = datetime.now(timezone.utc).isoformat()
        owner = (self.user_id, self.session_id)
        with closing(sqlite3.connect(self.store.database_path)) as connection, connection:
            connection.row_factory = sqlite3.Row
            if action == "list":
                sql = "SELECT * FROM todos WHERE user_id = ? AND session_id = ?"
                values = owner
                if "status" in fields:
                    sql += " AND status = ?"
                    values += (fields["status"],)
                rows = connection.execute(sql + " ORDER BY created_at, id", values).fetchall()
                return {"todos": [dict(row) for row in rows]}

            if action == "add":
                todo_id = uuid4().hex
                connection.execute(
                    "INSERT INTO todos VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (todo_id, *owner, fields["title"], fields.get("due_at"), "pending", now, now),
                )
            else:
                todo_id = fields["todo_id"]
                row = connection.execute(
                    "SELECT * FROM todos WHERE id = ? AND user_id = ? AND session_id = ?",
                    (todo_id, *owner),
                ).fetchone()
                if row is None:
                    raise ValueError("当前会话中不存在这条待办")
                if action == "delete":
                    connection.execute("DELETE FROM todos WHERE id = ? AND user_id = ? AND session_id = ?",
                                       (todo_id, *owner))
                    return {"deleted": True, "todo_id": todo_id}
                if action == "update" and row["status"] == "completed":
                    raise ValueError("已完成的待办不能再修改")
                connection.execute(
                    "UPDATE todos SET title = ?, due_at = ?, status = ?, updated_at = ? "
                    "WHERE id = ? AND user_id = ? AND session_id = ?",
                    (fields.get("title", row["title"]), fields.get("due_at", row["due_at"]),
                     "completed" if action == "complete" else row["status"], now, todo_id, *owner),
                )
            row = connection.execute("SELECT * FROM todos WHERE id = ?", (todo_id,)).fetchone()
            return {"todo": dict(row)}


# 向模型公开业务参数，不公开用户 ID 和会话 ID。
def create_todo_tool(store: SQLiteStore, user_id: str, session_id: str) -> Tool:
    service = TodoService(store, user_id, session_id)
    return Tool(
        name="todo",
        description=("管理当前窗口的持久化待办。add 新增（必填 title）；list 查询（可按 status 过滤）；"
                     "update 修改、complete 完成、delete 删除（必填 todo_id，可先 list 获取）。"
                     "due_at 必须使用带时区的日期时间，例如 2026-10-12T18:00:00+08:00；"
                     "时间不明确时先询问用户。update 传 due_at=null 可清除截止时间。"),
        parameters={
            "type": "object", "additionalProperties": False, "required": ["action"],
            "properties": {
                "action": {"type": "string", "enum": ["add", "list", "update", "complete", "delete"]},
                "title": {"type": "string", "minLength": 1, "maxLength": 200, "pattern": r"\S"},
                "todo_id": {"type": "string", "pattern": "^[a-f0-9]{32}$"},
                "due_at": {"type": ["string", "null"], "format": "date-time"},
                "status": {"type": "string", "enum": ["pending", "completed"]},
            },
        },
        func=service.execute,
    )
