"""使用 SQLite 按用户和 session ID 保存对话数据。"""

import json
import re
import sqlite3
from contextlib import closing
from pathlib import Path
from datetime import datetime, timezone
from uuid import uuid4
from typing import Any, Dict


# 会话状态字段统一在这里定义，建表和读写都复用它。
SESSION_COLUMNS = {
    "created_at": "TEXT",
    "updated_at": "TEXT",
    "status": "TEXT NOT NULL DEFAULT 'idle'",
    "current_task": "TEXT",
    "run_id": "TEXT",
    "step": "INTEGER NOT NULL DEFAULT 0",
    "last_error": "TEXT",
}


class SQLiteStore:
    """读写会话和工具日志；创建和切换会话由 session.py 负责。"""

    # 指定数据库文件，并建立保存会话的表。
    def __init__(self, database_path: str | Path):
        self.database_path = Path(database_path).resolve()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.database_path)) as connection:
            with connection:
                # 将字段名和类型拼成 SQL；这些定义来自上面的固定字典。
                state_columns = ", ".join(
                    f"{name} {definition}" for name, definition in SESSION_COLUMNS.items()
                )
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS sessions ("
                    "user_id TEXT NOT NULL, "
                    "session_id TEXT NOT NULL, "
                    "data TEXT NOT NULL, "
                    f"{state_columns}, "
                    "PRIMARY KEY (user_id, session_id)"
                    ")"
                )
                columns = {row[1] for row in connection.execute("PRAGMA table_info(sessions)")}
                required = {"user_id", "session_id", "data", *SESSION_COLUMNS}
                if not required.issubset(columns):
                    raise ValueError("不支持旧版 sessions 表，请使用新的数据库文件（--db）")
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS tool_executions ("
                    "id TEXT PRIMARY KEY, user_id TEXT NOT NULL, session_id TEXT NOT NULL, "
                    "run_id TEXT NOT NULL, step INTEGER NOT NULL, call_id TEXT NOT NULL, "
                    "name TEXT NOT NULL, arguments TEXT NOT NULL, status TEXT NOT NULL, "
                    "result TEXT, started_at TEXT NOT NULL, finished_at TEXT)"
                )
                connection.execute("CREATE INDEX IF NOT EXISTS executions_owner "
                                   "ON tool_executions(user_id, session_id)")

    # 检查用户 ID 是否是可用的非空标识符。
    def _check_user_id(self, user_id: str) -> None:
        if not isinstance(user_id, str) or not user_id.strip() or len(user_id) > 128:
            raise ValueError("user_id 必须是长度为 1～128 的非空字符串")

    # 检查会话 ID，避免空值和不适合用作标识符的字符。
    def _check_session_id(self, session_id: str) -> None:
        if not isinstance(session_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", session_id):
            raise ValueError("session_id 只能包含字母、数字、下划线和短横线，长度为 1～64")

    # 读取指定会话；不存在时返回 None，损坏时明确报错。
    def load(self, user_id: str, session_id: str) -> Dict[str, Any] | None:
        self._check_user_id(user_id)
        self._check_session_id(session_id)
        with closing(sqlite3.connect(self.database_path)) as connection:
            connection.row_factory = sqlite3.Row
            row = connection.execute(
                "SELECT * FROM sessions WHERE user_id = ? AND session_id = ?",
                (user_id, session_id),
            ).fetchone()
        if row is None:
            return None

        try:
            data = json.loads(row["data"])
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"会话数据损坏: {session_id}") from exc
        if not isinstance(data, dict):
            raise ValueError(f"会话数据格式错误: {session_id}")
        # 对上层仍提供 metadata 字典，实际数据库以独立列为准。
        data["metadata"] = {name: row[name] for name in SESSION_COLUMNS}
        return data

    # 在事务中新增或更新会话；失败时不会覆盖原有数据。
    def save(self, user_id: str, session_id: str, data: Dict[str, Any]) -> None:
        self._check_user_id(user_id)
        self._check_session_id(session_id)
        if not isinstance(data, dict):
            raise ValueError("会话数据必须是对象")
        metadata = data.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError("metadata 必须是对象")
        now = datetime.now(timezone.utc).isoformat()
        defaults = {"created_at": now, "updated_at": now, "status": "idle", "step": 0}
        values = [metadata.get(name, defaults.get(name)) for name in SESSION_COLUMNS]
        try:
            content = json.dumps({"messages": data.get("messages", []),
                                  "summary": data.get("summary", "")}, ensure_ascii=False,
                                 allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("会话数据无法转换成 JSON") from exc

        with closing(sqlite3.connect(self.database_path)) as connection:
            with connection:
                names = ", ".join(SESSION_COLUMNS)
                placeholders = ", ".join("?" for _ in SESSION_COLUMNS)
                updates = ", ".join(f"{name} = excluded.{name}" for name in SESSION_COLUMNS
                                    if name != "created_at")
                connection.execute(
                    f"INSERT INTO sessions (user_id, session_id, data, {names}) "
                    f"VALUES (?, ?, ?, {placeholders}) "
                    f"ON CONFLICT(user_id, session_id) DO UPDATE SET data = excluded.data, {updates}",
                    (user_id, session_id, content, *values),
                )

    # 工具执行前记录意图，崩溃后可以辨认尚未确认结果的调用。
    def start_tool(self, user_id: str, session_id: str, run_id: str, step: int,
                   call_id: str, name: str, arguments: Dict[str, Any]) -> str:
        execution_id = uuid4().hex
        with closing(sqlite3.connect(self.database_path)) as connection, connection:
            connection.execute(
                "INSERT INTO tool_executions "
                "(id, user_id, session_id, run_id, step, call_id, name, arguments, status, started_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'running', ?)",
                (execution_id, user_id, session_id, run_id, step, call_id, name,
                 json.dumps(arguments, ensure_ascii=False, allow_nan=False),
                 datetime.now(timezone.utc).isoformat()),
            )
        return execution_id

    # 保存工具结果；独立于聊天压缩，保留整个会话的执行历史。
    def finish_tool(self, execution_id: str, status: str, result: Any) -> None:
        if status not in {"ok", "error", "unknown"}:
            raise ValueError("无效的执行状态")
        with closing(sqlite3.connect(self.database_path)) as connection, connection:
            connection.execute(
                "UPDATE tool_executions SET status = ?, result = ?, finished_at = ? WHERE id = ?",
                (status, json.dumps(result, ensure_ascii=False, allow_nan=False),
                 datetime.now(timezone.utc).isoformat(), execution_id),
            )

    # 只返回指定用户、指定窗口的日志，参数和结果还原成 Python 对象。
    def tool_history(self, user_id: str, session_id: str) -> list[dict]:
        self._check_user_id(user_id)
        self._check_session_id(session_id)
        with closing(sqlite3.connect(self.database_path)) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                "SELECT * FROM tool_executions WHERE user_id = ? AND session_id = ? ORDER BY rowid",
                (user_id, session_id),
            ).fetchall()
        records = []
        for row in rows:
            record = dict(row)
            record["arguments"] = json.loads(record["arguments"])
            record["result"] = json.loads(record["result"]) if record["result"] is not None else None
            records.append(record)
        return records
