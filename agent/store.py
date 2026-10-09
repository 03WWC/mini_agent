"""使用 SQLite 按用户和 session ID 保存对话数据。"""

import json
import re
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any, Dict


class SQLiteStore:
    """只负责读写会话数据；创建和切换会话由 session.py 负责。"""

    # 指定数据库文件，并建立保存会话的表。
    def __init__(self, database_path: str | Path):
        self.database_path = Path(database_path).resolve()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.database_path)) as connection:
            with connection:
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS sessions ("
                    "user_id TEXT NOT NULL, "
                    "session_id TEXT NOT NULL, "
                    "data TEXT NOT NULL, "
                    "PRIMARY KEY (user_id, session_id)"
                    ")"
                )
                columns = {row[1] for row in connection.execute("PRAGMA table_info(sessions)")}
                if "user_id" not in columns:
                    raise ValueError("检测到旧版 sessions 表，缺少 user_id；请先迁移旧数据")

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
            row = connection.execute(
                "SELECT data FROM sessions WHERE user_id = ? AND session_id = ?",
                (user_id, session_id),
            ).fetchone()
        if row is None:
            return None

        try:
            data = json.loads(row[0])
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"会话数据损坏: {session_id}") from exc
        if not isinstance(data, dict):
            raise ValueError(f"会话数据格式错误: {session_id}")
        return data

    # 在事务中新增或更新会话；失败时不会覆盖原有数据。
    def save(self, user_id: str, session_id: str, data: Dict[str, Any]) -> None:
        self._check_user_id(user_id)
        self._check_session_id(session_id)
        if not isinstance(data, dict):
            raise ValueError("会话数据必须是对象")
        try:
            content = json.dumps(data, ensure_ascii=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("会话数据无法转换成 JSON") from exc

        with closing(sqlite3.connect(self.database_path)) as connection:
            with connection:
                connection.execute(
                    "INSERT INTO sessions (user_id, session_id, data) VALUES (?, ?, ?) "
                    "ON CONFLICT(user_id, session_id) DO UPDATE SET data = excluded.data",
                    (user_id, session_id, content),
                )
