"""创建、恢复和保存用户的独立会话。"""

from dataclasses import dataclass
from uuid import uuid4

from .context import Context
from .store import SQLiteStore


@dataclass(frozen=True)
class Session:
    """一个窗口对应一个会话，里面有自己的 Context。"""

    user_id: str
    session_id: str
    context: Context


class SessionManager:
    """连接 Context 和 SQLiteStore，不负责调用 LLM。"""

    # 保存存储对象以及新会话使用的上下文限制。
    def __init__(self, store: SQLiteStore, max_rounds: int = 10,
                 max_chars: int = 12000):
        self.store = store
        self.max_rounds = max_rounds
        self.max_chars = max_chars

    # 为用户创建一个新会话，并立即存入数据库。
    def create(self, user_id: str) -> Session:
        session_id = uuid4().hex
        context = Context(max_rounds=self.max_rounds, max_chars=self.max_chars)
        self.store.save(user_id, session_id, context.to_dict())
        return Session(user_id=user_id, session_id=session_id, context=context)

    # 按用户和会话 ID 恢复旧会话；找不到时不自动新建。
    def open(self, user_id: str, session_id: str) -> Session:
        data = self.store.load(user_id, session_id)
        if data is None:
            raise KeyError(f"未找到会话: {session_id}")
        context = Context.from_dict(data, max_rounds=self.max_rounds,
                                    max_chars=self.max_chars)
        return Session(user_id=user_id, session_id=session_id, context=context)

    # 保存当前会话；压缩由 Runtime 在调用模型前完成。
    def save(self, session: Session) -> None:
        self.store.save(session.user_id, session.session_id,
                        session.context.to_dict())
