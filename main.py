
import argparse
import json
import sqlite3

from agent.llm import DeepSeekClient, DeepSeekError
from agent.runtime import AgentRunError, AgentRuntime
from agent.session import SessionManager
from agent.store import SQLiteStore
from agent.tools import create_default_registry


# 成功和失败的对话都显示已经执行的工具，方便核对副作用。
def print_trace(trace) -> None:
    for item in trace:
        arguments = json.dumps(item.arguments, ensure_ascii=False)
        print(f"工具: {item.name}({arguments}) -> {item.status}: {item.result}")


# 解析启动参数，创建或打开会话，然后反复读取用户输入。
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Mini Agent 终端演示")
    parser.add_argument("--user-id", required=True, help="当前用户的 ID")
    parser.add_argument("--session-id", help="继续已有窗口时填写会话 ID")
    parser.add_argument("--db", default="data/sessions.db", help="SQLite 数据库路径")
    parser.add_argument("--docs-dir", default=".", help="read_docs 可读取的文档目录")
    parser.add_argument("--max-steps", type=int, default=8, help="每次输入的最大模型循环次数")
    parser.add_argument("--max-rounds", type=int, default=10, help="保留的原始对话轮数")
    parser.add_argument("--max-chars", type=int, default=12000, help="触发摘要的估算字符数")
    args = parser.parse_args(argv)

    try:
        client = DeepSeekClient()
        sessions = SessionManager(SQLiteStore(args.db), args.max_rounds, args.max_chars)
        tools = create_default_registry(args.docs_dir)
        runtime = AgentRuntime(client, tools, sessions, args.max_steps)
        session = (sessions.open(args.user_id, args.session_id) if args.session_id
                   else sessions.create(args.user_id))
    except (ValueError, KeyError, sqlite3.Error, OSError) as exc:
        print(f"启动失败: {exc}")
        return 1

    session_id = session.session_id
    print(f"会话 ID: {session_id}")
    print("输入 /exit 退出；下次可用 --session-id 继续这个窗口。")

    while True:
        try:
            question = input("你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        if question == "/exit":
            return 0
        if not question:
            continue

        try:
            result = runtime.run(args.user_id, question, session_id)
        except AgentRunError as exc:
            print_trace(exc.trace)
            print(f"错误: {exc}")
            continue
        except (DeepSeekError, ValueError, KeyError, sqlite3.Error, OSError) as exc:
            print(f"错误: {exc}")
            continue

        # 展示这一轮实际执行的工具；不展示模型内部的思考文本。
        print_trace(result.trace)
        print(f"助手: {result.answer}")


if __name__ == "__main__":
    raise SystemExit(main())
