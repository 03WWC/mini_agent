# Mini Agent

一个从零实现的命令行 Agent 示例。主循环由项目自己的 `AgentRuntime` 实现；DeepSeek 负责决定直接回答还是调用工具，OpenAI Python SDK 只用于发送请求。

## 运行

需要 Python 3.10 或更新版本，以及可用的 DeepSeek API Key。在 PowerShell 中从项目根目录运行：

```powershell
python -m pip install -r requirements.txt
$env:DEEPSEEK_API_KEY = "你的 API Key"
python main.py --user-id user-A
```

启动后会立即打印会话 ID。输入问题开始聊天，输入 `/exit` 退出。以后继续同一个会话：

```powershell
python main.py --user-id user-A --session-id 上次显示的会话ID
```

省略 `--session-id` 会新建会话，因此同一用户可以在两个终端窗口分别聊天。恢复时必须使用同一个 `--user-id` 和同一份数据库。默认数据库是 `data/sessions.db`，也可以通过 `--db` 指定：

```powershell
python main.py --user-id user-A --db data/sessions.db --docs-dir .
```

`--docs-dir` 是 `read_docs` 允许读取的目录，默认是当前工作目录。`DEEPSEEK_MODEL` 和 `DEEPSEEK_BASE_URL` 可选；代码中的默认值分别是 `deepseek-flash` 和 `https://api.deepseek.com`。API Key 只从环境变量读取，不需要写入项目文件。

运行全部测试：

```powershell
python -m unittest discover -s tests -v
```

在项目根目录使用 `-m` 运行测试。直接执行 `python tests/test_agent.py` 时，Python 可能找不到同级的 `agent` 包。

## 系统设计

```mermaid
flowchart LR
    U[终端用户] --> M[main.py]
    M --> S[SessionManager]
    S <--> DB[(SQLite sessions)]
    M --> R[AgentRuntime]
    R <--> C[Context]
    R --> L[DeepSeekClient]
    L --> P[Parser]
    P --> R
    R --> T[ToolRegistry]
    T --> R
```

| 模块 | 职责 |
| --- | --- |
| `agent/runtime.py` | 实现 Agent Loop、最大模型调用次数、工具执行记录和异常处理。 |
| `agent/llm.py` | 用 OpenAI SDK 调用 DeepSeek Chat Completions。 |
| `agent/parser.py` | 把模型回复解析为思考、工具调用或最终答案。 |
| `agent/tools.py` | 注册工具、提供 JSON Schema、检查参数并执行工具。 |
| `agent/context.py` | 保存消息，判断何时压缩旧对话，组织发给模型的上下文。 |
| `agent/session.py` | 创建、恢复和保存独立会话。 |
| `agent/store.py` | 用 SQLite 持久化会话数据。 |
| `main.py` | 命令行交互入口。 |

每次 `run` 的流程是：

1. 接收用户输入，按用户 ID 和会话 ID 打开会话，并把输入加入 Context。
2. 把上下文和工具 Schema 发给 DeepSeek。模型可以直接回答，也可以选择工具。
3. 如果模型要求调用工具，执行工具，并用 `tool_call_id` 把结果对应回该调用；工具错误也作为结果交回模型。
4. 带着工具结果继续循环，直到模型给出最终答案或达到 `max_steps`。默认最多进行 8 次主循环模型调用，压缩摘要的独立请求不计入这个次数。

内置工具有 `calculator`、本地模拟的 `search`、`read_docs` 和模拟的 `weather`。`search` 不访问互联网；`weather` 只提供北京、上海、深圳的演示数据，不是实时天气。当前没有独立的待办事项工具。

## 会话记忆：召回时机与放置方式

这里的 memory 指**当前会话的对话上下文**，不是向量检索或跨会话共享的长期知识库。

- **存在哪里**：SQLite 的 `sessions` 表以 `(user_id, session_id)` 为联合主键，`data` 字段保存 JSON，包含 `messages` 和 `summary`。不同会话的消息不会混在一起。
- **何时召回**：每次用户发送新消息时，`SessionManager.open()` 从 SQLite 恢复该会话的 Context。每次调用回答模型之前，`Context.for_llm()` 组织要发送的消息。无需关键词搜索或手动触发。
- **放到哪里**：如果已有摘要，它作为最前面的一条 `system` 消息，内容以“先前对话摘要：”开头；后面依次放保留的原始用户消息、助手消息和工具结果。模型可用这些内容回答纯对话追问，也可结合旧工具结果处理带工具的追问。
- **何时压缩**：默认原始对话超过 10 个用户轮次，或消息与摘要的估算字符数超过 12000 时，Context 选择较早的完整轮次，由 Runtime **额外调用一次 DeepSeek** 生成摘要。已有摘要也会传给这次请求；最新一轮保留原文。摘要长度最多取 `max_chars` 的一半。
- **思考内容**：模型返回的 `reasoning_content` 会在需要时随原始助手消息保存，以便工具调用后的请求能够正确延续；生成摘要时会排除它，不把内部思考写进长期摘要。

压缩按字符数估算，不是精确的 token 计算。若最新一轮自身过长，当前实现不会把这一轮再压缩。模型摘要可能丢失细节，且压缩会增加一次 API 请求及相应耗时和费用。

## Prompt 与问题记录

- [AI_PROMPT.md](AI_PROMPT.md)：当前程序实际发送的提示内容，以及开发阶段的关键要求。
- [问题解决记录.md](问题解决记录.md)：遇到的问题、原因、处理方式和验证结果。


![alt text](image.png)