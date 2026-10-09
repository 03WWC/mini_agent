# AI Prompt 记录

本文区分**程序运行时真正发送给 DeepSeek 的内容**和**开发阶段给 AI 辅助工具的要求**。开发阶段的部分按对话要点整理，不是逐字聊天记录。

## 运行时：普通回答与工具调用

普通对话目前**没有固定的全局系统提示词**。`AgentRuntime` 把 `Context.for_llm()` 返回的会话消息和 `ToolRegistry.all_schemas()` 返回的工具 JSON Schema 交给 `DeepSeekClient`。模型请求使用 `tool_choice="auto"`，由模型决定是否调用工具。

四个工具的名称、描述和参数 Schema 在 `agent/tools.py` 中定义。工具说明也是模型做选择时收到的信息，例如 `weather` 的描述明确标注它是模拟天气。工具结果使用对应的 `tool_call_id` 作为 `tool` 消息传回模型。

当会话已有摘要时，`Context` 在原始消息前插入以下系统消息：

```text
先前对话摘要：
{summary}
```

这条消息只用于放置已生成的会话摘要，不是通用的行为规则提示词。

## 运行时：压缩旧对话

达到 Context 的压缩条件后，`AgentRuntime._summarize()` 单独向 DeepSeek 发送一条 `user` 消息。其内容模板与代码一致：

```text
请把旧对话压缩成不超过 {limit} 字的中文摘要，只输出摘要。保留用户目标、已确认的事实、工具结果和未完成事项；不要编造。
已有摘要：{old_summary 或 无}
旧对话：{不含 reasoning_content 的旧消息 JSON}
```

这里 `limit` 为 `max_chars // 2`。压缩请求不提供工具 Schema，返回内容必须能被 Parser 识别为最终文本；否则保持原始消息，不删除旧对话。模型返回的摘要仍会经过长度限制。这个 Prompt 只要求保留对继续对话有用的信息，不能保证每个细节都留下。

## 开发阶段给 AI 的关键要求

以下是本项目开发对话中的要求概述，用于说明实现选择：

1. 核心 Agent Runtime 从零实现，不依赖现成 Agent 框架；分步骤开发，不在未得到指令时扩展功能。
2. 工具定义包含名称、描述、参数 Schema 和执行函数；先实现工具，再完成 Parser、Context、SQLite 会话和 Runtime。
3. 代码保持通俗易懂，函数前使用中文 `#` 注释。
4. 使用 DeepSeek，通过 OpenAI Python SDK 的 `chat.completions.create()` 请求模型。
5. 同一用户的不同窗口用独立 `session_id`，并能在 SQLite 中恢复。
6. Context 过长时由大模型生成摘要，不用手写的截取式摘要代替。
7. 未明确要求时不提交仓库。

开发过程中曾讨论过一版通用系统提示词草稿，但**没有加入程序**。因此不能把该草稿当作当前运行时 Prompt。
