"""通过 OpenAI Python SDK 调用 DeepSeek。"""

import os
from typing import Any, Dict, List

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI


class DeepSeekError(RuntimeError):
    """DeepSeek 请求失败。"""


class DeepSeekClient:
    """负责调用模型；返回字典，供 parser.py 判断回答或工具调用。"""

    # 优先使用传入的配置，否则从系统环境变量读取。
    def __init__(self, api_key: str | None = None, model: str | None = None,
                 base_url: str | None = None, timeout: int = 60):
        self.api_key = api_key if api_key is not None else os.getenv("DEEPSEEK_API_KEY")
        self.model = model if model is not None else os.getenv("DEEPSEEK_MODEL", "deepseek-flash")
        url = base_url if base_url is not None else os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        self.base_url = url.rstrip("/")
        self.timeout = timeout

        if not self.api_key:
            raise ValueError("请设置 DEEPSEEK_API_KEY 环境变量")
        if not self.model or not self.base_url:
            raise ValueError("DeepSeek 模型和接口地址不能为空")
        if timeout <= 0:
            raise ValueError("timeout 必须大于 0")

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url,
                             timeout=self.timeout)

    # SDK 负责 HTTP 请求；有工具时交给模型自主选择是否调用。
    def complete(self, messages: List[Dict[str, Any]],
                 tools: List[Dict[str, Any]] | None = None) -> Dict[str, Any]:
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages 不能为空")

        try:
            if tools:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                    temperature=0,
                )
            else:
                # 普通回答保留文本格式，供当前 Parser 直接读取。
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0,
                )
        except APIStatusError as exc:
            raise DeepSeekError(f"DeepSeek 请求失败，HTTP {exc.status_code}") from exc
        except (APIConnectionError, APITimeoutError) as exc:
            raise DeepSeekError("无法连接 DeepSeek") from exc

        return response.model_dump()
