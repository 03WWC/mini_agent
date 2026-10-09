import os
import unittest
from unittest.mock import patch

import httpx
from openai import APIStatusError

from agent.llm import DeepSeekClient, DeepSeekError
from agent.parser import parse_response


class DeepSeekClientTests(unittest.TestCase):
    # SDK 收到消息和工具 Schema，返回值仍能交给现有 Parser。
    def test_complete_uses_sdk_function_calling(self):
        response = {"choices": [{"message": {"role": "assistant", "content": "你好"}}]}
        tools = [{"type": "function", "function": {
            "name": "calculator", "description": "计算", "parameters": {"type": "object"},
        }}]
        with patch("agent.llm.OpenAI") as sdk:
            sdk.return_value.chat.completions.create.return_value.model_dump.return_value = response
            client = DeepSeekClient(api_key="test-key", model="deepseek-flash",
                                    base_url="https://api.deepseek.com/", timeout=30)
            result = client.complete([{"role": "user", "content": "你好"}], tools)

        sdk.assert_called_once_with(api_key="test-key", base_url="https://api.deepseek.com", timeout=30)
        sdk.return_value.chat.completions.create.assert_called_once_with(
            model="deepseek-flash", messages=[{"role": "user", "content": "你好"}],
            tools=tools, tool_choice="auto", temperature=0,
        )
        self.assertEqual(parse_response(result).final_answer, "你好")

    # 没有工具时不传工具字段，密钥和地址仍从 DeepSeek 环境变量读取。
    def test_reads_environment_and_omits_tools_when_empty(self):
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "env-key",
                                     "DEEPSEEK_BASE_URL": "https://example.com/v1",
                                     "DEEPSEEK_MODEL": "example-model"}):
            with patch("agent.llm.OpenAI") as sdk:
                sdk.return_value.chat.completions.create.return_value.model_dump.return_value = {
                    "choices": [{"message": {"content": "完成"}}]
                }
                DeepSeekClient().complete([{"role": "user", "content": "测试"}])

        sdk.assert_called_once_with(api_key="env-key", base_url="https://example.com/v1", timeout=60)
        sdk.return_value.chat.completions.create.assert_called_once_with(
            model="example-model", messages=[{"role": "user", "content": "测试"}],
            temperature=0,
        )

    # 没有密钥时立即报配置错误，不创建 SDK 客户端。
    def test_missing_api_key_is_rejected(self):
        with patch.dict(os.environ, {}, clear=True), patch("agent.llm.OpenAI") as sdk:
            with self.assertRaises(ValueError):
                DeepSeekClient()
            sdk.assert_not_called()

    # SDK 的 HTTP 状态错误转换成项目自己的异常。
    def test_http_error_is_reported(self):
        request = httpx.Request("POST", "https://api.deepseek.com/chat/completions")
        error = APIStatusError("Unauthorized", response=httpx.Response(401, request=request), body=None)
        with patch("agent.llm.OpenAI") as sdk:
            sdk.return_value.chat.completions.create.side_effect = error
            with self.assertRaisesRegex(DeepSeekError, "401"):
                DeepSeekClient(api_key="bad-key").complete([{"role": "user", "content": "测试"}])


if __name__ == "__main__":
    unittest.main()
