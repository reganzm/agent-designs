"""
routing.py - 使用 Google ADK + LiteLLM 调用 DeepSeek 的智能体路由示例

功能说明：
  - 基于 Google Agent Development Kit (ADK) 实现多智能体协作与路由。
  - 通过 LiteLLM 适配器接入 DeepSeek 模型（替代默认 Gemini），实现预订/问答等委派。
  - 协调者 (Coordinator) 根据用户请求将任务委派给 Booker（预订）或 Info（通用问答）子智能体。

依赖与配置：
  - 推荐在项目下用虚拟环境（避免 Windows 长路径问题）：
      python -m venv venv
      .\\venv\\Scripts\\Activate.ps1   # PowerShell 激活
      pip install "google-adk[extensions]" python-dotenv google-genai
  - 或直接安装：pip install google-genai "google-adk[extensions]" python-dotenv
  - 环境变量：DEEPSEEK_API_KEY（必填，见 https://platform.deepseek.com/api_keys）
  - Windows 若遇 Unicode 错误，可设置：set PYTHONUTF8=1
"""

import asyncio
import os
import uuid
from typing import Dict, Any, Optional

from dotenv import load_dotenv

load_dotenv()

if not os.getenv("DEEPSEEK_API_KEY"):
    raise RuntimeError(
        "未设置 DEEPSEEK_API_KEY。请在 .env 中配置或设置环境变量，"
        "申请地址：https://platform.deepseek.com/api_keys"
    )

from google.adk.agents import Agent
from google.adk.runners import InMemoryRunner
from google.adk.tools import FunctionTool
from google.genai import types
from google.adk.events import Event
from google.adk.models.lite_llm import LiteLlm

# 使用 LiteLLM 适配器接入 DeepSeek，供所有智能体共用（支持 function calling）
# 可选模型：deepseek/deepseek-chat、deepseek/deepseek-reasoner 等
deepseek_model = LiteLlm(model="deepseek/deepseek-chat")

# --- 定义工具函数 ---
# 以下函数模拟专业智能体所执行的具体操作。

def booking_handler(request: str) -> str:
    """
    处理机票、酒店等预订类请求。
    参数:
        request: 用户的预订需求描述。
    返回:
        表示已处理该预订的确认信息。
    """
    print("-------------------------- 预订处理器被调用 ----------------------------")
    return f"已模拟完成针对「{request}」的预订操作。"

def info_handler(request: str) -> str:
    """
    处理一般性信息咨询请求。
    参数:
        request: 用户的问题内容。
    返回:
        表示已处理该信息请求的回复内容。
    """
    print("-------------------------- 信息处理器被调用 ----------------------------")
    return f"针对「{request}」的信息查询已处理，结果为模拟检索内容。"

def unclear_handler(request: str) -> str:
    """处理无法被委派到子智能体的请求。"""
    return f"协调者无法委派该请求：「{request}」。请说明得更具体一些。"

# --- 从函数创建工具 ---
booking_tool = FunctionTool(booking_handler)
info_tool = FunctionTool(info_handler)

# --- 定义配备各自工具的专业子智能体（均使用 DeepSeek 模型）---
booking_agent = Agent(
    name="Booker",
    model=deepseek_model,
    description="专门负责机票、酒店等所有预订请求的智能体，通过调用预订工具完成任务。",
    tools=[booking_tool]
)

info_agent = Agent(
    name="Info",
    model=deepseek_model,
    description="负责通用信息咨询、回答用户各类问题的智能体，通过调用信息工具完成任务。",
    tools=[info_tool]
)

# --- 定义带明确委派指令的父智能体（协调者同样使用 DeepSeek）---
coordinator = Agent(
    name="Coordinator",
    model=deepseek_model,
    instruction=(
        "你是主协调者。你的唯一职责是分析用户请求，并委派给合适的专业智能体，不要直接回答用户。\n"
        "- 与机票、酒店等预订相关的请求，委派给「Booker」智能体。\n"
        "- 其他一般性信息、问答类请求，委派给「Info」智能体。"
    ),
    description="将用户请求路由到对应专业智能体的协调者。",
    # 定义了 sub_agents 即默认启用由大模型驱动的自动委派（Auto-Flow）。
    sub_agents=[booking_agent, info_agent]
)

# --- 从事件中提取文本（用于流式与非流式）---
def _event_text(event: Event) -> str:
    """从 ADK 事件中提取文本内容。"""
    if not event or not event.content:
        return ""
    if hasattr(event.content, "text") and event.content.text:
        return event.content.text
    if event.content.parts:
        return "".join(p.text or "" for p in event.content.parts if getattr(p, "text", None))
    return ""


# --- 执行逻辑：创建会话、发送用户消息、消费事件流并取最终回复 ---
def run_coordinator(runner: InMemoryRunner, request: str, stream: bool = False):
    """用给定的请求运行协调者智能体并进行委派（子智能体由 DeepSeek 驱动）。

    Args:
        runner: InMemoryRunner 实例。
        request: 用户输入文本。
        stream: 若为 True，则边收边打印（流式输出）；否则等完整回复后再打印。
    """
    print(f"\n--- 协调者处理请求: 「{request}」 ---")
    if stream:
        print("回复: ", end="", flush=True)
    final_result = ""
    try:
        user_id = "user_123"
        session_id = str(uuid.uuid4())
        asyncio.run(
            runner.session_service.create_session(
                app_name=runner.app_name, user_id=user_id, session_id=session_id
            )
        )

        for event in runner.run(
            user_id=user_id,
            session_id=session_id,
            new_message=types.Content(
                role='user',
                parts=[types.Part(text=request)]
            ),
        ):
            text = _event_text(event)
            if text:
                if stream:
                    # 流式：有内容就立即打印（后端可能发增量或整段，都直接输出）
                    print(text, end="", flush=True)
                if event.is_final_response():
                    final_result = text  # 最终事件通常带完整内容
                else:
                    final_result += text
            if event.is_final_response():
                if stream:
                    print()  # 换行
                break

        if not stream:
            print(f"协调者最终回复: {final_result}")
        return final_result
    except Exception as e:
        print(f"处理请求时出错: {e}")
        return f"处理请求时出错: {e}"

def main():
    """运行 ADK 示例的主函数（后端为 DeepSeek，经 LiteLLM 接入）。"""
    print("--- Google ADK 路由示例（DeepSeek + LiteLLM，自动委派）---")
    print("说明: 需配置 DEEPSEEK_API_KEY，并安装 google-adk、litellm。")

    runner = InMemoryRunner(coordinator)
    # 使用示例：stream=True 为流式输出（边生成边打印），False 为等完整回复后再打印
    result_a = run_coordinator(runner, "帮我订一家巴黎的酒店。", stream=True)
    print(f"[返回值长度: {len(result_a)}]")
    result_b = run_coordinator(runner, "世界上最高的山是哪座？", stream=False)
    print(f"输出 B: {result_b}")


if __name__ == "__main__":
    main()