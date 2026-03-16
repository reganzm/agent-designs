"""
parallelization_adk.py - 使用 Google ADK + DeepSeek 实现智能体并行化处理

功能说明：
  - 三个「研究员」子智能体并行运行（可再生能源、电动汽车、碳捕获），各自调用模拟搜索工具并生成摘要。
  - 并行阶段结束后，由「合并」智能体将三份摘要综合成一份结构化报告。
  - 使用 SequentialAgent(ParallelAgent(三个研究员), 合并智能体) 组织流程；模型为 DeepSeek（LiteLLM）。

依赖与配置：
  - pip install "google-adk[extensions]" python-dotenv google-genai
  - 环境变量：DEEPSEEK_API_KEY（必填）
"""

import asyncio
import os
import uuid

from dotenv import load_dotenv

load_dotenv()

if not os.getenv("DEEPSEEK_API_KEY"):
    raise RuntimeError(
        "未设置 DEEPSEEK_API_KEY。请在 .env 中配置或设置环境变量，"
        "申请地址：https://platform.deepseek.com/api_keys"
    )

from google.adk.agents import LlmAgent, ParallelAgent, SequentialAgent
from google.adk.runners import InMemoryRunner
from google.adk.tools import FunctionTool
from google.genai import types
from google.adk.events import Event
from google.adk.models.lite_llm import LiteLlm

# 使用 DeepSeek 作为所有智能体的模型
deepseek_model = LiteLlm(model="deepseek/deepseek-reasoner")

# --- 模拟搜索工具（无 Google API 时可替代 google_search）---
def mock_search(query: str) -> str:
    """模拟搜索：根据关键词返回简短模拟结果，便于本地跑通并行研究流程。"""
    q = (query or "").strip().lower()
    if "可再生能源" in q or "renewable" in query:
        return "（模拟）近年来光伏与风电成本持续下降，储能配套加快，可再生能源占比提升。"
    if "电动汽车" in q or "electric vehicle" in query:
        return "（模拟）电动汽车续航与充电网络持续改善，智能驾驶与电池技术迭代加快。"
    if "碳捕获" in q or "carbon capture" in query:
        return "（模拟）碳捕获与封存技术试点扩大，与工业过程结合的应用增多。"
    return f"（模拟）与「{query}」相关的检索结果摘要。"

search_tool = FunctionTool(mock_search)

# --- 1. 定义三个并行研究员子智能体 ---
researcher_agent_1 = LlmAgent(
    name="RenewableEnergyResearcher",
    model=deepseek_model,
    instruction="""你是专注能源方向的 AI 研究助手。
请针对「可再生能源」相关议题，使用提供的搜索工具获取信息后，用 1～2 句话概括要点。
只输出这段概括，不要其他解释。""",
    description="研究可再生能源方向并输出简短摘要。",
    tools=[search_tool],
)

researcher_agent_2 = LlmAgent(
    name="EVResearcher",
    model=deepseek_model,
    instruction="""你是专注交通与出行方向的 AI 研究助手。
请针对「电动汽车技术」相关议题，使用提供的搜索工具获取信息后，用 1～2 句话概括要点。
只输出这段概括，不要其他解释。""",
    description="研究电动汽车技术并输出简短摘要。",
    tools=[search_tool],
)

researcher_agent_3 = LlmAgent(
    name="CarbonCaptureResearcher",
    model=deepseek_model,
    instruction="""你是专注气候与减排方向的 AI 研究助手。
请针对「碳捕获技术」相关议题，使用提供的搜索工具获取信息后，用 1～2 句话概括要点。
只输出这段概括，不要其他解释。""",
    description="研究碳捕获技术并输出简短摘要。",
    tools=[search_tool],
)

# --- 2. 并行智能体：同时运行三个研究员 ---
parallel_research_agent = ParallelAgent(
    name="ParallelWebResearchAgent",
    sub_agents=[researcher_agent_1, researcher_agent_2, researcher_agent_3],
    description="并行运行三个研究方向的研究员子智能体，汇总多源信息。",
)

# --- 3. 合并智能体：根据三份摘要生成结构化报告 ---
# 注意：合并阶段需要三份并行结果，由运行逻辑从事件中收集后作为用户消息传入
merger_agent = LlmAgent(
    name="SynthesisAgent",
    model=deepseek_model,
    instruction="""你是负责整合研究结果的 AI 助手。
用户会提供三个方向的研究摘要（可再生能源、电动汽车、碳捕获）。
请仅基于这三段摘要，生成一份结构化报告，包含：
1. 可再生能源要点（基于第一段摘要）
2. 电动汽车要点（基于第二段摘要）
3. 碳捕获要点（基于第三段摘要）
4. 简要总结（1～2 句）
不要添加摘要中未出现的内容。只输出报告正文。""",
    description="将并行研究员的结果综合成一份结构化、有归因的报告。",
)

# --- 4. 主流程：先并行研究，再合并（通过运行逻辑分两阶段执行）---
# 因 ParallelAgent 子智能体在独立分支中运行，合并智能体无法直接读取其状态，
# 故在运行端先执行并行阶段、从事件中收集三份结果，再以用户消息形式调用合并智能体。
root_agent = parallel_research_agent
merger_agent_standalone = merger_agent


def _event_text(event: Event) -> str:
    """从事件中提取文本。"""
    if not event or not event.content:
        return ""
    if hasattr(event.content, "text") and event.content.text:
        return event.content.text
    if event.content.parts:
        return "".join(
            p.text or "" for p in event.content.parts if getattr(p, "text", None)
        )
    return ""


def _collect_parallel_results(runner: InMemoryRunner, user_id: str, session_id: str) -> dict[str, str]:
    """运行并行研究智能体，从事件流中按 author 收集每个研究员的最终回复。"""
    results = {}
    researcher_names = ("RenewableEnergyResearcher", "EVResearcher", "CarbonCaptureResearcher")
    try:
        for event in runner.run(
            user_id=user_id,
            session_id=session_id,
            new_message=types.Content(role="user", parts=[types.Part(text="请开始并行研究并输出各方向摘要。")]),
        ):
            author = (event.author or "").split(".")[-1]  # 分支可能为 Agent.SubAgent，取最后一段
            if author in researcher_names:
                text = _event_text(event)
                if text:
                    results[author] = text
    except Exception as e:
        print(f"并行研究阶段出错: {e}")
    return results


def run_parallel_pipeline(stream: bool = False) -> str:
    """
    执行「并行研究 + 合并」流程：
    1) 创建会话，运行并行研究智能体，收集三份摘要；
    2) 将三份摘要拼成一条用户消息，运行合并智能体得到最终报告。
    """
    print("--- ADK 并行研究 + 合并示例（DeepSeek）---\n")
    user_id = "user_parallel"
    session_id = str(uuid.uuid4())
    runner = InMemoryRunner(root_agent)

    asyncio.run(
        runner.session_service.create_session(
            app_name=runner.app_name, user_id=user_id, session_id=session_id
        )
    )

    # 阶段 1：并行研究，收集结果
    print("阶段 1：并行运行三个研究员子智能体 …")
    results = _collect_parallel_results(runner, user_id, session_id)
    for name, text in results.items():
        print(f"  [{name}]: {text[:80]}…" if len(text) > 80 else f"  [{name}]: {text}")
    if len(results) < 3:
        print("  警告：未收齐三份结果，将用已有内容继续合并。")

    # 构造合并阶段的用户消息
    r1 = results.get("RenewableEnergyResearcher", "（无）")
    r2 = results.get("EVResearcher", "（无）")
    r3 = results.get("CarbonCaptureResearcher", "（无）")
    merger_input = (
        "请根据以下三份研究摘要生成结构化报告（仅基于以下内容）：\n\n"
        "【可再生能源】\n" + r1 + "\n\n"
        "【电动汽车】\n" + r2 + "\n\n"
        "【碳捕获】\n" + r3
    )

    # 阶段 2：合并报告（新会话，只跑合并智能体）
    session_id_2 = str(uuid.uuid4())
    runner_merger = InMemoryRunner(merger_agent_standalone)
    asyncio.run(
        runner_merger.session_service.create_session(
            app_name=runner_merger.app_name, user_id=user_id, session_id=session_id_2
        )
    )
    print("\n阶段 2：合并智能体生成报告 …")
    if stream:
        print("报告: ", end="", flush=True)
    final_report = ""
    try:
        for event in runner_merger.run(
            user_id=user_id,
            session_id=session_id_2,
            new_message=types.Content(role="user", parts=[types.Part(text=merger_input)]),
        ):
            text = _event_text(event)
            if text:
                if stream:
                    print(text, end="", flush=True)
                final_report += text
            if event.is_final_response():
                if stream:
                    print()
                break
    except Exception as e:
        print(f"合并阶段出错: {e}")
        return ""
    if not stream:
        print("报告:\n" + final_report)
    return final_report


def main():
    run_parallel_pipeline(stream=True)


if __name__ == "__main__":
    main()
