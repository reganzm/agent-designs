"""
function_calling.py - LangChain 工具调用示例（DeepSeek）

说明：LangChain 1.x 已移除 AgentExecutor / create_tool_calling_agent，
改用 langchain.agents.create_agent（基于 LangGraph）。

依赖：pip install langchain langchain-deepseek langchain-core langgraph python-dotenv nest-asyncio
环境变量：DEEPSEEK_API_KEY
"""

import asyncio
import os

import nest_asyncio
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool as langchain_tool
from langchain_deepseek import ChatDeepSeek

load_dotenv()

if not os.getenv("DEEPSEEK_API_KEY"):
    raise RuntimeError(
        "未设置 DEEPSEEK_API_KEY。请在 .env 中配置或设置环境变量，"
        "申请地址：https://platform.deepseek.com/api_keys"
    )

llm = ChatDeepSeek(model="deepseek-chat", temperature=0)
print(f"已初始化语言模型：deepseek-chat")


@langchain_tool
def search_information(query: str) -> str:
    """
    根据查询主题返回事实性信息。当用户询问类似「法国的首都是哪里」「伦敦天气如何」
    「地球人口」等问题时，应使用本工具检索。
    """
    print(f"\n--- 工具调用：search_information，查询：「{query}」---")
    q = query.lower().strip()
    # 中英文关键词匹配（模型传入的 query 可能是中文或英文）
    if "伦敦" in query or "london" in q:
        if "天气" in query or "weather" in q:
            result = "伦敦目前多云，气温约 15°C。"
        else:
            result = "伦敦是英国的首都，著名城市。"
    elif "法国" in query or "france" in q:
        if "首都" in query or "capital" in q:
            result = "法国的首都是巴黎。"
        else:
            result = "法国是西欧国家，首都巴黎。"
    elif "人口" in query and ("地球" in query or "世界" in query or "earth" in q):
        result = "地球人口估计约 80 亿。"
    elif ("最高" in query and "山" in query) or ("tallest" in q and "mountain" in q):
        result = "珠穆朗玛峰是海拔最高的山峰。"
    else:
        result = f"（模拟检索）关于「{query}」暂无预设条目，但话题值得关注。"
    print(f"--- 工具返回：{result} ---")
    return result


tools = [search_information]

SYSTEM_PROMPT = (
    "你是一个乐于助人的助手。需要事实类信息时，请调用提供的搜索工具；"
    "回答请使用中文，简洁准确。"
)

# LangChain 1.x：create_agent 返回可编译图，输入为 {"messages": [...]}
agent_graph = create_agent(
    model=llm,
    tools=tools,
    system_prompt=SYSTEM_PROMPT,
    debug=True,
)


def _final_text_from_agent_result(result: dict) -> str:
    """从图状态里取出最后一条 AI 回复文本。"""
    messages = result.get("messages") or []
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            return str(msg.content)
    return str(result)


async def run_agent_with_tool(query: str):
    """执行智能体并打印最终输出。"""
    print(f"\n--- 运行智能体，用户问题：「{query}」---")
    try:
        result = await agent_graph.ainvoke(
            {"messages": [HumanMessage(content=query)]}
        )
        print("\n--- 智能体最终回复 ---")
        print(_final_text_from_agent_result(result))
    except Exception as e:
        print(f"\n执行出错：{e}")


async def main():
    """并发运行多个示例查询。"""
    tasks = [
        run_agent_with_tool("法国的首都是哪里？"),
        run_agent_with_tool("伦敦天气怎么样？"),
        run_agent_with_tool("随便讲讲狗这种动物。"),  # 可能走模拟检索的默认结果
    ]
    await asyncio.gather(*tasks)


nest_asyncio.apply()
asyncio.run(main())
