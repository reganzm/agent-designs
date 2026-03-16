"""
parallelization_langchain.py - 使用 LangChain 的并行化智能体示例（DeepSeek）

功能说明：
  - 三条独立链（摘要、提问、关键词）通过 RunnableParallel 并行执行，结果再经一条综合链合并。
  - 模型使用 DeepSeek（langchain-deepseek），提示词为中文。

依赖与配置：
  - pip install langchain langchain-deepseek langchain-core python-dotenv
  - 环境变量：DEEPSEEK_API_KEY（必填，见 https://platform.deepseek.com/api_keys）
"""

import os
import asyncio
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

if not os.getenv("DEEPSEEK_API_KEY"):
    raise RuntimeError(
        "未设置 DEEPSEEK_API_KEY。请在 .env 中配置或设置环境变量，"
        "申请地址：https://platform.deepseek.com/api_keys"
    )

from langchain_deepseek import ChatDeepSeek
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableParallel, RunnablePassthrough

# --- 初始化 DeepSeek 模型 ---
llm = ChatDeepSeek(model="deepseek-chat", temperature=0.7)

# --- 定义三条可并行执行的独立链 ---
# 分别完成：简要摘要、生成问题、提取关键词
summarize_chain: Runnable = (
    ChatPromptTemplate.from_messages([
        ("system", "请对下面的话题做简明扼要的总结。"),
        ("user", "{topic}")
    ])
    | llm
    | StrOutputParser()
)

questions_chain: Runnable = (
    ChatPromptTemplate.from_messages([
        ("system", "针对下面的话题，生成三个有趣的问题。"),
        ("user", "{topic}")
    ])
    | llm
    | StrOutputParser()
)

terms_chain: Runnable = (
    ChatPromptTemplate.from_messages([
        ("system", "从下面的话题中提炼 5～10 个关键术语，用逗号分隔。"),
        ("user", "{topic}")
    ])
    | llm
    | StrOutputParser()
)

# --- 构建「并行 + 综合」链 ---
# 1. 并行执行三条链，并保留原始话题
map_chain = RunnableParallel(
    {
        "summary": summarize_chain,
        "questions": questions_chain,
        "key_terms": terms_chain,
        "topic": RunnablePassthrough(),
    }
)

# 2. 综合提示：将并行得到的摘要、问题、关键词合并成一段完整回答
synthesis_prompt = ChatPromptTemplate.from_messages([
    ("system", """根据以下信息进行综合回答：
     摘要：{summary}
     相关提问：{questions}
     关键术语：{key_terms}
     请整合成一段条理清晰的综合说明。"""),
    ("user", "原始话题：{topic}")
])

# 3. 完整链：并行结果 → 综合提示 → LLM → 解析输出
full_parallel_chain = map_chain | synthesis_prompt | llm | StrOutputParser()


# --- 运行链 ---
async def run_parallel_example(topic: str) -> None:
    """
    异步调用并行处理链，对给定话题执行并行任务并打印综合结果。

    Args:
        topic: 输入话题，将交由各条链并行处理。
    """
    print(f"\n--- 并行 LangChain 示例，话题：「{topic}」---")
    try:
        response = await full_parallel_chain.ainvoke(topic)
        print("\n--- 综合结果 ---")
        print(response)
    except Exception as e:
        print(f"\n执行过程中出错：{e}")


if __name__ == "__main__":
    test_topic = "太空探索的历史"
    asyncio.run(run_parallel_example(test_topic))
