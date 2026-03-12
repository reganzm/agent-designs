"""
routing_langchain.py - 使用 LangChain 的智能体路由示例（DeepSeek + 流式输出）

================================================================================
一、功能概述
================================================================================
本脚本用 LangChain LCEL 实现一个「协调者 + 多处理器」的路由智能体：根据用户输入
自动判断应走预订、信息或未分类逻辑，调用对应处理器后，再让大模型生成面向用户的
自然语言回复，并可选择流式打印该回复。

与 routing_adk.py 的对应关系：逻辑上等价于 ADK 的「协调者委派子智能体 + 子智能体
由 LLM 生成回复」，此处用 RunnableBranch 做路由、用 Python 函数模拟子智能体行为、
再用单独一条 LLM 链生成最终回复。

================================================================================
二、流程说明（两步）
================================================================================
1) 路由与处理
   - 用户请求 request 传入后，先经过「协调者路由链」：
     - 大模型（DeepSeek）根据 system 提示只输出一个词：booker / info / unclear。
     - RunnableBranch 根据该词选择分支，调用对应处理器：
       - booker → booking_handler：模拟机票/酒店预订。
       - info   → info_handler：模拟通用信息/问答。
       - unclear → unclear_handler：无法归类时的提示。
   - 处理器返回的是「内部结果」字符串（如“预订处理器已处理请求：…”），用于后续
     生成回复，不直接作为面向用户的最终说法。

2) 大模型生成回复
   - 将「用户请求」与「处理器内部结果」一起交给第二条链（reply_chain）。
   - 大模型根据这两部分生成一两句自然、面向用户的中文回复。
   - 若 stream=True，该回复以流式方式逐 chunk 打印；否则一次性打印。

================================================================================
三、主要组件
================================================================================
- coordinator_router_chain：协调者路由链。输入 request，输出 booker/info/unclear。
- coordinator_agent：路由 + 委派 + 取处理器输出。输入 request，输出 handler 的内部结果。
- reply_chain：回复生成链。输入 request + handler_result，输出大模型生成的自然语言回复。
- run_coordinator(request, stream=False)：对外入口。执行上述两步并返回最终回复字符串。

================================================================================
四、依赖与配置
================================================================================
- 安装：pip install langchain langchain-deepseek langchain-core python-dotenv
- 环境变量：DEEPSEEK_API_KEY（必填，见 https://platform.deepseek.com/api_keys）
- 模型：deepseek-chat（可在本文件中修改 llm 初始化）
"""

import os

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
from langchain_core.runnables import RunnablePassthrough, RunnableBranch

# --- 初始化 DeepSeek 模型 ---
llm = ChatDeepSeek(model="deepseek-chat", temperature=0)

# =============================================================================
# 子智能体处理器（等价于 ADK 的 sub_agents；此处用函数模拟，可替换为真实 API/工具调用）
# =============================================================================

def booking_handler(request: str) -> str:
    """模拟预订智能体处理请求。"""
    print("\n--- 委派到预订处理器 ---")
    return f"预订处理器已处理请求：「{request}」。结果：已模拟完成预订操作。"

def info_handler(request: str) -> str:
    """模拟信息智能体处理请求。"""
    print("\n--- 委派到信息处理器 ---")
    return f"信息处理器已处理请求：「{request}」。结果：已模拟完成信息检索。"

def unclear_handler(request: str) -> str:
    """处理无法被委派的请求。"""
    print("\n--- 处理无法委派的请求 ---")
    return f"协调者无法委派该请求：「{request}」。请说明得更具体一些。"

# =============================================================================
# 协调者路由链：大模型根据用户请求输出 booker / info / unclear 之一
# =============================================================================
coordinator_router_prompt = ChatPromptTemplate.from_messages([
    ("system", """分析用户的请求，判断应由哪个专业处理器处理。
     - 若与预订机票、酒店相关，仅输出：booker。
     - 若为其他一般性信息、问答类请求，仅输出：info。
     - 若请求不清晰或不属于以上两类，仅输出：unclear。
     只输出一个词：booker、info 或 unclear。"""),
    ("user", "{request}")
])

coordinator_router_chain = coordinator_router_prompt | llm | StrOutputParser()

# =============================================================================
# 委派逻辑：根据路由结果选择对应处理器，并汇总为统一结构 {"request", "output"}
# =============================================================================
branches = {
    "booker": RunnablePassthrough.assign(output=lambda x: booking_handler(x["request"]["request"])),
    "info": RunnablePassthrough.assign(output=lambda x: info_handler(x["request"]["request"])),
    "unclear": RunnablePassthrough.assign(output=lambda x: unclear_handler(x["request"]["request"])),
}

delegation_branch = RunnableBranch(
    (lambda x: x["decision"].strip() == "booker", branches["booker"]),
    (lambda x: x["decision"].strip() == "info", branches["info"]),
    branches["unclear"],
)

# 完整协调者链：输入 request → 得到 decision + request → 分支执行 → 输出为处理器返回的内部结果
coordinator_agent = (
    {"decision": coordinator_router_chain, "request": RunnablePassthrough()}
    | delegation_branch
    | (lambda x: x["output"])
)

# =============================================================================
# 回复生成链：根据「用户请求 + 处理器内部结果」由大模型生成面向用户的自然语言回复（可流式）
# =============================================================================
reply_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是助手。根据「用户请求」和「处理结果」用一两句中文生成对用户的自然回复，直接说人话、不要复述系统说明。"),
    ("user", "用户请求：{request}\n处理结果：{handler_result}\n请生成对用户的回复："),
])
reply_chain = reply_prompt | llm | StrOutputParser()

# =============================================================================
# 调用入口：先执行路由与处理器，再调用大模型生成回复；可选流式打印
# =============================================================================
def run_coordinator(request: str, stream: bool = False) -> str:
    """执行协调者路由并返回最终结果。

    - 先路由到处理器得到内部结果，再让大模型根据该结果生成面向用户的回复（与 ADK 子智能体行为一致）。
    - stream=True 时流式打印大模型生成的回复；否则一次性打印。
    """
    # 步骤 1：路由 + 委派，得到处理器的内部结果字符串
    handler_result = coordinator_agent.invoke({"request": request})
    print('inner message:',request, handler_result)
    # 步骤 2：将「用户请求 + 内部结果」交给大模型，生成面向用户的自然语言回复
    reply_input = {"request": request, "handler_result": handler_result}
    if stream:
        print("回复: ", end="", flush=True)
        full = ""
        for chunk in reply_chain.stream(reply_input):
            if chunk:
                print(chunk, end="", flush=True)
                full += chunk
        print()
        return full
    result = reply_chain.invoke(reply_input)
    print(f"回复: {result}")
    return result

# =============================================================================
# 使用示例
# =============================================================================
def main():
    print("--- LangChain 路由示例（DeepSeek，支持流式）---")
    print("说明: 已配置 DEEPSEEK_API_KEY，模型为 deepseek-chat。\n")

    # 预订类请求（流式输出）
    print("--- 预订类请求（流式）---")
    run_coordinator("帮我订一张去伦敦的机票。", stream=True)

    # 信息类请求（流式输出）
    print("\n--- 信息类请求（流式）---")
    run_coordinator("意大利的首都是哪里？", stream=True)

    # 未分类请求（非流式）
    print("\n--- 未分类请求（非流式）---")
    run_coordinator("随便讲讲量子物理。", stream=False)

if __name__ == "__main__":
    main()
