import os
from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
from langchain_core.messages import SystemMessage, HumanMessage


# 安装依赖
# pip install langchain-deepseek langchain-core python-dotenv

# --- 配置 ---
load_dotenv()

if not os.getenv("DEEPSEEK_API_KEY"):
    raise RuntimeError(
        "未设置 DEEPSEEK_API_KEY。请在 .env 中配置或设置环境变量，"
        "申请地址：https://platform.deepseek.com/api_keys"
    )

# 使用 DeepSeek 对话模型；较低温度使输出更稳定、聚焦
llm = ChatDeepSeek(model="deepseek-chat", temperature=0.1)


def run_reflection_loop():
    """
    通过多步骤反思循环，逐步改进 Python 函数（生成 → 评审 → 再生成）。
    """

    task_prompt = """
    你的任务是编写一个名为 `calculate_factorial` 的 Python 函数，需满足：
    1. 接受一个整数参数 `n`。
    2. 计算其阶乘（n!）。
    3. 包含清晰的 docstring，说明函数用途。
    4. 边界情况：0 的阶乘为 1。
    5. 非法输入：若输入为负数，抛出 ValueError。
    """

    max_iterations = 3
    current_code = ""
    message_history = [HumanMessage(content=task_prompt)]

    for i in range(max_iterations):
        print("\n" + "=" * 25 + f" 反思循环：第 {i + 1} 轮 " + "=" * 25)

        if i == 0:
            print("\n>>> 阶段 1：生成初始代码 …")
            response = llm.invoke(message_history)
            current_code = response.content
        else:
            print("\n>>> 阶段 1：根据上一轮评审意见改进代码 …")
            message_history.append(
                HumanMessage(content="请根据上面给出的评审意见改进代码。")
            )
            response = llm.invoke(message_history)
            current_code = response.content

        print("\n--- 当前代码（版本 " + str(i + 1) + "）---\n" + current_code)
        message_history.append(response)

        print("\n>>> 阶段 2：对代码进行反思与评审 …")

        reflector_prompt = [
            SystemMessage(
                content="""你是一名资深 Python 工程师，负责对代码做细致审查。
请对照原始任务要求，严格评估给出的 Python 代码。
查找缺陷、风格问题、遗漏的边界情况以及可改进之处。
若代码已完全符合要求，请仅回复短语：CODE_IS_PERFECT
否则，用分条列表写出你的评审意见。"""
            ),
            HumanMessage(
                content=f"原始任务：\n{task_prompt}\n\n待审查代码：\n{current_code}"
            ),
        ]

        critique_response = llm.invoke(reflector_prompt)
        critique = critique_response.content

        if "CODE_IS_PERFECT" in critique:
            print("\n--- 评审意见 ---\n无进一步问题，代码已满足要求。")
            break

        print("\n--- 评审意见 ---\n" + critique)
        message_history.append(
            HumanMessage(content=f"对上一版代码的评审意见：\n{critique}")
        )

    print("\n" + "=" * 30 + " 最终结果 " + "=" * 30)
    print("\n反思流程结束后的最终代码：\n")
    print(current_code)


if __name__ == "__main__":
    run_reflection_loop()
