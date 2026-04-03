"""
crewai_like_langchain.py - 用 LangChain 模拟 CrewAI 风格的「智能体 + 任务 + 班组」编排

与 CrewAI 的对应关系：
  - Agent(role, goal, backstory, verbose, allow_delegation, llm)
  - Task(description, expected_output, agent)
  - Crew(agents, tasks, process=sequential) → kickoff()

当前实现：
  - Process.sequential：每个 Task 单独一次 LLM 调用；下一步会收到「前序步骤输出」作为上下文。
  - 多任务时 kickoff() 返回各步结果合并字符串（带任务标题），便于对照 CrewAI 的多 Task 执行。
  - 不实现 CrewAI 的委派、多智能体并行、工具绑定等高级能力。

依赖：pip install langchain-deepseek langchain-core python-dotenv
环境变量：DEEPSEEK_API_KEY
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_deepseek import ChatDeepSeek

load_dotenv()

if not os.getenv("DEEPSEEK_API_KEY"):
    raise RuntimeError(
        "未设置 DEEPSEEK_API_KEY。请在 .env 中配置，"
        "申请地址：https://platform.deepseek.com/api_keys"
    )


class Process(str, Enum):
    """执行策略（与 CrewAI 命名对齐，目前仅实现 sequential）。"""

    sequential = "sequential"


@dataclass
class Agent:
    """类似 CrewAI 的 Agent：角色、目标、背景，可选独立 llm。"""

    role: str
    goal: str
    backstory: str
    verbose: bool = False
    allow_delegation: bool = False
    llm: Optional[BaseChatModel] = None


@dataclass
class Task:
    """类似 CrewAI 的 Task：描述、期望输出格式、负责的智能体。"""

    description: str
    expected_output: str
    agent: Agent
    name: str = ""
    """可选任务名，用于 verbose 日志与合并输出中的标题。"""


@dataclass
class Crew:
    """
    类似 CrewAI 的 Crew：智能体列表、任务列表、流程类型。
    kickoff() 按顺序执行每个 Task（每步一次模型调用），返回合并结果或单步结果。
    """

    agents: list[Agent]
    tasks: list[Task]
    process: Process = Process.sequential
    llm: Optional[BaseChatModel] = None
    """未在 Agent 上指定 llm 时使用的默认模型。"""

    def _resolve_llm(self, agent: Agent) -> BaseChatModel:
        if agent.llm is not None:
            return agent.llm
        if self.llm is not None:
            return self.llm
        raise ValueError("Crew 或 Agent 至少需要指定一个 llm")

    def _run_one_task(self, task: Task, prior_output: str) -> str:
        agent = task.agent
        llm = self._resolve_llm(agent)

        system = (
            f"你的角色：{agent.role}\n"
            f"你的目标：{agent.goal}\n"
            f"背景设定：{agent.backstory}\n"
        )
        if not agent.allow_delegation:
            system += "说明：你独立完成本任务，不要将工作委派给其他智能体。\n"

        user_parts = [
            "【任务说明】\n" + task.description.strip(),
            "\n【期望输出格式】\n" + task.expected_output.strip(),
        ]
        if prior_output.strip():
            user_parts.append(
                "\n【前序步骤输出（供你参考或延续）】\n" + prior_output.strip()
            )

        user_content = "\n".join(user_parts)

        chain = (
            ChatPromptTemplate.from_messages(
                [
                    ("system", "{system}"),
                    ("human", "{user}"),
                ]
            )
            | llm
            | StrOutputParser()
        )
        result = chain.invoke({"system": system, "user": user_content})

        title = (task.name or "未命名任务").strip()
        if agent.verbose:
            print(f"\n--- 执行任务：{title} | 智能体：{agent.role} ---\n{result}\n")
        return result

    def kickoff(self) -> str:
        """按 Process 执行：每个 Task 独立一次 invoke；多任务时返回带标题的合并文本。"""
        if self.process != Process.sequential:
            raise NotImplementedError(f"暂不支持 process={self.process!r}，请使用 Process.sequential")

        step_outputs: list[tuple[str, str]] = []
        prior = ""
        for task in self.tasks:
            out = self._run_one_task(task, prior)
            title = (task.name or f"任务 {len(step_outputs) + 1}").strip()
            step_outputs.append((title, out))
            prior = out

        if len(step_outputs) == 1:
            return step_outputs[0][1]
        parts = [f"【{title}】\n{body}" for title, body in step_outputs]
        sep = "─" * 40
        return f"\n\n{sep}\n\n" + f"\n\n{sep}\n\n".join(parts)


# ---------------------------------------------------------------------------
# 演示：两个独立 Task = 两次模型调用（先大纲，后摘要），行为接近 Crew 顺序多任务
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    llm = ChatDeepSeek(model="deepseek-chat", temperature=0.3)

    planner_writer_agent = Agent(
        role="文章规划与撰写专员",
        goal="先规划再撰写：就指定主题写出简洁、有吸引力的摘要。",
        backstory=(
            "你是资深技术写作者与内容策划。你擅长先列出清晰可执行的大纲再动笔，"
            "保证最终摘要既有信息量又易读。"
        ),
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    topic = "强化学习在人工智能中的重要性"

    # 任务 1：仅规划（一次 LLM 调用）
    task_outline = Task(
        name="步骤一：撰写大纲",
        description=(
            f"主题：「{topic}」。\n"
            "本步只做规划：列出后续撰写摘要时要覆盖的要点（项目符号列表）。\n"
            "不要写摘要正文，不要写开场白或结束语。"
        ),
        expected_output=(
            "只输出如下结构：\n\n"
            "### 大纲\n"
            "- 要点一\n"
            "- 要点二\n"
            "（依此类推）"
        ),
        agent=planner_writer_agent,
    )

    # 任务 2：仅撰写（第二次 LLM 调用，会收到任务 1 的完整输出作为「前序步骤输出」）
    task_summary = Task(
        name="步骤二：根据大纲写摘要",
        description=(
            f"主题仍为：「{topic}」。\n"
            "请**严格依据**「前序步骤输出」中的大纲要点，撰写一段中文摘要，约 200 字。\n"
            "本步只输出摘要正文，不要重复粘贴上一版大纲全文；可在摘要中自然体现要点即可。"
        ),
        expected_output=(
            "只输出如下结构：\n\n"
            "### 摘要\n"
            "（一段连贯的正文，约 200 字）"
        ),
        agent=planner_writer_agent,
    )

    crew = Crew(
        agents=[planner_writer_agent],
        tasks=[task_outline, task_summary],
        process=Process.sequential,
        llm=llm,
    )

    print("## 运行 Crew（2 个 Task = 2 次模型调用：先大纲 → 后摘要）##")
    result = crew.kickoff()

    print("\n---\n## kickoff() 合并返回（含各任务标题）##\n---")
    print(result)
    print(
        "\n提示：若 verbose=True，上面会先分任务打印一遍；"
        "不需要重复时可将 Agent.verbose 设为 False。"
    )
