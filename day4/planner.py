"""Day 4：任务拆解与规划 —— Planner-Executor 模式。

Day 3 的 agent 是 ReAct 风格：模型一步接一步，靠工具反馈自己往前走。
弱点是**没有全局观**——第一步抓错方向，会一路错到底，事后才知道。

Planner-Executor 强制它先想清楚再动手：

    用户任务 -> Planner(拆步骤) -> Executor(逐步执行) -> Synthesizer(综合结论)

代价是 N+2 次调用。单步任务上这是纯浪费，所以本文件同时实现**条件路由**：
简单问题走单 agent，复杂问题才拆解。

四个必须理解的设计点：

1. **Planner 的输出同样是模型输出，和工具参数一样不可信。**
   必须过 Pydantic 校验，且要额外校验「步骤数上限」「id 唯一」「依赖无环」。

2. **每个 Executor 步骤是独立的 agent 调用，构造全新的 messages。**
   不要把 planner 的完整对话历史传给每个 executor，否则第 3 步会看到
   第 1 步的完整推理过程，容易混淆「当前该做什么」。

3. **最终答案由 Synthesizer 生成，不是直接用 plan。**
   planner 可能拆错。如果直接拿 plan 当结论，错误会被固化。
   综合器看到的是「计划 + 每步实际结果」，它能发现某步失败了并如实说明。

4. **步骤之间的依赖要显式建模。**
   第 2 步依赖第 1 步的结果，就必须等第 1 步完成。
   独立的步骤可以并发——这才是 planner-executor 真正的性能收益。
"""

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pydantic import BaseModel, Field, ValidationError, model_validator

from day3.retry import RetryExhausted, call_with_retry, describe_error
from tools import ai_model_call

MAX_PLAN_STEPS = 5
MIN_PLAN_STEPS = 2
MAX_PARALLEL_WORKERS = 4


# ---------------------------------------------------------------------------
# 计划的数据结构
# ---------------------------------------------------------------------------


class PlanStep(BaseModel):
    """一个可执行步骤。

    depends_on 用步骤 id 表示依赖关系。id 较小的先执行。
    """

    id: int = Field(description="步骤序号，从 1 开始")
    goal: str = Field(description="这一步要达成的目标，必须具体且可验证")
    expected_output: str = Field(description="这一步应该产出什么，用于事后核对")
    depends_on: list[int] = Field(default_factory=list, description="依赖的步骤 id 列表")


class Plan(BaseModel):
    """完整计划。

    校验分两层：
    - Pydantic 字段级校验（类型、必填、取值范围）
    - model_validator 做的结构级校验（id 唯一、依赖合法、无环）

    第二层是 Day 2 学到的核心：**模型输出必须校验**。
    这里额外做结构校验，是因为「步骤 3 依赖步骤 3 自己」这种错误
    Pydantic 的字段类型检查抓不住，但会让执行阶段死锁。
    """

    task: str = Field(description="对原始任务的简述")
    steps: list[PlanStep] = Field(description="按顺序执行的步骤")

    @model_validator(mode="after")
    def validate_structure(self) -> "Plan":
        """结构级校验：id 唯一、依赖存在、无环、步骤数在范围内。"""

        # 1. 步骤数上下限
        if len(self.steps) < MIN_PLAN_STEPS:
            raise ValueError(f"计划至少需要 {MIN_PLAN_STEPS} 个步骤，实际 {len(self.steps)} 个")

        if len(self.steps) > MAX_PLAN_STEPS:
            raise ValueError(f"计划最多允许 {MAX_PLAN_STEPS} 个步骤，实际 {len(self.steps)} 个")

        # 2. id 必须唯一且从 1 连续递增
        ids = [step.id for step in self.steps]
        if len(ids) != len(set(ids)):
            raise ValueError(f"步骤 id 重复：{ids}")

        if sorted(ids) != list(range(1, len(ids) + 1)):
            raise ValueError(f"步骤 id 必须是 1 到 {len(ids)} 的连续整数，实际 {sorted(ids)}")

        # 3. 依赖必须指向存在的步骤，且不能指向自己或后面的步骤
        id_set = set(ids)

        for step in self.steps:
            for dependency in step.depends_on:
                if dependency not in id_set:
                    raise ValueError(f"步骤 {step.id} 依赖了不存在的步骤 {dependency}")

                if dependency == step.id:
                    raise ValueError(f"步骤 {step.id} 不能依赖自己")

                if dependency > step.id:
                    raise ValueError(
                        f"步骤 {step.id} 依赖了后面的步骤 {dependency}，"
                        f"只能依赖 id 更小的步骤（否则会形成环）"
                    )

        return self


# ---------------------------------------------------------------------------
# Planner：把任务拆成步骤
# ---------------------------------------------------------------------------


def make_plan(task: str) -> Plan:
    """调用模型生成计划。

    注意这里用 use_tools=False —— 规划阶段不该让模型调工具，
    否则模型可能返回 tool_calls 而我们只期待一段 JSON。
    """
    schema = Plan.model_json_schema()

    def request() -> object:
        return ai_model_call(
            stream=False,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是任务规划器。把用户任务拆成 2 到 5 个可执行步骤。\n"
                        "规则：\n"
                        "1. 步骤之间不重叠，前一步的输出是后一步的输入\n"
                        "2. 每一步的目标必须具体、可验证\n"
                        "3. 独立于其他步骤的步骤，depends_on 留空数组\n"
                        "4. 只输出 JSON，不要输出 Markdown 代码块\n"
                        "必须严格符合以下 JSON Schema：\n"
                        + json.dumps(schema, ensure_ascii=False)
                    ),
                },
                {
                    "role": "user",
                    "content": f"任务：{task}",
                },
            ],
            response_format={"type": "json_object"},
            use_tools=False,
        )

    def on_retry(attempt: int, error: BaseException, delay: float) -> None:
        print(f"  [规划] API 失败（第 {attempt} 次）：{describe_error(error)}，{delay:.2f}s 后重试")

    try:
        response = call_with_retry(request, on_retry=on_retry)
    except RetryExhausted as error:
        raise RuntimeError(f"规划失败：{describe_error(error.last_error)}") from error

    content = response.choices[0].message.content

    if not content:
        raise RuntimeError("规划器返回了空内容")

    try:
        return Plan.model_validate_json(content)
    except ValidationError as error:
        # 规划失败不是灾难，可以退回单 agent 路径
        raise RuntimeError(f"计划校验失败：{error}") from error


# ---------------------------------------------------------------------------
# Executor：执行单个步骤
# ---------------------------------------------------------------------------


def run_step(task: str, step: PlanStep, context: str, tools_enabled: bool = True) -> str:
    """执行一个步骤，返回这一步的结论。

    **全新的 messages**——只带「总任务 + 当前步骤 + 已完成步骤的结论」。
    不传 planner 的历史，不传其他步骤的推理过程。
    """
    messages = [
        {
            "role": "system",
            "content": (
                "你是执行者，只负责完成当前这一步。"
                "需要数据时调用工具。"
                "只输出这一步的结论，不要输出与当前步骤无关的内容。"
            ),
        },
        {
            "role": "user",
            "content": (
                f"总任务：{task}\n"
                f"当前步骤（第 {step.id} 步）：{step.goal}\n"
                f"这一步预期产出：{step.expected_output}\n"
                f"已完成步骤的结论：\n{context or '（无）'}\n\n"
                "请完成这一步，只输出这一步的结论。"
            ),
        },
    ]

    if not tools_enabled:
        response = ai_model_call(stream=False, messages=messages, use_tools=False)
        return response.choices[0].message.content or ""

    # 复用 Day 3 的带重试与自愈的 agent 循环
    from day3.self_healing import run_agent

    return run_agent(messages)


def resolve_context(step: PlanStep, results: dict[int, str]) -> str:
    """把依赖步骤的结论拼成上下文。"""
    if not step.depends_on:
        return ""

    parts = []

    for dependency in sorted(step.depends_on):
        outcome = results.get(dependency)
        if outcome is None:
            parts.append(f"第 {dependency} 步：（未完成）")
        else:
            parts.append(f"第 {dependency} 步结论：{outcome}")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# 并发执行：独立的步骤同时跑
# ---------------------------------------------------------------------------


def group_by_dependency(plan: Plan) -> list[list[int]]:
    """把步骤分组：同一组内互相独立，可以并发执行。

    返回若干「批次」，批次之间有依赖关系，必须顺序执行。
    组内步骤无依赖，可以并发。

    这就是 planner-executor 真正的性能收益所在——
    3 个独立步骤并行跑，耗时接近 1 个步骤。
    """
    remaining = {step.id: set(step.depends_on) for step in plan.steps}
    batches: list[list[int]] = []
    completed: set[int] = set()

    while remaining:
        # 找出依赖已全部满足的步骤
        ready = sorted(
            step_id
            for step_id, deps in remaining.items()
            if deps <= completed
        )

        if not ready:
            # 校验层已经拦过环，这里是兜底
            raise RuntimeError(f"依赖无法满足，可能存在环：{remaining}")

        batches.append(ready)
        completed |= set(ready)

        for step_id in ready:
            del remaining[step_id]

    return batches


def execute_plan(plan: Plan, task: str) -> dict[int, str]:
    """执行整个计划，返回 {步骤 id: 结论}。"""
    results: dict[int, str] = {}
    step_by_id = {step.id: step for step in plan.steps}

    batches = group_by_dependency(plan)

    for batch_index, batch in enumerate(batches, start=1):
        print(f"\n--- 批次 {batch_index}/{len(batches)}：并行执行步骤 {batch} ---")

        def execute_one(step_id: int) -> tuple[int, str]:
            step = step_by_id[step_id]
            context = resolve_context(step, results)
            return step_id, run_step(task, step, context)

        if len(batch) == 1:
            # 单个步骤没必要开线程池
            step_id, outcome = execute_one(batch[0])
            results[step_id] = outcome
            print(f"  步骤 {step_id} 完成")
        else:
            # 独立步骤并发执行
            with ThreadPoolExecutor(max_workers=min(MAX_PARALLEL_WORKERS, len(batch))) as executor:
                for step_id, outcome in executor.map(execute_one, batch):
                    results[step_id] = outcome
                    print(f"  步骤 {step_id} 完成")

    return results


# ---------------------------------------------------------------------------
# Synthesizer：综合各步结果
# ---------------------------------------------------------------------------


def synthesize(task: str, plan: Plan, results: dict[int, str]) -> str:
    """根据计划和每步实际结果给出最终答案。

    关键：综合器必须能看到「哪步失败了」。
    如果只给它成功的结论，它会编一个圆满的答案。
    """
    steps_report = []

    for step in plan.steps:
        outcome = results.get(step.id, "（该步骤没有产出结果）")
        steps_report.append({
            "id": step.id,
            "goal": step.goal,
            "expected_output": step.expected_output,
            "actual_output": outcome,
        })

    def request() -> object:
        return ai_model_call(
            stream=False,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是总结者。根据计划和每一步的实际结果给出最终答案。\n"
                        "硬性要求：\n"
                        "1. 如果某一步失败或信息不足，必须明确指出，不能编造\n"
                        "2. 如果实际产出与预期产出不符，要说明差异\n"
                        "3. 只输出最终结论，不要重复罗列全部中间过程"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"任务：{task}\n\n"
                        f"计划：{json.dumps([s.model_dump() for s in plan.steps], ensure_ascii=False)}\n\n"
                        f"各步实际结果：\n"
                        + json.dumps(steps_report, ensure_ascii=False, indent=2)
                    ),
                },
            ],
            use_tools=False,
        )

    try:
        response = call_with_retry(request)
    except RetryExhausted as error:
        return f"（综合阶段调用 API 连续失败：{describe_error(error.last_error)}）"

    return response.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# 条件路由：简单任务不值得拆
# ---------------------------------------------------------------------------


def classify_task(task: str) -> tuple[bool, str]:
    """判断任务是否值得拆解。返回 (是否需要规划, 理由)。

    这是纯本地的启发式判断，不额外调用模型——因为路由本身
    就在省调用次数，再调一次模型来省调用是自相矛盾的。

    判据（任一命中就认为需要拆解）：
    - 明确的多步信号词
    - 长度较长（复杂问题通常更长）
    - 包含比较、评估类动词
    """
    text = task.lower()

    multi_step_signals = [
        "然后", "接着", "之后", "再", "并且", "同时",
        "分别", "各自", "每个", "所有",
    ]

    analysis_signals = [
        "比较", "对比", "评估", "分析", "优缺点", "选型",
        "调研", "总结", "综述", "梳理",
    ]

    if any(signal in text for signal in analysis_signals):
        return True, "包含分析/比较类动词，需要分维度考察"

    if any(signal in text for signal in multi_step_signals):
        return True, "包含多步执行信号词"

    if len(task) > 60:
        return True, "任务描述较长，可能包含多个子目标"

    if len(task) < 15:
        return False, "任务很短，大概率是单步查询"

    return False, "未检测到多步特征，按单步处理"


# ---------------------------------------------------------------------------
# 组装：完整的 Planner-Executor Agent
# ---------------------------------------------------------------------------


def run_planner_agent(task: str) -> str:
    """完整的 Planner-Executor 流程。"""
    try:
        plan = make_plan(task)
    except RuntimeError as error:
        print(f"  规划失败（{error}），退回单步执行")
        return run_step(
            task,
            PlanStep(id=1, goal=task, expected_output="对任务的直接回答"),
            "",
        )

    print("=== 计划 ===")
    for step in plan.steps:
        dependency = f"，依赖 {step.depends_on}" if step.depends_on else ""
        print(f"  {step.id}. {step.goal}{dependency}")

    results = execute_plan(plan, task)

    print("\n=== 综合结论 ===")
    return synthesize(task, plan, results)


def run_routed_agent(task: str) -> str:
    """条件路由入口。

    简单任务走单 agent（1..N 次调用），
    复杂任务走 planner-executor（N+2 次调用）。
    """
    needs_planning, reason = classify_task(task)

    print(f"[路由] {'需要拆解' if needs_planning else '单步处理'}：{reason}")

    if not needs_planning:
        return run_step(
            task,
            PlanStep(id=1, goal=task, expected_output="对任务的直接回答"),
            "",
        )

    return run_planner_agent(task)


def main() -> None:
    print("输入任务，Agent 会自动判断是否需要拆解。输入 exit 退出。\n")

    while True:
        task = input("任务：").strip()

        if task.lower() in {"exit", "quit"}:
            break

        if not task:
            continue

        answer = run_routed_agent(task)
        print(f"\n最终答案：{answer}\n")


if __name__ == "__main__":
    main()
