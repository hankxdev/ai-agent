"""Day 4：Planner-Executor 的单元测试。

计划校验、依赖分组、条件路由都是纯逻辑，不需要 API Key 就能完整验证。

跑法：
    uv run python day4/test_planner.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pydantic import ValidationError

from day4.planner import (
    MAX_PLAN_STEPS,
    Plan,
    PlanStep,
    classify_task,
    group_by_dependency,
    resolve_context,
)

# ---------------------------------------------------------------------------
# 计划结构校验
# ---------------------------------------------------------------------------


def make_plan(steps: list[dict], task: str = "测试任务") -> Plan:
    return Plan.model_validate({"task": task, "steps": steps})


def test_valid_plan_passes() -> None:
    plan = make_plan([
        {"id": 1, "goal": "收集数据", "expected_output": "数据集", "depends_on": []},
        {"id": 2, "goal": "分析数据", "expected_output": "分析报告", "depends_on": [1]},
    ])

    assert len(plan.steps) == 2
    assert plan.steps[1].depends_on == [1]


def test_duplicate_ids_rejected() -> None:
    """id 重复会让结果回填错位置，必须拦掉。"""
    try:
        make_plan([
            {"id": 1, "goal": "A", "expected_output": "a", "depends_on": []},
            {"id": 1, "goal": "B", "expected_output": "b", "depends_on": []},
        ])
    except ValidationError as error:
        assert "重复" in str(error)
    else:
        raise AssertionError("重复 id 应被拒绝")


def test_non_sequential_ids_rejected() -> None:
    """id 不连续会让 depends_on 的语义变得模糊。"""
    try:
        make_plan([
            {"id": 1, "goal": "A", "expected_output": "a", "depends_on": []},
            {"id": 3, "goal": "B", "expected_output": "b", "depends_on": []},
        ])
    except ValidationError as error:
        assert "连续" in str(error)
    else:
        raise AssertionError("非连续 id 应被拒绝")


def test_self_dependency_rejected() -> None:
    try:
        make_plan([
            {"id": 1, "goal": "A", "expected_output": "a", "depends_on": [1]},
            {"id": 2, "goal": "B", "expected_output": "b", "depends_on": []},
        ])
    except ValidationError as error:
        assert "依赖自己" in str(error)
    else:
        raise AssertionError("自依赖应被拒绝")


def test_forward_dependency_rejected() -> None:
    """依赖后面的步骤 = 环，执行阶段会死锁。"""
    try:
        make_plan([
            {"id": 1, "goal": "A", "expected_output": "a", "depends_on": [2]},
            {"id": 2, "goal": "B", "expected_output": "b", "depends_on": []},
        ])
    except ValidationError as error:
        assert "后面的步骤" in str(error)
    else:
        raise AssertionError("前向依赖应被拒绝")


def test_nonexistent_dependency_rejected() -> None:
    try:
        make_plan([
            {"id": 1, "goal": "A", "expected_output": "a", "depends_on": [99]},
            {"id": 2, "goal": "B", "expected_output": "b", "depends_on": []},
        ])
    except ValidationError as error:
        assert "不存在" in str(error)
    else:
        raise AssertionError("不存在的依赖应被拒绝")


def test_too_many_steps_rejected() -> None:
    """步骤数必须有上限，否则 planner 拆出 20 步就是灾难。"""
    steps = [
        {"id": i, "goal": f"步骤{i}", "expected_output": "x", "depends_on": []}
        for i in range(1, MAX_PLAN_STEPS + 2)
    ]

    try:
        make_plan(steps)
    except ValidationError as error:
        assert "最多允许" in str(error)
    else:
        raise AssertionError("步骤数超限应被拒绝")


def test_too_few_steps_rejected() -> None:
    """只有一步的计划不值得拆，直接走单 agent。"""
    try:
        make_plan([{"id": 1, "goal": "A", "expected_output": "a", "depends_on": []}])
    except ValidationError as error:
        assert "至少需要" in str(error)
    else:
        raise AssertionError("步骤数过少应被拒绝")


def test_missing_required_field_rejected() -> None:
    """expected_output 是必填——没有预期产出就无法事后核对。"""
    try:
        make_plan([{"id": 1, "goal": "A", "depends_on": []}])
    except ValidationError as error:
        assert "expected_output" in str(error)
    else:
        raise AssertionError("缺少必填字段应被拒绝")


# ---------------------------------------------------------------------------
# 依赖分组
# ---------------------------------------------------------------------------


def test_all_independent_steps_in_one_batch() -> None:
    """全部独立 -> 一个批次 -> 全部并发。"""
    plan = make_plan([
        {"id": 1, "goal": "A", "expected_output": "a", "depends_on": []},
        {"id": 2, "goal": "B", "expected_output": "b", "depends_on": []},
        {"id": 3, "goal": "C", "expected_output": "c", "depends_on": []},
    ])

    batches = group_by_dependency(plan)

    assert len(batches) == 1
    assert batches[0] == [1, 2, 3]


def test_linear_dependency_creates_sequential_batches() -> None:
    plan = make_plan([
        {"id": 1, "goal": "A", "expected_output": "a", "depends_on": []},
        {"id": 2, "goal": "B", "expected_output": "b", "depends_on": [1]},
        {"id": 3, "goal": "C", "expected_output": "c", "depends_on": [2]},
    ])

    batches = group_by_dependency(plan)

    assert batches == [[1], [2], [3]]


def test_mixed_dependency_groups_correctly() -> None:
    """步骤 1 完成���2 和 3 都可以开始，4 要等 2 和 3 都完成。"""
    plan = make_plan([
        {"id": 1, "goal": "A", "expected_output": "a", "depends_on": []},
        {"id": 2, "goal": "B", "expected_output": "b", "depends_on": [1]},
        {"id": 3, "goal": "C", "expected_output": "c", "depends_on": [1]},
        {"id": 4, "goal": "D", "expected_output": "d", "depends_on": [2, 3]},
    ])

    batches = group_by_dependency(plan)

    assert batches == [[1], [2, 3], [4]]


# ---------------------------------------------------------------------------
# 上下文传递
# ---------------------------------------------------------------------------


def test_resolve_context_includes_dependencies() -> None:
    step = PlanStep(id=3, goal="C", expected_output="c", depends_on=[1, 2])
    results = {1: "第一步结果", 2: "第二步结果"}

    context = resolve_context(step, results)

    assert "第 1 步结论：第一步结果" in context
    assert "第 2 步结论：第二步结果" in context


def test_resolve_context_handles_missing_dependency() -> None:
    """依赖步骤没完成时要如实标注，不能静默当成空。"""
    step = PlanStep(id=2, goal="B", expected_output="b", depends_on=[1])
    context = resolve_context(step, {})

    assert "未完成" in context


def test_resolve_context_empty_for_no_dependencies() -> None:
    step = PlanStep(id=1, goal="A", expected_output="a")
    assert resolve_context(step, {1: "x"}) == ""


# ---------------------------------------------------------------------------
# 条件路由
# ---------------------------------------------------------------------------


def test_short_task_routes_to_single_step() -> None:
    needs_planning, _ = classify_task("现在几点")
    assert needs_planning is False


def test_analysis_task_routes_to_planner() -> None:
    needs_planning, _ = classify_task("比较一下 FastAPI 和 Django 的优缺点")
    assert needs_planning is True


def test_multi_step_signal_routes_to_planner() -> None:
    needs_planning, _ = classify_task("先查当前时间，然后算 128 加 45，最后把结果合并成一句话")
    assert needs_planning is True


def test_long_task_routes_to_planner() -> None:
    long_task = "分析一下我们团队过去三个季度的交付效率，指出主要瓶颈并给出改进建议"
    needs_planning, _ = classify_task(long_task)
    assert needs_planning is True


def test_medium_task_routes_to_single_step() -> None:
    """不明显的任务默认走单步——planner-executor 是例外路径，不是默认路径。"""
    needs_planning, _ = classify_task("帮我写一个 Python 函数来反转字符串")
    assert needs_planning is False


# ---------------------------------------------------------------------------
# 测试运行器
# ---------------------------------------------------------------------------


def _run_all() -> None:
    tests = [
        (name, value)
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]

    passed = 0
    failed: list[tuple[str, BaseException]] = []

    for name, test in tests:
        try:
            test()
            passed += 1
            print(f"  PASS  {name}")
        except BaseException as error:
            failed.append((name, error))
            print(f"  FAIL  {name}: {type(error).__name__}: {error}")

    print(f"\n{passed} passed, {len(failed)} failed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    _run_all()
