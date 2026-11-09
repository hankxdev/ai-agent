# Python AI Agent 开发速成课

面向资深工程师的 7 天课程。核心目标不是"学会框架"，而是**手写一遍 Agent 的内核**，再回头看框架就知道它在做什么。

## 环境

- Python 3.12+，用 [uv](https://docs.astral.sh/uv/) 管理
- 模型：DeepSeek（`deepseek-flash`），OpenAI 兼容协议
- 依赖：`openai`、`pydantic`、`python-dotenv`

```bash
uv sync
uv run python day0/chat.py
```

`.env` 放在项目根目录（已被 git 忽略），模板见 `.env.example`：

```
API_KEY=sk-xxx
BASE_URL=https://api.deepseek.com
MODEL=deepseek-flash
```

## 目录结构

```
ai-agent/
├─ config.py            # 读取 .env，集中管理 API_KEY / BASE_URL / MODEL
├─ tools.py             # 工具定义、工具注册表、模型调用封装
├─ day0/chat.py         # 流式对话 + 多轮历史
├─ day1/
│  ├─ chat.py           # 多轮对话
│  ├─ structured.py     # JSON Output + Pydantic 校验
│  ├─ tool_calling.py   # 单次工具调用闭环
│  └─ agent.py          # 模型 → 工具 → 模型 循环
├─ day2/
│  ├─ streaming_agent.py # 流式 + 工具调用（参数分片重组）
│  └─ context.py         # 历史裁剪、摘要、token 统计
├─ day3/
│  ├─ retry.py           # 错误分类、指数退避 + 抖动、重试执行器
│  ├─ self_healing.py    # API 层重试 + 工具层自愈 + 死循环检测
│  ├─ test_retry.py      # 22 个单元测试（无需 API Key）
│  └─ test_self_healing_e2e.py  # 9 个端到端测试（假客户端驱动）
├─ day4/
│  ├─ planner.py         # Planner-Executor + 依赖分组并发 + 条件路由
│  └─ test_planner.py    # 20 个单元测试（无需 API Key）
└─ src/                 # 练习代码
```

## 课程进度

| 天 | 主题 | 关键产出 | 状态 |
|---|---|---|---|
| Day 0 | 环境 + 第一次 API 调用 | `day0/chat.py` | 完成 |
| Day 1 | 流式、多轮对话、异常处理 | `day1/chat.py` | 完成 |
| Day 2 | 结构化输出、Tool Calling、Agent 循环 | `day1/structured.py`、`day1/tool_calling.py`、`day1/agent.py` | 完成 |
| Day 3 | 流式+工具、上下文管理、重试自愈 | `day2/streaming_agent.py`、`day2/context.py`、`day3/self_healing.py` | 完成 |
| Day 4 | 任务拆解与规划（planner-executor） | `day4/planner.py` | 完成 |
| Day 5 | 异步并发与多 agent 编排 | 待写 | 未开始 |
| Day 6 | 封装 CLI / API，选一个框架对照 | 待写 | 未开始 |
| Day 7 | 实战项目收尾 | 待写 | 未开始 |

## 五条必须记住的规则

1. **决策权在模型，执行权在程序。** 模型只生成调用请求，函数永远由你的代码执行。
2. **模型的输出是不可信输入。** 工具参数、计划结构、最终答案，都要过 Pydantic 校验，白名单派发，异常要转成可回灌的信息。
3. **上下文是最贵的资源。** 工具返回内容是 token 黑洞，历史要按轮次裁剪，长对话用摘要。
4. **失败要变成信息，不是异常。** 工具失败要转成模型读得懂的结构化错误，模型才能自我纠正；确定性错误立即失败，瞬时故障才退避重试。
5. **拆解是有成本的例外路径。** Planner-Executor 换来全局可控，代价是 N+2 次调用。简单任务直接干，复杂任务才拆。

## Git 约定

每个阶段一次提交，提交信息说明该阶段做了什么：

```
feat(day3): 增加 planner-executor 任务拆解
fix(day2): 流式工具参数改为累加拼接
docs: 补充上下文管理笔记
```
