# MiroThinker 面试深度准备指南

> 面向 LLM 算法实习岗（Agent 应用 / 后训练 方向）
> 基于 MiroThinker v1.0-v1.7 全系列代码与论文整理

---

## 目录

1. [项目全景：一句话能讲清楚的定位](#1-项目全景)
2. [系统架构：自顶向下的拆解](#2-系统架构)
3. [核心机制深度解析](#3-核心机制深度解析)
4. [后训练管线（SFT / DPO）](#4-后训练管线)
5. [Benchmark 评估体系](#5-benchmark-评估体系)
6. [高频面试问题 & 参考回答](#6-高频面试问题)
7. [行为面 / 项目介绍话术](#7-项目介绍话术)
8. [延伸阅读与关联论文](#8-延伸阅读)

---

## 1. 项目全景

### 1.1 一句话定位

**MiroThinker** 是一个面向深度研究任务（Deep Research）的开源 Agent 框架 + 后训练模型系列，在 BrowseComp、GAIA、HLE 等 benchmark 上达到开源 SOTA。

### 1.2 核心创新点（面试时要能脱口而出）

| 维度 | 传统做法 | MiroThinker 做法 |
|------|----------|-----------------|
| 模型规模 Scaling | 增大参数量 | 30B/235B MoE |
| 上下文长度 Scaling | 增大 context window | 256K context + **上下文管理** |
| **Interactive Scaling**（核心创新） | N/A | 训练 Agent 处理更深、更频繁的 **Agent-环境交互**，作为性能提升的第三维度 |

### 1.3 性能速览（背诵关键数字）

- **BrowseComp-ZH**: MiroThinker-1.7-mini (30B) 达到 72.3%，开源 SOTA
- **BrowseComp**: MiroThinker-1.7 达到 75.3%
- **GAIA-Val-165**: 82.7%
- **HLE-Text**: 42.9%
- 单任务支持最多 **300 次工具调用**，256K context window

---

## 2. 系统架构

### 2.1 项目结构一览

```
MiroThinker/
├── apps/
│   ├── miroflow-agent/          # 核心 Agent 框架（MiroFlow）
│   │   ├── src/
│   │   │   ├── core/            # 核心执行逻辑
│   │   │   │   ├── orchestrator.py     # 编排器：主循环
│   │   │   │   ├── pipeline.py         # 管线：初始化 + 入口
│   │   │   │   ├── tool_executor.py    # 工具执行器
│   │   │   │   ├── answer_generator.py # 答案生成 + 上下文管理
│   │   │   │   └── stream_handler.py   # 流式事件处理
│   │   │   ├── llm/             # LLM 客户端
│   │   │   │   ├── base_client.py      # 抽象基类
│   │   │   │   ├── providers/
│   │   │   │   │   ├── openai_client.py
│   │   │   │   │   └── anthropic_client.py
│   │   │   │   └── factory.py
│   │   │   ├── config/          # 配置管理（Hydra）
│   │   │   │   └── settings.py         # MCP 服务器参数 + 环境变量
│   │   │   ├── io/              # 输入输出处理
│   │   │   ├── utils/           # 工具函数
│   │   │   │   └── prompt_utils.py     # System Prompt 模板
│   │   │   └── logging/         # 日志与追踪
│   │   ├── conf/                # Hydra YAML 配置
│   │   │   ├── agent/           # Agent 配置（单/多代理）
│   │   │   ├── llm/             # LLM 配置
│   │   │   └── benchmark/       # 评估配置
│   │   ├── benchmarks/          # 评估管线
│   │   └── scripts/             # 运行脚本
│   ├── collect-trace/           # 训练数据收集
│   ├── gradio-demo/             # 可视化 Demo
│   └── visualize-trace/         # 轨迹可视化
├── libs/
│   └── miroflow-tools/          # MCP 工具库
│       └── src/miroflow_tools/
│           ├── manager.py               # 工具管理器
│           ├── mcp_servers/             # 标准 MCP 服务器
│           │   ├── python_mcp_server.py       # E2B 代码沙箱
│           │   ├── searching_google_mcp_server.py
│           │   ├── vision_mcp_server.py
│           │   └── ...
│           └── dev_mcp_servers/         # 开发/生产 MCP 服务器
│               ├── search_and_scrape_webpage.py  # 搜索+抓取
│               ├── jina_scrape_llm_summary.py    # Jina+LLM摘要
│               └── task_planner.py               # 任务规划器
└── README.md
```

### 2.2 核心数据流

```
用户问题
    │
    ▼
┌─────────────────────────────────────────────┐
│  Pipeline (pipeline.py)                     │
│  1. 创建 ToolManager（主代理 + 子代理）      │
│  2. 创建 LLM Client                         │
│  3. 创建 Orchestrator                        │
│  4. 调用 orchestrator.run_main_agent()       │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│  Orchestrator (orchestrator.py)             │
│                                             │
│  主循环 (while turn < max_turns):           │
│    1. LLM 生成响应（含/不含工具调用）        │
│    2. 解析工具调用                           │
│    3. 执行工具（ToolManager.execute_tool）   │
│    4. 检查重复查询 / 格式错误 / 拒绝关键词  │
│    5. 更新 message_history                   │
│    6. 检查上下文长度 → 必要时压缩           │
│                                             │
│  循环结束后 → AnswerGenerator 生成最终答案  │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│  Context Management (answer_generator.py)   │
│                                             │
│  if context_compress_limit > 0:             │
│    - 达到 max_turns → 跳过答案生成          │
│    - 生成 Failure Experience Summary        │
│    - 将失败经验注入下一轮 task_description  │
│    - 重试（最多 N 次）                      │
│  else:                                      │
│    - 直接生成答案 → 兜底用中间 boxed 答案   │
└─────────────────────────────────────────────┘
```

### 2.3 MCP 协议（Model Context Protocol）

MiroThinker 使用 **MCP** 作为工具调用协议，而不是原生 function calling。

**为什么用 MCP 而不是原生 function calling？**
1. MCP 是标准化的工具协议，支持 Stdio/SSE 两种传输方式
2. 工具定义和执行解耦：MCP Server 独立进程，Agent 通过协议调用
3. 便于扩展：新增工具只需新增 MCP Server，无需修改 Agent 代码
4. 支持工具黑名单机制

**System Prompt 中的工具调用格式：**

```xml
<use_mcp_tool>
<server_name>search_and_scrape_webpage</server_name>
<tool_name>google_search</tool_name>
<arguments>
{"q": "query string", "gl": "us"}
</arguments>
</use_mcp_tool>
```

> **面试关键点**：解释为什么选择 MCP 而不是 OpenAI function calling，以及 MCP 的 Stdio vs SSE 传输模式区别。

---

## 3. 核心机制深度解析

### 3.1 Interactive Scaling（最重要的创新）

**概念**：除了 model scaling（参数量）和 context scaling（上下文长度），Interactive Scaling 是性能提升的第三维度——通过训练 Agent 处理更深、更频繁的 Agent-环境交互。

**实现方式**：
1. **训练数据收集**：使用强模型（Claude-3.7/GPT-5）在 RLVR 数据集上跑 Agent，收集正确轨迹
2. **SFT 阶段**：用正确轨迹微调基座模型，学习工具调用模式
3. **DPO 阶段**：用正确/错误轨迹对做偏好优化
4. **评估时**：支持最多 300 次工具调用，256K 上下文

**为什么 Interactive Scaling 有效？**
- Agent 的能力不仅取决于模型本身的推理能力，还取决于与环境交互的效率
- 更深的交互意味着更多的信息获取和验证
- 通过训练，模型学会了在长链交互中保持一致性

> **面试深挖点**：
> - Interactive Scaling 和 Chain-of-Thought 有什么区别？
> - 如何判断什么时候该继续搜索 vs 停下来回答？
> - 如何处理长链交互中的错误累积？

### 3.2 Context Management（上下文管理）

这是 MiroThinker 的核心工程创新之一。

**问题**：在 ReAct 范式中，所有工具输出都保留在 message_history 中，导致：
- 上下文快速膨胀
- 模型注意力被远处的无关信息稀释
- 无法支持长链交互

**解决方案：Recency-based Context Retention**

```python
# base_client.py: _remove_tool_result_from_messages()
if keep_tool_result == -1:
    return messages_copy  # 保留所有
# 只保留最近 K 个工具响应，其余替换为 "Tool result is omitted to save tokens."
```

**配置参数**：
- `keep_tool_result: 5` — 只保留最近 5 个工具响应
- `context_compress_limit: 5` — 启用上下文压缩（最多重试 5 次）

**为什么有效？**
- Agent 的后续行动主要依赖最近的观察，而非远处的历史
- 保留完整的 thought-action 序列（只压缩 observation）
- 为更长的工具使用轨迹释放上下文空间

> **面试深挖点**：
> - 为什么不直接截断整个 message_history？
> - keep_tool_result 的值如何选择？太小会丢失什么信息？
> - 有没有更好的上下文管理策略？（如 summarization-based）

### 3.3 Failure Experience（失败经验机制）

**问题**：当 Agent 在给定的 turns 和上下文窗口内未能完成任务时，如何利用失败信息？

**解决方案**：结构化的失败经验压缩 + 注入

**Failure Summary 模板**：
```
Failure type: [incomplete / blocked / misdirected / format_missed]
  - incomplete: 用完了 turns 还没完成
  - blocked: 工具失败或信息缺失
  - misdirected: 走错了方向
  - format_missed: 找到答案但忘记用 \boxed{}
What happened: [描述采取的方法和为什么没有得到最终答案]
Useful findings: [列出发现的事实、中间结果、可复用的结论]
```

**重试流程**：
```
第 1 次尝试 → 失败 → 生成 Failure Summary
第 2 次尝试 → 将 Failure Summary 注入 task_description → 失败 → 生成新的 Failure Summary
第 3 次尝试 → 注入所有累积的 Failure Summaries → ...
```

**代码实现**（`answer_generator.py`）：
```python
async def generate_failure_summary(self, ...):
    failure_summary_history = message_history.copy()
    failure_summary_history.append({"role": "user", "content": FAILURE_SUMMARY_PROMPT})
    failure_summary_history.append({"role": "assistant", "content": FAILURE_SUMMARY_ASSISTANT_PREFIX})
    # 调用 LLM 生成结构化失败总结
```

> **面试深挖点**：
> - Failure type 的分类依据是什么？每种类型如何影响重试策略？
> - 如何避免失败经验中的错误信息误导后续尝试？
> - 这个机制和 ReAct 中的 Reflexion 有什么区别？

### 3.4 Rollback Mechanism（回滚机制）

**触发条件**：
1. MCP 格式错误（响应中包含 MCP XML 标签）
2. 拒绝关键词（如 "time constraint", "I'm sorry, but I can't"）
3. 重复查询（已经搜索过的 query）
4. 工具执行错误（Unknown tool, Error executing tool）
5. Google 搜索返回空结果

**回滚逻辑**（`orchestrator.py`）：
```python
if consecutive_rollbacks < MAX_CONSECUTIVE_ROLLBACKS - 1:
    turn_count -= 1
    consecutive_rollbacks += 1
    message_history.pop()  # 移除错误的 assistant 响应
else:
    # 达到最大连续回滚次数，强制结束
    break
```

> **面试深挖点**：
> - 为什么需要回滚而不是直接跳过？
> - 重复查询检测的实现细节？（基于 query 字符串匹配）
> - 如何平衡回滚次数和任务完成率？

### 3.5 Duplicate Query Detection（重复查询检测）

**目的**：避免 Agent 重复执行相同的搜索/抓取操作

**实现**（`tool_executor.py`）：
```python
def get_query_str_from_tool_call(self, tool_name, arguments):
    if tool_name == "google_search":
        return tool_name + "_" + arguments.get("q", "")
    elif tool_name == "scrape_and_extract_info":
        return tool_name + "_" + arguments.get("url", "") + "_" + arguments.get("info_to_extract", "")
    # ...
```

**缓存策略**：
- 每个工具调用的 query 字符串作为 key
- 使用 `defaultdict(int)` 记录调用次数
- 子代理有独立的缓存空间（`cache_name = sub_agent_id + "_" + tool_name`）

### 3.6 Sub-Agent Architecture（子代理架构）

**主代理（Main Agent）**：
- 负责任务分解、工具调用、答案生成
- 可以调用子代理作为"工具"

**子代理（Sub-Agent）**：
- `agent-browsing`：专门的搜索浏览代理
- 有独立的工具集、上下文、turn 限制
- 生成结构化报告返回给主代理

**子代理作为工具的定义**：
```python
def expose_sub_agents_as_tools(sub_agents_cfg):
    # 将子代理暴露为主代理的工具
    tool = {
        "name": "search_and_browse",
        "description": "This tool is an agent that performs the subtask of searching and browsing the web...",
        "schema": {"properties": {"subtask": {"type": "string"}}}
    }
```

> **面试深挖点**：
> - 什么时候该用单代理 vs 多代理？
> - 子代理的上下文和主代理是隔离的吗？
> - 如何避免主代理和子代理之间的信息丢失？

---

## 4. 后训练管线

### 4.1 整体流程

```
RLVR 数据集 (Question + Verifiable Answer)
    │
    ▼
┌─────────────────────────────────┐
│  Trace Collection (collect-trace)│
│  1. 加载 RLVR 数据              │
│  2. 用强模型（Claude/GPT-5）跑 Agent │
│  3. LLM-as-a-Judge 验证答案     │
│  4. 收集正确的多轮交互轨迹      │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  SFT (Supervised Fine-Tuning)   │
│  - 用正确轨迹微调基座模型       │
│  - 学习工具调用模式、推理链     │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  DPO (Direct Preference Opt.)   │
│  - 正确轨迹 vs 错误轨迹         │
│  - 优化模型偏好                  │
└─────────────────────────────────┘
```

### 4.2 Trace Collection 细节

**输入**：RLVR 格式数据集（Question + 可验证的 Answer）

**流程**：
1. 将 Question 作为 task_description 传入 Agent
2. Agent 执行多轮工具调用，生成最终答案
3. 使用 LLM-as-a-Judge 对比模型答案和 ground truth
4. 仅保留判断为 CORRECT 的轨迹

**输出格式**：
```json
{
  "task_id": "xxx",
  "task_description": "...",
  "system_prompt": "...",
  "message_history": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "...<use_mcp_tool>...</use_mcp_tool>"},
    {"role": "user", "content": "Tool result: ..."},
    ...
  ],
  "final_boxed_answer": "...",
  "ground_truth": "...",
  "final_judge_result": "CORRECT"
}
```

### 4.3 SFT 数据格式

将多轮交互轨迹转换为 SFT 训练样本：
- 每一轮 tool call 是一个训练样本
- System prompt + 历史对话作为 input
- 模型的响应（含工具调用）作为 target

### 4.4 DPO 数据格式

- **Chosen**：正确的轨迹（工具调用准确、最终答案正确）
- **Rejected**：错误的轨迹（工具调用错误、答案错误）
- 通过偏好优化，让模型学会选择更好的工具调用策略

> **面试深挖点**：
> - SFT 和 DPO 各自解决什么问题？
> - 如何保证收集到的轨迹质量？
> - LLM-as-a-Judge 的评判标准是什么？如何避免误判？
> - 为什么不用 RLHF 而用 DPO？

---

## 5. Benchmark 评估体系

### 5.1 评估的 Benchmark 列表

| Benchmark | 描述 | 评估方式 |
|-----------|------|----------|
| GAIA | General AI Assistants | LLM-as-a-Judge |
| HLE | Humanity's Last Exam | 精确匹配 |
| BrowseComp | 浏览理解 | 精确匹配 |
| BrowseComp-ZH | 中文浏览理解 | 精确匹配 |
| WebWalkerQA | Web 导航问答 | 精确匹配 |
| XBench-DeepSearch | 深度搜索 | LLM-as-a-Judge |
| FutureX | 未来预测 | LLM-as-a-Judge |
| AIME2025 | 数学竞赛 | 精确匹配 |

### 5.2 Pass@K 评估

**定义**：对同一个问题尝试 K 次，只要有一次正确就算通过。

**实现**（`common_benchmark.py`）：
```python
for attempt in range(1, self.pass_at_k + 1):
    # 执行任务
    result = await execute_task_pipeline(...)
    # 验证答案
    if is_correct:
        found_correct_answer = True
        break  # 早停
```

### 5.3 LLM-as-a-Judge

**用途**：验证模型答案是否正确

**实现**：
- 使用 GPT-5 作为 judge
- 对比模型答案和 ground truth
- 输出 CORRECT / INCORRECT

**为什么需要 LLM-as-a-Judge？**
- 很多 benchmark 的答案格式不统一（数字、文本、列表等）
- 精确匹配无法处理语义等价的情况
- LLM 可以理解答案的语义

> **面试深挖点**：
> - LLM-as-a-Judge 的可靠性如何？有没有误判的情况？
> - 如何降低 LLM-as-a-Judge 的成本？
> - 对于数学题，LLM-as-a-Judge 和精确匹配哪个更好？

### 5.4 信息泄露防护

**措施**：
- 评估时屏蔽特定网站（如 HuggingFace datasets/spaces）
- Canary string testing：检查工具输出是否包含 benchmark 答案
- 发现污染的轨迹直接丢弃

```python
# settings.py 中的 URL 过滤
def _should_block_hf_scraping(self, tool_name, arguments):
    return "huggingface.co/datasets" in url or "huggingface.co/spaces" in url
```

---

## 6. 高频面试问题

### Q1: 介绍一下 MiroThinker 的核心创新？

**参考回答**：

MiroThinker 的核心创新是提出了 **Interactive Scaling** 的概念。之前提升 Agent 性能主要靠两个维度：增大模型参数（Model Scaling）和增加上下文长度（Context Scaling）。MiroThinker 提出了第三个维度：通过后训练（SFT + DPO）让模型学会处理更深、更频繁的 Agent-环境交互。

具体来说，它：
1. 收集强模型在 RLVR 数据集上的正确交互轨迹
2. 用 SFT 让基座模型学习工具调用模式
3. 用 DPO 进一步优化模型的工具选择偏好
4. 评估时支持 256K 上下文和 300 次工具调用

---

### Q2: MiroThinker 的上下文管理是怎么做的？

**参考回答**：

MiroThinker 使用 **Recency-based Context Retention** 策略。核心思想是：Agent 的后续行动主要依赖最近的观察，而非远处的历史。

具体实现：
- `keep_tool_result` 参数控制保留最近 K 个工具响应
- 被丢弃的工具响应替换为 "Tool result is omitted to save tokens."
- 保留完整的 thought-action 序列（只压缩 observation）
- 配合 `context_compress_limit` 参数，当上下文达到限制时触发失败经验压缩和重试

这个策略的好处：
1. 保持推理链的完整性
2. 释放上下文空间给更长的交互
3. 实验表明不会导致性能下降

---

### Q3: 详细讲讲 Failure Experience 机制？

**参考回答**：

当 Agent 在给定的 turns 内未能完成任务时，MiroThinker 会生成结构化的失败经验总结，用于指导下一次尝试。

失败经验包含三个部分：
1. **Failure Type**：incomplete（用完 turns）、blocked（工具失败）、misdirected（走错方向）、format_missed（忘记格式）
2. **What Happened**：描述采取的方法和失败原因
3. **Useful Findings**：列出可复用的事实和结论

重试时，将所有累积的失败经验注入 task_description，让模型避免重复错误。这个机制类似于 Reflexion，但更结构化，且支持多次重试的累积。

---

### Q4: MCP 协议和 OpenAI function calling 有什么区别？为什么选择 MCP？

**参考回答**：

**MCP（Model Context Protocol）** 是一个标准化的工具调用协议，支持 Stdio 和 SSE 两种传输方式。与 OpenAI function calling 的主要区别：

1. **解耦性**：MCP Server 是独立进程，Agent 通过协议调用；function calling 是直接集成在 API 中
2. **可扩展性**：新增工具只需新增 MCP Server，无需修改 Agent 代码
3. **标准化**：MCP 是开放协议，不绑定特定 LLM 提供商
4. **工具管理**：MCP 支持工具黑名单、动态工具发现等高级功能

选择 MCP 的原因：
- 便于集成多种工具（搜索、代码执行、视觉理解等）
- 工具执行和 Agent 推理解耦，便于独立优化
- 支持并发工具调用

---

### Q5: SFT 和 DPO 各自解决什么问题？

**参考回答**：

**SFT（Supervised Fine-Tuning）**：
- 目标：让基座模型学会 Agent 的工具调用模式
- 数据：正确的多轮交互轨迹（system prompt → user → assistant → tool result → ...）
- 解决的问题：从"纯文本模型"变成"会用工具的 Agent"
- 类比：教模型"怎么用工具"

**DPO（Direct Preference Optimization）**：
- 目标：优化模型在多个可能的工具调用策略中选择更好的那个
- 数据：正确轨迹（chosen）vs 错误轨迹（rejected）
- 解决的问题：在"会用工具"的基础上，学会"更好地用工具"
- 类比：教模型"什么时候该用什么工具"

两者的关系：SFT 是基础，DPO 是进一步优化。没有 SFT 的基础，DPO 的效果会大打折扣。

---

### Q6: MiroThinker 的 Rollback 机制是怎么工作的？

**参考回答**：

Rollback 机制用于处理工具调用中的错误，避免错误信息污染后续推理。触发条件包括：

1. **MCP 格式错误**：响应中包含 XML 标签（说明模型没有正确理解工具调用格式）
2. **拒绝关键词**：模型说"我无法回答"等
3. **重复查询**：已经搜索过的 query
4. **工具执行错误**：Unknown tool、Error executing tool
5. **搜索空结果**：Google 搜索返回空结果

回滚逻辑：
- 移除错误的 assistant 响应（`message_history.pop()`）
- 将 turn_count 减 1
- 增加 consecutive_rollbacks 计数
- 如果连续回滚达到上限（默认 5 次），强制结束循环

---

### Q7: 如何保证评估的公平性？如何防止信息泄露？

**参考回答**：

MiroThinker 采取了多重防护措施：

1. **URL 黑名单**：评估时屏蔽 HuggingFace datasets/spaces 等可能包含 benchmark 答案的网站
2. **Canary String Testing**：检查工具输出中是否包含 benchmark 答案的特定字符串
3. **污染轨迹丢弃**：发现污染的轨迹直接丢弃，不计入评估结果
4. **多次运行取最佳**：使用 Pass@K 评估，报告 Best Pass@1 和 Avg@8 两种指标
5. **独立验证**：使用 LLM-as-a-Judge 进行答案验证，而不是简单的字符串匹配

---

### Q8: MiroThinker 的工具调用是怎么解析的？

**参考回答**：

MiroThinker 使用 XML 格式的工具调用，而不是 OpenAI 的 JSON function calling。解析过程：

1. 在 System Prompt 中定义工具调用格式（XML 标签）
2. 模型生成响应时，将工具调用嵌入在 XML 标签中
3. 解析器使用正则表达式提取 `<use_mcp_tool>` 中的 server_name、tool_name、arguments
4. 调用对应的 MCP Server 执行工具

这种设计的好处：
- 不需要模型支持原生 function calling
- 兼容任何能生成文本的 LLM
- 工具调用和自然语言推理可以在同一个响应中

---

### Q9: keep_tool_result 参数怎么选择？有什么 trade-off？

**参考回答**：

`keep_tool_result` 参数控制保留最近多少个工具响应。

**Trade-off**：
- **太小（如 1-2）**：可能丢失重要的历史信息，导致模型无法基于之前的搜索结果进行推理
- **太大（如 -1，保留所有）**：上下文快速膨胀，无法支持长链交互
- **经验值**：MiroThinker 使用 `keep_tool_result: 5`，在实验中取得了最好的平衡

**为什么 5 是一个好的选择？**
- Agent 的后续行动通常依赖最近 3-5 个工具结果
- 保留完整的 thought-action 序列（只压缩 observation），不丢失推理逻辑
- 为更长的交互释放足够的上下文空间

---

### Q10: MiroThinker 和 Deep Research（如 OpenAI Deep Research）有什么区别？

**参考回答**：

**相似之处**：
- 都是深度研究 Agent，支持多轮工具调用
- 都使用搜索、浏览、代码执行等工具
- 都有长时间的推理过程

**不同之处**：
1. **开源 vs 闭源**：MiroThinker 完全开源，包括框架、模型、训练数据
2. **Interactive Scaling**：MiroThinker 明确提出了这个概念，并通过后训练实现
3. **Context Management**：MiroThinker 有专门的上下文管理机制
4. **Failure Experience**：MiroThinker 有结构化的失败经验机制
5. **成本**：MiroThinker-1.7-mini (30B) 参数量小，部署成本低

---

## 7. 项目介绍话术

### 7.1 一分钟版本

> 我在 MiroThinker 项目中负责/了解了深度研究 Agent 的核心架构。MiroThinker 提出了 Interactive Scaling 的概念，通过后训练（SFT + DPO）让模型学会处理更深的 Agent-环境交互。
>
> 技术上，它使用 MCP 协议进行工具调用，支持 256K 上下文和 300 次工具调用。核心创新包括：Recency-based Context Retention（只保留最近 K 个工具响应）、Failure Experience 机制（结构化失败经验压缩和重试）、以及 Rollback 机制（错误回滚和重复查询检测）。
>
> 在评估上，它在 BrowseComp、GAIA、HLE 等 benchmark 上达到开源 SOTA，其中 BrowseComp-ZH 72.3%。

### 7.2 三分钟版本

> （在一分钟版本基础上，补充以下内容）
>
> 训练流程是这样的：首先，我们用强模型（Claude-3.7 或 GPT-5）在 RLVR 数据集上跑 Agent，通过 LLM-as-a-Judge 验证答案正确性，只保留正确的多轮交互轨迹。然后用这些轨迹做 SFT，让基座模型学会工具调用模式。再用正确/错误轨迹对做 DPO，优化模型的工具选择偏好。
>
> 评估时，我们使用 Pass@K 评估，支持多次尝试。为了防止信息泄露，我们屏蔽了 HuggingFace 等可能包含 benchmark 答案的网站，并进行 Canary String Testing。
>
> 框架层面，使用 Hydra 做配置管理，MCP 协议做工具调用，支持单代理和多代理两种架构。代码结构清晰，核心模块包括 Orchestrator（编排器）、ToolExecutor（工具执行器）、AnswerGenerator（答案生成器）等。

### 7.3 技术深挖版本

> （根据面试官的问题，深入某个技术点）
>
> **例如被问到上下文管理**：
>
> MiroThinker 的上下文管理基于一个观察：Agent 的后续行动主要依赖最近的观察，而非远处的历史。所以我们使用 Recency-based Context Retention，只保留最近 K 个工具响应（默认 K=5），其余替换为占位符。
>
> 但这个策略有个问题：如果直接丢弃，可能会丢失关键信息。所以我们保留完整的 thought-action 序列，只压缩 observation。这样既释放了上下文空间，又不丢失推理逻辑。
>
> 配合 context_compress_limit 参数，当上下文达到限制时，会触发失败经验压缩：生成结构化的失败总结，注入下一轮 task_description，让模型避免重复错误。这个机制最多支持 5 次重试。

---

## 8. 延伸阅读

### 8.1 相关论文

| 论文 | 关联点 |
|------|--------|
| ReAct (Yao et al., 2022) | MiroThinker 的基础推理框架 |
| Reflexion (Shinn et al., 2023) | Failure Experience 机制的灵感来源 |
| DPO (Rafailov et al., 2023) | 后训练使用的偏好优化算法 |
| Qwen3 Technical Report | MiroThinker 的基座模型 |
| GAIA Benchmark | 评估使用的 benchmark |
| BrowseComp | 评估使用的 benchmark |

### 8.2 相关技术栈

| 技术 | 用途 |
|------|------|
| Hydra | 配置管理 |
| MCP (Model Context Protocol) | 工具调用协议 |
| E2B | 代码执行沙箱 |
| Jina.ai | Web 抓取 |
| Serper.dev | Google 搜索 API |
| SGLang / vLLM | 模型推理框架 |
| tiktoken | Token 计数 |

### 8.3 面试前的准备清单

- [ ] 理解 Interactive Scaling 的概念和实现
- [ ] 理解 Context Management 的 Recency-based 策略
- [ ] 理解 Failure Experience 机制
- [ ] 理解 MCP 协议和工具调用流程
- [ ] 理解 SFT + DPO 后训练流程
- [ ] 理解 Pass@K 评估和 LLM-as-a-Judge
- [ ] 能用一分钟介绍项目
- [ ] 能用三分钟介绍项目
- [ ] 准备 2-3 个技术深挖的详细回答
- [ ] 准备 1-2 个"我遇到的挑战和解决方案"

---

## 附录：关键代码位置速查

| 功能 | 文件 | 行号 |
|------|------|------|
| 主循环 | `src/core/orchestrator.py` | 812-1136 |
| 上下文管理（keep_tool_result） | `src/llm/base_client.py` | 124-220 |
| 失败经验生成 | `src/core/answer_generator.py` | 170-254 |
| 重试逻辑 | `src/core/answer_generator.py` | 462-591 |
| MCP 工具定义解析 | `src/utils/prompt_utils.py` | 85-164 |
| 工具调用解析 | `src/utils/parsing_utils.py` | - |
| Rollback 逻辑 | `src/core/orchestrator.py` | 180-256 |
| 重复查询检测 | `src/core/tool_executor.py` | 102-160 |
| Trace 收集 | `apps/collect-trace/` | - |
| Benchmark 评估 | `apps/miroflow-agent/benchmarks/common_benchmark.py` | - |
| MCP 服务器配置 | `src/config/settings.py` | 69-380 |
| System Prompt 模板 | `src/utils/prompt_utils.py` | 85-164 |

---

*最后更新：2026-06-27*
*基于 MiroThinker v1.0-v1.7 代码库整理*
