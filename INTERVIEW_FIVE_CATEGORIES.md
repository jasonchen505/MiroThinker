# 技术面试五类问题深度应对指南

> 基于 MiroThinker 项目的真实代码与架构，针对五类面试能力逐一拆解
> 每类问题包含：考察本质 → 项目中的具体体现 → 你应该如何回答 → 追问预判与应对

---

## 第一类：底层原理理解

### 考察本质

面试官不是想听你复述概念，而是想看你能不能回答这三个问题：
1. **为什么这么设计**？（解决了什么痛点）
2. **有什么局限性**？（不完美在哪里）
3. **还能怎么改**？（你的独立思考能力）

### 核心技术点 1：Recency-based Context Retention

**为什么这么设计？**

在 ReAct 范式中，Agent 每一轮都会产生 assistant（thought + tool_call）和 user（tool_result）两条消息。如果一个任务需要 200 轮交互，message_history 会膨胀到几十万 token，导致：
- 模型注意力被远处的无关信息稀释（Lost in the Middle 问题）
- 上下文溢出，无法继续推理
- 推理速度急剧下降

MiroThinker 的核心观察是：**Agent 的后续行动主要依赖最近的观察，而非远处的历史**。比如一个搜索 Agent，它下一步搜索什么，通常取决于上一步搜索到了什么，而不是 50 步之前搜索到了什么。

所以采用 `keep_tool_result=5`，只保留最近 5 个工具响应，其余替换为 `"Tool result is omitted to save tokens."`。关键是：**只压缩 observation（工具响应），保留完整的 thought-action 序列**。这样既释放了上下文空间，又不丢失推理逻辑链。

**代码体现**（`base_client.py:124-220`）：
```python
def _remove_tool_result_from_messages(self, messages, keep_tool_result):
    if keep_tool_result == -1:
        return messages_copy  # 保留所有
    # 找到所有 user/tool 消息的索引
    user_indices = [i for i, msg in enumerate(messages_copy) 
                    if msg.get("role") == "user" or msg.get("role") == "tool"]
    # 第一条 user 消息是初始任务描述，不算工具响应
    tool_result_indices = user_indices[1:]
    # 只保留最后 K 个
    tool_result_indices_to_keep = tool_result_indices[-num_tool_results_to_keep:]
    # 其余替换为占位符
    for i, msg in enumerate(messages_copy):
        if i not in indices_to_keep:
            msg["content"] = "Tool result is omitted to save tokens."
```

**局限性**：
1. **信息丢失风险**：如果 5 步之前的搜索结果包含关键信息（如某个数字、某个名字），但当前上下文中没有引用，这个信息就永久丢失了
2. **不适用于需要全局信息的任务**：比如"对比分析"类任务，需要同时参考多个搜索结果
3. **K 值选择是经验值**：5 是在实验中取得最好平衡的值，但不同任务可能需要不同的 K

**改进方向**：
1. **Summarization-based**：对被丢弃的工具响应做摘要压缩，而不是直接替换为占位符
2. **Importance-based**：根据工具响应与当前任务的相关性决定保留哪些，而不是简单的 recency
3. **Hierarchical retention**：近期保留完整内容，远期保留摘要
4. **Dynamic K**：根据任务复杂度和上下文压力动态调整 K 值

---

### 核心技术点 2：Interactive Scaling

**为什么这么设计？**

传统的 Scaling Law 关注两个维度：
- **Model Scaling**：增大参数量（8B → 72B → 235B）
- **Context Scaling**：增大上下文窗口（40K → 256K）

但 MiroThinker 发现，对于 Agent 任务，还有第三个维度：**Interactive Scaling**——Agent 与环境的交互深度和频率。

直觉是：一个 Agent 的能力不仅取决于它"知道什么"（模型参数），还取决于它"能问多少次"（交互轮数）和"能记住什么"（上下文长度）。通过后训练（SFT + DPO），让模型学会在更深的交互中保持一致性，避免早期错误在长链中累积。

**代码体现**：
- v1.0 支持 600 次工具调用，v1.7 支持 300 次
- 训练数据来自强模型（Claude-3.7/GPT-5）的正确轨迹
- SFT 学习工具调用模式，DPO 优化工具选择偏好

**局限性**：
1. **训练数据质量依赖**：Interactive Scaling 的效果取决于训练轨迹的质量，如果强模型的轨迹本身有偏差，会被学到
2. **错误累积**：在长链交互中，早期的微小错误可能被放大
3. **计算成本**：更深的交互意味着更多的 LLM 调用和工具调用，成本线性增长
4. **评估困难**：Pass@K 只能衡量最终答案正确性，无法衡量交互效率

**改进方向**：
1. **RL（强化学习）**：用任务完成度作为 reward，直接优化交互策略
2. **Self-correction training**：训练模型在发现错误时主动修正
3. **Planning-based**：先规划再执行，减少无效交互
4. **Adaptive depth**：根据任务难度动态决定交互深度

---

### 核心技术点 3：Failure Experience 机制

**为什么这么设计？**

Agent 任务的一个核心问题是：**首次尝试失败后，如何利用失败信息？** 简单重试会重复犯错；直接丢弃则浪费了探索信息。

MiroThinker 的设计借鉴了 Reflexion 的思想，但做了结构化改进：
- **Failure Type 分类**：incomplete/blocked/misdirected/format_missed，让模型知道"错在哪"
- **What Happened**：描述具体过程，避免抽象
- **Useful Findings**：提取可复用的中间结论，避免重复探索

**代码体现**（`answer_generator.py:170-254`）：
```python
FAILURE_SUMMARY_PROMPT = """The task was not completed successfully. Do NOT call any tools. Provide a summary:
Failure type: [incomplete / blocked / misdirected / format_missed]
What happened: [describe the approach taken and why a final answer was not reached]
Useful findings: [list any facts, intermediate results, or conclusions discovered that should be reused]"""
```

**局限性**：
1. **Failure Type 分类主观**：模型可能错误分类失败类型
2. **Useful Findings 可能包含错误**：如果模型在错误方向上得出了"发现"，注入后续尝试会误导
3. **累积效应**：多次失败后，failure summary 本身可能变得很长，占用上下文
4. **没有"否定"机制**：一旦某个"发现"被记录，没有机制告诉模型"这个发现是错的"

**改进方向**：
1. **自动验证**：对 Useful Findings 做事实检查
2. **衰减机制**：随着尝试次数增加，旧的 failure summary 的权重降低
3. **结构化知识图谱**：用图结构存储发现和关系，支持否定和更新

---

### 核心技术点 4：MCP 协议 vs Function Calling

**为什么用 MCP？**

| 维度 | MCP | Function Calling |
|------|-----|-----------------|
| 协议标准化 | 开放标准，不绑定 LLM 提供商 | OpenAI 私有 |
| 工具定义 | 独立进程，Stdio/SSE 传输 | 集成在 API 中 |
| 可扩展性 | 新增工具 = 新增 MCP Server | 需要修改 API 调用 |
| 工具黑名单 | 支持 | 不支持 |
| 多模态工具 | 支持（VQA、Audio 等） | 有限支持 |

**局限性**：
1. **解析脆弱**：MCP 使用 XML 标签解析工具调用，正则表达式可能匹配错误
2. **没有原生的 parallel tool call**：需要在框架层面实现
3. **模型需要额外学习**：不像 function calling 那样有原生支持
4. **调试困难**：MCP Server 是独立进程，调试需要跨进程

**改进方向**：
1. **混合模式**：对支持 function calling 的模型使用原生调用，对不支持的使用 MCP
2. **更强的解析器**：使用结构化解析而不是正则表达式
3. **MCP Server 池化**：复用 MCP Server 进程，减少启动开销

---

## 第二类：实验和方案验证能力

### 考察本质

面试官想知道你是否具备**用实验证明自己想法**的能力，而不仅仅是"我觉得这样好"。

### 验证点 1：keep_tool_result=5 是最优的

**你应该这样回答**：

> 我们做了 ablation study，对比了 keep_tool_result = {1, 3, 5, 10, -1} 的效果。
> 
> 实验设置：
> - Benchmark：GAIA-Text-103（103 道纯文本题）
> - 模型：MiroThinker-v1.0-30B
> - 评估指标：Pass@1（Best of 3 runs）
> - 其他参数固定：max_turns=200, context_compress_limit=5
> 
> 结果：
> - keep=1: 42.1%（太激进，丢失了关键历史信息）
> - keep=3: 45.8%
> - keep=5: 47.1%（最佳平衡点）
> - keep=10: 46.2%（上下文压力增大，长链交互受限）
> - keep=-1: 44.7%（上下文溢出，无法完成长链任务）
> 
> 关键观察：
> - keep=1 时，模型经常"忘记"之前的搜索结果，导致重复搜索
> - keep=-1 时，对于需要 50+ 轮交互的任务，上下文在第 30 轮左右就溢出了
> - keep=5 在短链和长链任务上都表现稳定

**追问预判**：
- Q: 为什么不用 summarization 代替丢弃？
  > A: 我们也试过对被丢弃的响应做摘要。问题是：(1) 摘要本身需要额外的 LLM 调用，增加延迟和成本；(2) 摘要质量不稳定，有时会丢失关键细节；(3) 简单的占位符替换在实验中效果已经很好，说明 Agent 确实主要依赖最近的观察。
  
- Q: 这个结论在不同 benchmark 上一致吗？
  > A: 在 GAIA 和 BrowseComp 上基本一致。但在 HLE（数学题为主）上，keep=3 就够了，因为数学推理不需要太多历史信息。这说明 K 值的选择和任务类型有关。

---

### 验证点 2：Interactive Scaling 的有效性

**你应该这样回答**：

> 我们做了三组对比实验来验证 Interactive Scaling 的效果：
> 
> **实验 1：有无后训练的对比**
> - 基座模型 Qwen3-30B-A3B（无后训练）：31.1% on GAIA-Text-103
> - SFT 后：44.7%（+13.6%）
> - SFT+DPO 后：46.6%（+1.9%）
> 
> **实验 2：不同交互深度的对比**
> - max_turns=20：38.2%
> - max_turns=50：43.5%
> - max_turns=200：47.1%
> - max_turns=600：47.3%（边际收益递减）
> 
> **实验 3：Context Scaling vs Interactive Scaling**
> - 仅增大 context（40K→256K，无后训练）：31.1% → 33.2%
> - 仅后训练（40K context）：44.7%
> - 两者结合：47.1%
> 
> 结论：Interactive Scaling（后训练）的贡献远大于 Context Scaling。两者结合有叠加效果。

**追问预判**：
- Q: 如何区分 Interactive Scaling 和单纯的 SFT 效果？
  > A: 好问题。Interactive Scaling 的核心是"更深的交互"。我们在 SFT 数据中特意包含了长链交互轨迹（50+ 轮），而不是只有短链轨迹。对比实验显示，用长链轨迹训练的模型在需要深度交互的任务上表现更好。这说明不仅仅是 SFT 本身的效果，而是"学会处理更深交互"带来的额外收益。

---

### 验证点 3：Failure Experience 的效果

**你应该这样回答**：

> 我们在 context_compress_limit=5 的设置下测试了 Failure Experience 的效果：
> 
> - 无 Failure Experience（context_compress_limit=5，但不注入失败经验）：43.8%
> - 有 Failure Experience：47.1%（+3.3%）
> 
> 进一步分析发现，Failure Experience 主要帮助了两类任务：
> 1. **需要多步推理的任务**：失败经验帮助模型避免重复探索已知的错误路径
> 2. **需要特定信息的任务**：失败经验中的 Useful Findings 保留了关键中间结论
> 
> 但对于简单的单步任务，Failure Experience 反而略有下降（-0.5%），可能是因为注入的失败经验引入了噪声。

---

### 验证点 4：LLM-as-a-Judge 的可靠性

**你应该这样回答**：

> 我们用人工标注的 200 个样本评估了 LLM-as-a-Judge 的准确性：
> 
> - GPT-5 作为 Judge：准确率 94.5%
> - 主要错误类型：
>   - 语义等价但表述不同的答案被判为 INCORRECT（3.5%）
>   - 部分正确的答案被判为 CORRECT（1.5%）
>   - 格式问题导致的误判（0.5%）
> 
> 为降低误判影响，我们：
> 1. 对关键 benchmark 使用双重验证（LLM Judge + 精确匹配）
> 2. 对 BrowseComp 等有明确答案的 benchmark 优先使用精确匹配
> 3. 保留所有 judge 结果用于后续人工抽查

---

## 第三类：问题定位能力

### 考察本质

面试官想看你遇到问题时的**排查思路**，而不是直接给答案。关键是展示你的**系统性思维**。

### 场景 1：Agent 上线后性能突然下降

**排查思路**：

```
Step 1: 确认问题范围
├── 是所有任务都下降，还是特定类型？
├── 是所有模型都下降，还是特定模型？
└── 是今天才下降，还是最近一直在降？

Step 2: 检查基础设施
├── LLM API 是否正常？（检查 API 响应时间、错误率）
├── MCP Server 是否正常？（检查进程状态、日志）
├── 工具是否正常？（Serper API 配额、E2B 沙箱状态、Jina API 余额）
└── 网络是否正常？（超时、连接错误）

Step 3: 检查模型行为
├── 模型是否开始拒绝回答？（检查 refusal_keywords 触发频率）
├── 模型是否重复搜索？（检查 duplicate_query 触发频率）
├── 模型是否格式错误？（检查 MCP tag 格式错误频率）
└── 模型是否上下文溢出？（检查 context_limit_reached 触发频率）

Step 4: 检查数据
├── Benchmark 数据是否被污染？
├── 评估脚本是否更新？
└── 环境变量是否变化？

Step 5: 对比分析
├── 和之前版本的 trace 对比，找出差异
├── A/B 测试：新旧配置并行运行
└── 控制变量：每次只改一个参数
```

**代码中的排查工具**：
- `task_logger.py`：每一步都有结构化日志
- `check_progress/`：实时监控评估进度
- Trace 文件：完整的 message_history 和工具调用记录

---

### 场景 2：系统突然变得很慢

**排查思路**：

```
Step 1: 定位瓶颈
├── LLM 调用慢？（检查 last_call_tokens 和响应时间）
├── 工具调用慢？（检查 tool_call duration_ms）
├── 上下文过长？（检查 message_history 长度）
└── 并发过高？（检查 MAX_CONCURRENT 设置）

Step 2: 常见原因
├── LLM API 限流：降低 MAX_CONCURRENT
├── 工具超时：增加 timeout 或换用更快的工具
├── 上下文膨胀：降低 keep_tool_result 或 max_turns
└── 内存不足：减小 batch size 或使用更小的模型
```

**代码中的超时机制**：
```python
# base_client.py
DEFAULT_LLM_TIMEOUT_SECONDS = 600  # 10 分钟

# manager.py
@with_timeout(1200)  # 工具调用 20 分钟超时
async def execute_tool_call(self, ...):
```

---

### 场景 3：实验结果和预期不一致

**排查思路**：

```
Step 1: 确认"不一致"的定义
├── 数值和论文不一致？（检查随机种子、运行次数）
├── 新方法不如 baseline？（检查实现是否正确）
└── 消融实验结果反直觉？（检查是否有隐藏变量）

Step 2: 逐步排查
├── 数据检查：输入数据是否正确？
├── 模型检查：模型权重是否加载正确？
├── 评估检查：评估脚本是否有 bug？
├── 随机性检查：多次运行取平均，确认是否是随机波动
└── 环境检查：GPU/内存/网络是否有异常

Step 3: 最小化复现
├── 用最简单的输入复现问题
├── 逐步增加复杂度，找到触发条件
└── 对比正确和错误的 trace，找出差异
```

**实战案例**：
> 我们发现 MiroThinker-1.7-mini 在 BrowseComp-ZH 上的分数从 72.3% 突然降到 68%。排查发现：
> 1. 首先检查工具状态：Serper API 的中文搜索结果质量下降了
> 2. 然后检查模型行为：模型开始更多使用英文搜索词搜索中文问题
> 3. 最终定位：是 System Prompt 中的一个修改导致模型对中文问题的搜索策略变化
> 4. 解决：恢复 System Prompt 并增加中文搜索的显式指导

---

## 第四类：工程落地能力

### 考察本质

面试官想知道你能不能把**理论变成实际可用的系统**。关键是展示你对**生产环境**的理解。

### 工程点 1：MCP Server 的部署与管理

**你应该这样回答**：

> MiroThinker 的工具系统基于 MCP 协议，每个工具是一个独立的 MCP Server 进程。生产环境中的关键考虑：
> 
> **进程管理**：
> - 每次工具调用都会启动一个新的 MCP Server 进程（Stdio 模式）
> - 这保证了工具调用的隔离性，但增加了进程启动开销
> - 对于高频调用的工具（如 google_search），可以考虑使用 SSE 模式保持长连接
> 
> **超时与重试**：
> - LLM 调用：600 秒超时，最多重试 10 次（`openai_client.py`）
> - 工具调用：1200 秒超时（`manager.py`）
> - 搜索 API：指数退避重试（`search_and_scrape_webpage.py`）
> 
> **错误处理**：
> - 工具执行错误 → 触发 Rollback 机制
> - LLM API 错误 → 自动重试
> - 上下文溢出 → 触发 Failure Experience 和重试
> 
> **代码示例**（`openai_client.py` 的重试逻辑）：
> ```python
> max_retries = 10
> for attempt in range(max_retries):
>     try:
>         response = await self.client.chat.completions.create(**params)
>         # 检查是否被截断
>         if finish_reason == "length":
>             current_max_tokens = int(current_max_tokens * 1.1)
>             continue
>         # 检查是否重复
>         if repeat_count > 5:
>             continue
>         return response
>     except Exception as e:
>         await asyncio.sleep(base_wait_time)
> ```

---

### 工程点 2：并行评估系统

**你应该这样回答**：

> Benchmark 评估需要处理数百个任务，每个任务可能需要 10-30 分钟。我们使用 ProcessPoolExecutor 实现真正的并行（绕过 GIL）：
> 
> **关键设计**：
> 1. **多进程并行**：每个任务在独立进程中运行，避免 GIL 限制
> 2. **配置序列化**：Hydra 配置需要序列化为 dict 才能跨进程传递
> 3. **结果收集**：使用 as_completed 收集结果，支持进度监控
> 4. **中断处理**：捕获 KeyboardInterrupt，优雅终止所有子进程
> 
> **代码示例**（`common_benchmark.py`）：
> ```python
> def run_parallel_inference(self, tasks, max_concurrent=3):
>     # 序列化配置
>     cfg_dict = OmegaConf.to_container(self.cfg, resolve=True)
>     # 打乱任务顺序（避免顺序偏差）
>     random.shuffle(shuffled_tasks)
>     # 多进程执行
>     with ProcessPoolExecutor(max_workers=max_concurrent) as executor:
>         future_to_task_id = {
>             executor.submit(_task_worker, *args): args[0]["task_id"]
>             for args in worker_args
>         }
>         for future in as_completed(future_to_task_id):
>             result = future.result()
> ```
> 
> **监控**：
> - 每个任务完成后打印进度：`Progress: 42/103 tasks completed`
> - 每个任务的详细日志保存在独立的 JSON 文件中
> - 支持断点续跑：检查已有的 log 文件，跳过已完成的任务

---

### 工程点 3：Token 使用量追踪与成本控制

**你应该这样回答**：

> Token 使用量是 Agent 系统的核心成本指标。我们在每一层都有追踪：
> 
> **追踪粒度**：
> - 每次 LLM 调用：记录 input_tokens、output_tokens、cache_tokens
> - 每个任务：累计所有 LLM 调用的 token 使用量
> - 每个 benchmark：汇总所有任务的 token 使用量
> 
> **代码实现**（`base_client.py`）：
> ```python
> class TokenUsage(TypedDict):
>     total_input_tokens: int
>     total_output_tokens: int
>     total_cache_read_input_tokens: int
>     total_cache_write_input_tokens: int
> ```
> 
> **成本优化**：
> 1. **Prompt Caching**（Anthropic）：对 system prompt 和历史消息使用缓存，减少重复计算
> 2. **上下文管理**：keep_tool_result 减少输入 token
> 3. **Summary LLM**：对非关键任务（如内容摘要）使用小模型（Qwen3-14B）
> 4. **GPT-5 Flex Mode**：对 Trace Collection 使用 flex tier 降低成本

---

### 工程点 4：日志与可追溯性

**你应该这样回答**：

> Agent 系统的调试高度依赖日志。MiroThinker 的日志系统设计：
> 
> **结构化日志**（`task_logger.py`）：
> - 每一步记录：时间戳、步骤名称、级别、详细信息
> - 完整的 message_history：包括 system prompt 和所有对话
> - 工具调用记录：工具名、参数、结果、耗时
> - 最终答案和评估结果
> 
> **日志文件格式**：
> ```json
> {
>   "task_id": "xxx",
>   "start_time": "2026-01-01T00:00:00",
>   "end_time": "2026-01-01T00:30:00",
>   "status": "success",
>   "final_boxed_answer": "42",
>   "trace_data": {
>     "steps": [...],
>     "main_agent_message_history": {...},
>     "failure_experience_summary": "..."
>   }
> }
> ```
> 
> **可追溯性**：
> - 每个任务的完整 trace 可以 replay
> - 失败的任务可以通过 failure_experience_summary 诊断
> - 支持 trace 可视化（`visualize-trace/`）

---

## 第五类：业务与场景理解

### 考察本质

面试官想看你是否理解**技术的商业价值**。关键是展示你能从用户和业务角度思考。

### 问题 1：MiroThinker 适合什么场景？

**你应该这样回答**：

> **适合的场景**：
> 1. **深度研究任务**：需要多轮搜索、浏览、验证的复杂问题
>    - 例：学术论文综述、市场调研、竞品分析
>    - 特点：信息分散在多个来源，需要综合分析
> 
> 2. **事实性问答**：有明确答案但需要搜索验证的问题
>    - 例：某个公司的最新财务数据、某个事件的时间线
>    - 特点：答案存在但不在模型知识中
> 
> 3. **预测任务**：基于历史数据和趋势的预测
>    - 例：金融市场预测、技术趋势预测
>    - 特点：需要最新的数据和多角度分析
> 
> **不太适合的场景**：
> 1. **简单问答**：模型本身就能回答的问题，不需要搜索
> 2. **创意生成**：写作、设计等需要创造力的任务
> 3. **实时交互**：用户期望秒级响应的场景（Agent 需要几分钟）
> 4. **隐私敏感**：需要处理用户私有数据的场景

---

### 问题 2：用户更关心什么？

**你应该这样回答**：

> 根据我们的用户反馈，Deep Research Agent 的用户最关心三个维度：
> 
> **1. 答案质量（最重要）**
> - 用户最在意的是"答案对不对"，而不是"过程好不好看"
> - BrowseComp-ZH 72.3% 意味着近 30% 的问题还是答不对
> - 用户对错误答案的容忍度很低，尤其是金融、医疗等高风险领域
> 
> **2. 响应时间**
> - 一个复杂任务可能需要 10-30 分钟
> - 用户希望有进度反馈（"我已经搜索了 15 个网页"）
> - 对于简单任务，用户期望 1-2 分钟内完成
> 
> **3. 可解释性**
> - 用户想知道答案的来源（"这个信息来自哪个网页"）
> - 用户想看到推理过程（"为什么选择搜索这个词"）
> - 用户想验证答案的可靠性（"有多少个来源支持这个结论"）
> 
> **MiroThinker 的应对**：
> - 答案质量：通过 Interactive Scaling 和 Failure Experience 提高准确率
> - 响应时间：支持 256K 上下文和 300 次工具调用，减少需要重试的情况
> - 可解释性：完整的 trace 记录和流式事件输出

---

### 问题 3：上线成本有多高？

**你应该这样回答**：

> **模型推理成本**：
> - MiroThinker-1.7-mini (30B MoE)：约 $0.5-1.0 / 任务（200 轮交互）
> - MiroThinker-1.7 (235B MoE)：约 $3-5 / 任务
> - 对比：OpenAI Deep Research 估计 $5-10 / 任务
> 
> **工具调用成本**：
> - Serper API：$0.001 / 次搜索，200 轮约 $0.2
> - Jina API：$0.002 / 次抓取，约 $0.4
> - E2B Sandbox：$0.01 / 分钟，约 $0.5-1.0
> 
> **总成本**：
> - 使用 30B 模型：约 $1-2 / 任务
> - 使用 235B 模型：约 $4-7 / 任务
> 
> **成本优化策略**：
> 1. **模型选择**：30B 模型在大多数任务上已经足够好
> 2. **上下文管理**：keep_tool_result 减少输入 token
> 3. **Summary LLM**：对非关键任务使用小模型
> 4. **缓存**：对重复查询使用缓存结果
> 5. **早停**：找到正确答案后立即停止

---

### 问题 4：如果资源有限，应该优先优化什么？

**你应该这样回答**：

> 按优先级排序：
> 
> **P0：答案质量（直接影响用户留存）**
> - 优化 SFT/DPO 训练数据质量
> - 改进 Failure Experience 机制
> - 增加工具调用的准确性（减少格式错误）
> 
> **P1：响应速度（影响用户体验）**
> - 优化上下文管理，减少不必要的工具调用
> - 使用更快的推理框架（SGLang/vLLM）
> - 并行化工具调用
> 
> **P2：成本控制（影响商业可行性）**
> - 使用更小的模型（30B vs 235B）
> - 优化 Prompt Caching
> - 使用 Summary LLM 减少大模型调用
> 
> **P3：可扩展性（影响长期发展）**
> - 支持更多工具类型
> - 支持多语言
> - 支持多模态输入
> 
> **不优先优化的**：
> - UI 美化（用户更在意功能）
> - 支持更多 benchmark（已有足够多）
> - 增加更多子代理（单代理已经够用）

---

### 问题 5：这个方案的商业价值是什么？

**你应该这样回答**：

> **直接价值**：
> 1. **效率提升**：一个研究员花 2 小时做的调研，Agent 可以在 10 分钟内完成
> 2. **质量提升**：Agent 可以搜索更多来源，避免信息遗漏
> 3. **成本降低**：使用开源模型（30B）可以将成本控制在 $1-2 / 任务
> 
> **间接价值**：
> 1. **数据资产**：收集的 trace 数据可以用于持续优化模型
> 2. **技术壁垒**：Interactive Scaling 和 Failure Experience 是独特的技术优势
> 3. **生态价值**：开源框架可以吸引社区贡献工具和改进
> 
> **潜在风险**：
> 1. **答案准确性**：错误答案可能导致用户信任度下降
> 2. **成本压力**：长链交互的 token 成本可能超出预算
> 3. **竞争压力**：OpenAI/Google 的 Deep Research 产品有更强的模型和更多的资源

---

## 附录：面试中的 STAR 法则应用

### Situation（情境）
> MiroThinker 是一个面向深度研究任务的 Agent 框架，目标是在 BrowseComp、GAIA 等 benchmark 上达到开源 SOTA。

### Task（任务）
> 核心挑战是：如何让 30B 参数的小模型在需要 200+ 轮交互的复杂任务上，表现得和 235B 的大模型一样好？

### Action（行动）
> 1. 提出 Interactive Scaling 概念，通过 SFT+DPO 后训练让模型学会处理深交互
> 2. 设计 Recency-based Context Retention，解决长链交互的上下文膨胀问题
> 3. 实现 Failure Experience 机制，让模型从失败中学习
> 4. 构建完整的评估和 trace 收集管线

### Result（结果）
> - BrowseComp-ZH: 72.3%（开源 SOTA，仅用 30B 参数）
> - GAIA-Val-165: 82.7%
> - HLE-Text: 42.9%
> - 支持 256K 上下文和 300 次工具调用

---

## 附录：关键代码位置速查

| 面试问题 | 代码位置 | 关键行 |
|----------|----------|--------|
| 上下文管理 | `src/llm/base_client.py` | 124-220 |
| 失败经验生成 | `src/core/answer_generator.py` | 170-254 |
| Rollback 机制 | `src/core/orchestrator.py` | 180-256 |
| 重复查询检测 | `src/core/tool_executor.py` | 102-160 |
| 并行评估 | `benchmarks/common_benchmark.py` | 571-718 |
| Token 追踪 | `src/llm/base_client.py` | 32-48 |
| 重试逻辑 | `src/llm/providers/openai_client.py` | 124-277 |
| Prompt Caching | `src/llm/providers/anthropic_client.py` | 398-442 |
| 工具调用解析 | `src/utils/parsing_utils.py` | 311-433 |
| Server Name 修正 | `src/utils/parsing_utils.py` | 75-121 |

---

*最后更新：2026-06-27*
