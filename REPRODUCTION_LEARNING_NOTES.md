# MiroThinker 复现学习笔记

> 基于 8×4090 全流程复现过程中的学习记录
> 持续更新中

---

## 学习模块 1：模型部署与推理

### 1.1 MoE 模型的显存计算

**学到的点**：

MoE（Mixture of Experts）模型的显存计算和 Dense 模型不同：

```
Dense 模型：
  显存 = 参数量 × 每参数字节数
  例：8B × 2 bytes (BF16) = 16GB

MoE 模型：
  权重显存 = 总参数量 × 每参数字节数（所有 expert 都要加载）
  推理显存 = 激活参数 × 每参数字节数 + KV Cache
  例：30B MoE → 权重 60GB，但推理时只激活 3B → 单序列只需 ~6GB 计算
```

**为什么 MoE 省显存但不省权重存储？**
- MoE 的"稀激活性"体现在计算量上，不是存储上
- 所有 expert 的权重都要加载到 GPU，但每次只用其中一小部分
- 30B MoE 的权重大小和 30B Dense 一样，但推理速度接近 3B Dense

**4090 上的实际体验**：
- 30B MoE (FP8)：权重 ~30GB，单卡放不下，需要 TP=2
- 推理时每卡实际使用的计算单元很少，但 KV Cache 会占大量显存
- 256K context 的 KV Cache 在 30B MoE 上约占 8-16GB per card

---

### 1.2 Tensor Parallelism 在 PCIe 上的效率

**学到的点**：

Tensor Parallelism (TP) 需要跨卡通信，NVLink 和 PCIe 的差距很大：

```
NVLink (A100): 600 GB/s bidirectional
PCIe 4.0 x16 (4090): 32 GB/s per direction

差距：~18x
```

**实际影响**：
- TP=2 在 PCIe 上：推理速度约为 NVLink 的 60-70%
- TP=4 在 PCIe 上：推理速度约为 NVLink 的 40-50%
- TP=8 在 PCIe 上：推理速度约为 NVLink 的 20-30%（不推荐）

**最佳实践**：
- 4090 上尽量用 TP=2，最多 TP=4
- 如果模型能放单卡，优先 DP（Data Parallelism）而不是 TP
- 对于 30B MoE FP8 (~30GB)，TP=2 是最佳选择

---

### 1.3 FP8 量化的影响

**学到的点**：

FP8 是 Ada Lovelace 架构（4090）原生支持的量化格式：

```
BF16: 1 sign + 8 exponent + 7 mantissa = 16 bits
FP8 (E4M3): 1 sign + 4 exponent + 3 mantissa = 8 bits
FP8 (E5M2): 1 sign + 5 exponent + 2 mantissa = 8 bits
```

**精度影响**：
- FP8 量化对模型质量的影响很小（<1% 准确率下降）
- 但显存节省 50%（60GB → 30GB for 30B MoE）
- 推理速度提升 ~30%（减少内存带宽压力）

**SGLang 的 FP8 实现**：
```bash
--quantization fp8  # 启用 FP8 量化
--mem-fraction-static 0.85  # 85% 显存用于 KV Cache
```

---

### 1.4 KV Cache 与 Context Length 的关系

**学到的点**：

KV Cache 是推理时的主要显存消耗：

```
KV Cache 显存 = 2 × num_layers × num_heads × head_dim × seq_len × batch_size × bytes_per_element

对于 30B MoE (假设 64 layers, 32 heads, 128 head_dim):
- 1 sequence, 128K context, BF16: 2 × 64 × 32 × 128 × 131072 × 2 = ~13GB
- 1 sequence, 256K context, BF16: ~26GB
```

**实际优化**：
- 降低 context length 是最直接的省显存方法
- MiroThinker 的 keep_tool_result 机制本质上就是在减少 KV Cache
- 128K context 对大多数任务已经足够

---

## 学习模块 2：Agent 框架

### 2.1 ReAct 范式的实现细节

**学到的点**：

MiroThinker 的 ReAct 循环比教科书上的更复杂：

```
教科书 ReAct:
  while not done:
    thought = LLM(history)
    action = LLM(history + thought)
    observation = execute(action)
    history += thought + action + observation

MiroThinker ReAct:
  while turn < max_turns:
    response = LLM(system_prompt + history)  # 包含工具定义
    if response has tool_call:
      if is_duplicate(tool_call):  # 重复查询检测
        rollback()
        continue
      if is_format_error(response):  # 格式错误检测
        rollback()
        continue
      result = execute(tool_call)
      if is_error(result):  # 工具执行错误
        rollback()
        continue
      history += response + result
    else:
      break  # 没有工具调用，结束
    if context_too_long():  # 上下文长度检查
      compress_context()
```

**关键差异**：
1. 有重复查询检测（避免浪费）
2. 有格式错误检测（MCP XML 解析）
3. 有 rollback 机制（错误恢复）
4. 有上下文压缩（支持长链交互）

---

### 2.2 System Prompt 的设计

**学到的点**：

MiroThinker 的 System Prompt 包含三个关键部分：

```
1. 工具定义（MCP 格式）
   - 每个工具的 name、description、schema
   - 服务器名称和工具名称的映射

2. 格式指令
   - XML 格式的工具调用模板
   - 强调工具调用必须放在响应末尾

3. Agent 目标
   - 主代理：使用工具逐步解决问题
   - 浏览代理：搜索和浏览网页获取信息
```

**设计要点**：
- 工具定义放在 system prompt 而不是 user message，保证每轮都能看到
- XML 格式比 JSON 更容易被 LLM 生成（不需要转义）
- 不同 agent 类型有不同的目标指令

---

### 2.3 Context Management 的工程实现

**学到的点**：

Recency-based Context Retention 的实现比想象中更细致：

```python
# 关键：只替换 user/tool 消息（工具响应），保留 assistant 消息（推理链）
def _remove_tool_result_from_messages(self, messages, keep_tool_result):
    # 第一条 user 消息是初始任务描述，不算工具响应
    tool_result_indices = user_indices[1:]
    
    # 保留最近 K 个工具响应
    indices_to_keep = [first_user_idx] + tool_result_indices_to_keep
    
    # 替换其余的为占位符
    for i, msg in enumerate(messages_copy):
        if i not in indices_to_keep:
            msg["content"] = "Tool result is omitted to save tokens."
```

**为什么保留 assistant 消息？**
- assistant 消息包含 thought 和 tool_call，是推理逻辑链
- 如果删除 assistant 消息，模型不知道"为什么"要调用那个工具
- 只删除 tool result（observation），保留 thought-action 序列

---

### 2.4 Failure Experience 的结构化设计

**学到的点**：

Failure Experience 不是简单的"重试"，而是结构化的学习：

```
Failure Type 分类：
├── incomplete: turns 用完了还没完成
│   → 重试策略：减少 turns 浪费，更高效地搜索
├── blocked: 工具失败或信息缺失
│   → 重试策略：换用其他工具或搜索词
├── misdirected: 走错了方向
│   → 重试策略：完全换一个思路
└── format_missed: 找到答案但忘记用 \boxed{}
    → 重试策略：注意输出格式
```

**为什么分类很重要？**
- 不同类型的失败需要不同的重试策略
- 如果不分类，模型可能重复同样的错误
- 结构化的 failure summary 比自由文本更容易被模型理解

---

## 学习模块 3：后训练

### 3.1 SFT 数据的质量比数量更重要

**学到的点**：

从 Trace Collection 的设计可以看出：

```
数据收集流程：
1. 用强模型（Claude-3.7/GPT-5）跑 Agent
2. 用 LLM-as-a-Judge 验证答案正确性
3. 只保留正确的轨迹

为什么只保留正确的？
- SFT 是"模仿学习"，模型会模仿训练数据中的行为
- 如果训练数据中有错误的工具调用，模型也会学到错误的模式
- 质量 > 数量：100 条高质量数据 > 1000 条低质量数据
```

**实际经验**：
- SFT 数据的最佳数量：500-2000 条
- 太少（<100）：模型学不到足够的工具调用模式
- 太多（>5000）：可能包含低质量数据，反而降低性能

---

### 3.2 SFT 和 DPO 的区别

**学到的点**：

```
SFT（监督微调）：
- 数据：正确的轨迹
- 目标：模仿正确行为
- 类比：教模型"怎么用工具"
- 损失函数：Cross-Entropy Loss on target tokens

DPO（直接偏好优化）：
- 数据：正确轨迹 (chosen) vs 错误轨迹 (rejected)
- 目标：学会区分好坏
- 类比：教模型"什么时候该用什么工具"
- 损失函数：log(σ(β(log π_θ(chosen)/π_ref(chosen) - log π_θ(rejected)/π_ref(rejected))))
```

**关键区别**：
- SFT 只告诉模型"正确答案是什么"
- DPO 告诉模型"这个好，那个不好"
- DPO 需要 SFT 作为基础（先学会用工具，再学会更好地用工具）

---

### 3.3 LoRA 训练的工程细节

**学到的点**：

LoRA（Low-Rank Adaptation）是参数高效微调的核心：

```
原始权重：W ∈ R^{d×d}
LoRA 分解：W' = W + α × A × B
其中：A ∈ R^{d×r}, B ∈ R^{r×d}, r << d

参数量对比：
- 全参训练：d×d = d² 个参数
- LoRA (r=64)：2 × d × r = 128d 个参数
- 比例：128d / d² = 128/d ≈ 0.1% (for d=1024)
```

**为什么 MoE 模型用 LoRA 更合适？**
- MoE 模型参数量大（30B），全参训练显存需求高
- LoRA 只训练 ~0.1% 的参数，显存需求大幅降低
- 实验表明 LoRA 对 MoE 模型的效果接近全参训练

---

### 3.4 DPO 的温度参数 β

**学到的点**：

DPO 损失函数中的 β 参数控制偏好强度：

```
L_DPO = -log(σ(β × [log π_θ(chosen)/π_ref(chosen) - log π_θ(rejected)/π_ref(rejected)]))

β 的影响：
- β 大（如 0.5）：模型更激进地追求 chosen，可能过拟合
- β 小（如 0.05）：模型更保守，学习速度慢
- β = 0.1：通常是最好的平衡点
```

---

## 学习模块 4：评估体系

### 4.1 Pass@K 评估的意义

**学到的点**：

Pass@K 评估的是"至少有一次正确"的概率：

```
Pass@1 = 正确次数 / 总次数
Pass@K = 1 - (1-P)^K  （如果每次独立）

实际含义：
- Pass@1 = 50%：每次尝试有 50% 概率正确
- Pass@3 = 87.5%：3 次尝试中至少有一次正确的概率
- Pass@8 = 99.6%：8 次尝试几乎保证正确
```

**为什么 MiroThinker 同时报告 Best Pass@1 和 Avg@8？**
- Best Pass@1：3 次运行中最好的那次，反映模型上限
- Avg@8：8 次运行的平均，反映模型稳定性
- 两个指标一起看才能全面评估模型

---

### 4.2 LLM-as-a-Judge 的可靠性

**学到的点**：

LLM-as-a-Judge 的主要问题是**语义等价判断**：

```
Ground truth: "42"
Model answer: "The answer is 42"
→ LLM Judge: CORRECT（正确识别语义等价）

Ground truth: "42"
Model answer: "42.0"
→ LLM Judge: CORRECT（正确识别数值等价）

Ground truth: "New York City"
Model answer: "NYC"
→ LLM Judge: CORRECT（正确识别缩写等价）
```

**但也有误判**：
```
Ground truth: "42"
Model answer: "42 or 43"
→ LLM Judge: 可能判 CORRECT（误判）

Ground truth: "42.0"
Model answer: "42"
→ LLM Judge: 可能判 INCORRECT（误判）
```

**最佳实践**：
- 对有明确答案的题目（数学、选择题），优先用精确匹配
- 对开放式题目（文本回答），用 LLM Judge
- 对关键结果，人工抽查验证

---

### 4.3 信息泄露防护

**学到的点**：

Benchmark 评估中的信息泄露是一个严重问题：

```
泄露来源：
1. 训练数据中包含 benchmark 答案
2. 工具（如搜索引擎）返回 benchmark 答案
3. 网页中包含 benchmark 讨论

MiroThinker 的防护：
1. URL 黑名单：屏蔽 HuggingFace datasets/spaces
2. Canary String Testing：检查工具输出是否包含答案
3. 污染轨迹丢弃：发现泄露的轨迹直接丢弃
```

**为什么重要？**
- 如果模型在评估时"看到"了答案，评估结果就不可信
- 这也是为什么不能用公开的 benchmark 做训练

---

## 学习模块 5：工程实践

### 5.1 MCP 协议的优缺点

**学到的点**：

```
优点：
1. 工具和 Agent 解耦：可以独立开发和测试工具
2. 标准化：不绑定特定 LLM 提供商
3. 可扩展：新增工具只需新增 MCP Server
4. 支持多模态：VQA、Audio 等工具可以统一接口

缺点：
1. 解析脆弱：XML 标签解析容易出错
2. 没有原生 parallel tool call：需要框架层面实现
3. 调试困难：MCP Server 是独立进程
4. 性能开销：每次工具调用都有进程启动开销
```

**工程优化**：
- 使用 SSE 模式代替 Stdio 模式，减少进程启动开销
- 对高频工具使用连接池
- 实现更健壮的 XML 解析器（而不是正则表达式）

---

### 5.2 并行评估的实现

**学到的点**：

Benchmark 评估需要处理数百个任务，每个任务可能需要 10-30 分钟：

```python
# 关键设计：ProcessPoolExecutor 而不是 ThreadPoolExecutor
# 因为 Python 有 GIL，多线程无法真正并行
with ProcessPoolExecutor(max_workers=max_concurrent) as executor:
    future_to_task_id = {
        executor.submit(_task_worker, *args): args[0]["task_id"]
        for args in worker_args
    }
```

**为什么用 ProcessPoolExecutor？**
- 每个任务需要独立的 event loop（asyncio）
- 多进程可以绕过 GIL，实现真正的并行
- 每个进程有独立的内存空间，避免状态冲突

---

### 5.3 断点续跑的实现

**学到的点**：

长时间运行的评估需要支持断点续跑：

```python
# 检查已有的 log 文件
log_pattern = f"task_{task.task_id}_attempt-{attempt}_*.json"
matching_logs = list(logs_dir.glob(log_pattern))

if matching_logs:
    # 跳过已完成的任务
    log_file = matching_logs[-1]
    with open(log_file) as f:
        log_data = json.loads(f.read())
        if log_data.get("status") == "success":
            # 已经完成，跳过
            continue
```

**关键设计**：
- 每个任务的 log 文件名包含 task_id 和 attempt
- 重启时检查已有 log，跳过已完成的任务
- 支持从失败的地方继续，而不是从头开始

---

### 5.4 错误处理与重试

**学到的点**：

MiroThinker 的错误处理非常细致：

```python
# LLM API 调用：最多重试 10 次
for attempt in range(max_retries):
    try:
        response = await client.create(...)
        if finish_reason == "length":
            # 响应被截断，增加 max_tokens 重试
            current_max_tokens = int(current_max_tokens * 1.1)
            continue
        if repeat_count > 5:
            # 检测到重复，重试
            continue
        return response
    except TimeoutError:
        await asyncio.sleep(base_wait_time)
    except Exception:
        await asyncio.sleep(base_wait_time)

# 工具调用：错误触发 rollback
if should_rollback_result(tool_name, result, tool_result):
    message_history.pop()  # 移除错误的响应
    turn_count -= 1
    consecutive_rollbacks += 1
```

**设计哲学**：
- 宁可多试几次，也不要轻易放弃
- 但有最大重试次数限制，避免无限循环
- 不同类型的错误有不同的处理策略

---

## 学习模块 6：业务理解

### 6.1 Deep Research Agent 的用户价值

**学到的点**：

```
用户真正需要的：
1. 答案准确性（最重要）
   - 错误答案比没有答案更糟糕
   - 用户信任度一旦下降就很难恢复

2. 响应速度
   - 10-30 分钟的等待是可接受的（对于深度研究）
   - 但需要进度反馈（"已搜索 15 个网页"）

3. 可解释性
   - 用户想知道答案的来源
   - 用户想验证答案的可靠性
```

---

### 6.2 成本与性能的 trade-off

**学到的点**：

```
成本结构：
├── 模型推理：~60% 的成本
│   └── 30B MoE vs 235B MoE：成本差 5-10x
├── 工具调用：~30% 的成本
│   └── 搜索 API、网页抓取、代码执行
└── 评估验证：~10% 的成本
    └── LLM-as-a-Judge

优化优先级：
1. 选择合适的模型大小（30B 在大多数场景够用）
2. 减少不必要的工具调用（重复查询检测）
3. 使用上下文管理减少 token 消耗
4. 对非关键任务使用小模型（Summary LLM）
```

---

## 学习模块 7：关键代码阅读笔记

### 7.1 orchestrator.py 主循环

**核心逻辑**：

```python
while turn < max_turns and total_attempts < max_attempts:
    # 1. LLM 调用
    response = await answer_generator.handle_llm_call(...)
    
    # 2. 解析工具调用
    tool_calls = extract_tool_calls(response)
    
    # 3. 执行工具
    for call in tool_calls:
        # 检查重复
        if is_duplicate(call): rollback(); continue
        # 执行
        result = await tool_manager.execute(call)
        # 检查错误
        if is_error(result): rollback(); continue
    
    # 4. 更新历史
    message_history = update_history(message_history, results)
    
    # 5. 检查上下文长度
    if context_too_long():
        break  # 触发 failure experience
```

---

### 7.2 answer_generator.py 上下文管理

**核心逻辑**：

```python
async def generate_and_finalize_answer(self, ...):
    context_management_enabled = self.context_compress_limit > 0
    
    # Case 1: 上下文管理开启 + 达到 max_turns + 非最终重试
    if context_management_enabled and reached_max_turns and not is_final_retry:
        # 跳过答案生成，直接生成 failure summary
        failure_summary = await self.generate_failure_summary(...)
        return FORMAT_ERROR_MESSAGE, failure_summary
    
    # Case 2: 其他情况
    # 先尝试生成答案
    answer = await self.generate_final_answer_with_retries(...)
    
    # 如果没有上下文管理，使用中间答案兜底
    if not context_management_enabled:
        if answer == FORMAT_ERROR_MESSAGE and self.intermediate_boxed_answers:
            answer = self.intermediate_boxed_answers[-1]
    
    return answer
```

---

### 7.3 base_client.py 消息过滤

**核心逻辑**：

```python
def _remove_tool_result_from_messages(self, messages, keep_tool_result):
    if keep_tool_result == -1:
        return messages  # 保留所有
    
    # 找到所有 user 消息的索引
    user_indices = [i for i, msg in enumerate(messages) 
                    if msg["role"] in ["user", "tool"]]
    
    # 第一条是初始任务，不算工具响应
    tool_result_indices = user_indices[1:]
    
    # 保留最近 K 个
    indices_to_keep = [user_indices[0]] + tool_result_indices[-keep_tool_result:]
    
    # 替换其余为占位符
    for i, msg in enumerate(messages):
        if i not in indices_to_keep and msg["role"] in ["user", "tool"]:
            msg["content"] = "Tool result is omitted to save tokens."
    
    return messages
```

**设计要点**：
- 只替换 user/tool 消息，保留 assistant 消息
- 第一条 user 消息（初始任务）永远保留
- 占位符而不是删除，保持消息列表结构完整

---

*最后更新：2026-06-27*
*基于 /data/home/yizhou/MiroThinker 代码库*
