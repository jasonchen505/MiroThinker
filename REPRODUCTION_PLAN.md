# 8×4090 算力下 MiroThinker 全流程复现 Plan

> 评估日期：2026-06-27
> 硬件配置：8 × NVIDIA RTX 4090 (24GB VRAM each, 192GB total)
> 目标：完整复现 MiroThinker 的 Agent 推理 → 评估 → Trace 收集 → SFT/DPO 后训练全流程

---

## 一、算力评估与可行性分析

### 1.1 硬件能力

| 指标 | 单卡 4090 | 8 卡合计 |
|------|-----------|----------|
| VRAM | 24 GB | 192 GB |
| FP16 算力 | 82.6 TFLOPS | 660.8 TFLOPS |
| FP8 算力 | 165.2 TFLOPS | 1321.6 TFLOPS |
| 内存带宽 | 1008 GB/s | 8064 GB/s |
| NVLink | 不支持 | 仅 PCIe 4.0 x16 |

**关键限制**：
- 无 NVLink，跨卡通信走 PCIe（~32GB/s per direction），TP 效率低于 A100
- 单卡 24GB，需要量化才能运行大模型
- PCIe 带宽限制了 Tensor Parallelism 的扩展效率

### 1.2 模型规模与可行性矩阵

| 模型 | 总参数 | 激活参数 | BF16 权重 | FP8 权重 | 8×4090 可行性 |
|------|--------|----------|-----------|----------|--------------|
| **v1.0-8B** | 8B | 8B | ~16GB | ~8GB | 单卡即可 |
| **v1.0-30B (MoE)** | 30B | 3B | ~60GB | ~30GB | TP=2 可行 |
| **v1.0-72B** | 72B | 72B | ~144GB | ~72GB | TP=8 勉强可行 |
| **v1.5-30B / v1.7-mini** | 30B | 3B | ~60GB | ~30GB | TP=2 推荐 |
| **v1.5-235B / v1.7** | 235B | 22B | ~470GB | ~235GB | **不可行** |

### 1.3 结论

| 目标 | 可行性 | 推荐方案 |
|------|--------|----------|
| 复现 Agent 推理框架 | 完全可行 | 30B MoE, TP=2 |
| 复现 Benchmark 评估 | 完全可行 | 30B MoE, 选代表性 benchmark |
| 复现 Trace Collection | 完全可行 | 自部署 Qwen3-30B 或用商业 API |
| 复现 SFT 后训练 | 可行（LoRA） | 30B MoE, LoRA, 8 卡并行 |
| 复现 DPO 后训练 | 可行（LoRA） | 30B MoE, LoRA, 8 卡并行 |
| 复现 235B 模型结果 | 不可行 | 用 30B 验证流程，235B 结果引用论文 |

---

## 二、全流程复现 Plan

### Phase 0：环境搭建（Day 1）

**目标**：安装所有依赖，验证基础环境

**步骤**：

```bash
# 1. 克隆项目
cd /data/home/yizhou
git clone https://github.com/MiroMindAI/MiroThinker
cd MiroThinker

# 2. 安装 uv 包管理器
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. 安装 Agent 框架依赖
cd apps/miroflow-agent
uv sync

# 4. 安装 Trace 收集依赖
cd ../collect-trace
uv sync

# 5. 安装推理框架
pip install sglang[all]
# 或
pip install vllm

# 6. 配置环境变量
cd ../miroflow-agent
cp .env.example .env
# 编辑 .env 填入 API keys
```

**API Keys 申请**：
- Serper：https://serper.dev/ （免费 2500 次搜索）
- Jina：https://jina.ai/ （免费额度）
- E2B：https://e2b.dev/ （免费额度）
- OpenAI：用于 LLM-as-a-Judge（需要 GPT-5 access）

**验证命令**：
```bash
cd apps/miroflow-agent
uv run python main.py llm=qwen-3 agent=single_agent_keep5 llm.base_url=http://localhost:61005/v1
```

---

### Phase 1：模型部署（Day 1-2）

**目标**：在 8×4090 上部署 MiroThinker 模型

#### 方案 A：SGLang + FP8（推荐）

**模型选择**：MiroThinker-1.7-mini (30B MoE, 3B active)

```bash
# 下载模型
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='miromind-ai/MiroThinker-1.7-mini',
    local_dir='/data/home/yizhou/models/MiroThinker-1.7-mini'
)
"

# 部署：TP=2, 使用 2 张 4090
NUM_GPUS=2
PORT=61002

python3 -m sglang.launch_server \
    --model-path /data/home/yizhou/models/MiroThinker-1.7-mini \
    --tp $NUM_GPUS \
    --dp 1 \
    --quantization fp8 \
    --mem-fraction-static 0.85 \
    --host 0.0.0.0 \
    --port $PORT \
    --trust-remote-code
```

**显存估算**：
- FP8 权重：~30GB → TP=2 每卡 ~15GB
- KV Cache（128K context）：~8GB per card
- 总计：每卡 ~23GB，接近 4090 上限

**如果显存不足**：
```bash
# 方案 1：降低 context length
--context-length 65536  # 64K instead of 256K

# 方案 2：降低 mem-fraction-static
--mem-fraction-static 0.8

# 方案 3：使用 TP=4（4 张卡）
NUM_GPUS=4
```

#### 方案 B：llama.cpp + Q4 量化（轻量）

```bash
# 下载量化模型
wget https://huggingface.co/mradermacher/MiroThinker-v1.5-30B-GGUF/resolve/main/MiroThinker-v1.5-30B.Q4_K_M.gguf

# 部署（单卡即可）
llama-server -m MiroThinker-v1.5-30B.Q4_K_M.gguf \
    --port 61005 \
    -ngl 99 \
    -v
```

#### 验证部署

```bash
curl http://localhost:61002/v1/models
curl http://localhost:61002/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "MiroThinker-1.7-mini",
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "max_tokens": 100
  }'
```

---

### Phase 2：Benchmark 评估（Day 2-4）

**目标**：在代表性 benchmark 上评估模型

#### 2.1 选择 Benchmark

由于算力有限，选择**最有代表性的 3 个 benchmark**：

| Benchmark | 任务数 | 运行次数 | 预估时间 | 选择理由 |
|-----------|--------|----------|----------|----------|
| **GAIA-Text-103** | 103 | 3 | ~8h | 综合能力评估，MiroThinker 核心 benchmark |
| **HLE-Text-500** | 500 | 3 | ~12h | 知识深度评估，文本子集 |
| **BrowseComp-ZH** | ~100 | 3 | ~6h | 中文搜索能力，MiroThinker 的优势项 |

#### 2.2 运行评估

```bash
cd apps/miroflow-agent

# 设置通用变量
export BASE_URL="http://localhost:61002/v1"
export LLM_MODEL="MiroThinker-1.7-mini-4090"
export MAX_CONCURRENT=5  # 降低并发，避免显存压力
export MAX_CONTEXT_LENGTH=131072  # 128K，降低显存需求

# Benchmark 1: GAIA-Text-103
NUM_RUNS=3 bash scripts/run_evaluate_multiple_runs_gaia-validation-text-103.sh

# Benchmark 2: HLE-Text-500
NUM_RUNS=3 bash scripts/run_evaluate_multiple_runs_hle-text-500.sh

# Benchmark 3: BrowseComp-ZH
NUM_RUNS=3 AGENT_SET="mirothinker_1.7_keep5_max300" bash scripts/run_evaluate_multiple_runs_browsecomp_zh.sh
```

#### 2.3 监控进度

```bash
# 查看 GAIA 进度
python benchmarks/check_progress/check_progress_gaia-validation-text-103.py ../../logs/xxx

# 查看实时日志
tail -f ../../logs/xxx/output.log
```

#### 2.4 预期结果

| Benchmark | 论文结果 (235B) | 预期结果 (30B, 8×4090) | 差距原因 |
|-----------|----------------|----------------------|----------|
| GAIA-Text-103 | 82.7% | ~55-65% | 模型规模差异 |
| HLE-Text | 42.9% | ~25-35% | 模型规模差异 |
| BrowseComp-ZH | 75.3% | ~55-65% | 模型规模差异 |

---

### Phase 3：Trace Collection（Day 4-7）

**目标**：收集 SFT 和 DPO 训练数据

#### 3.1 数据准备

```bash
# 下载 RLVR 数据集（如果有公开的）
# 或者使用 GAIA-Text-103 作为训练数据源
cd /data/home/yizhou/MiroThinker
mkdir -p data/rlvr

# 准备数据格式（JSONL）
# 每行包含：task_id, task_question, ground_truth
```

#### 3.2 SFT Trace Collection（使用商业 API）

使用 Claude-3.7 或 GPT-5 收集高质量轨迹：

```bash
cd apps/collect-trace

# 方案 1：使用 Claude-3.7（推荐，质量最高）
export ANTHROPIC_API_KEY="your_key"
bash scripts/collect_trace_claude37.sh

# 方案 2：使用 GPT-5
export OPENAI_API_KEY="your_key"
bash scripts/collect_trace_gpt5.sh
```

#### 3.3 DPO Trace Collection（使用自部署模型）

收集正确和错误轨迹对：

```bash
cd apps/collect-trace

# 使用自部署的 Qwen3-30B 收集轨迹
bash scripts/collect_trace_qwen3.sh
# 需要修改脚本中的 base_url 为你的部署地址
```

#### 3.4 Trace 后处理

```bash
# 处理日志，提取成功轨迹
cd apps/collect-trace
uv run python utils/process_logs.py ../../logs/collect_trace_claude37/benchmark_results.jsonl

# 输出在 successful_logs/ 目录下
# 然后转换为 ChatML 格式
uv run python utils/converters/convert_to_chatml_auto_batch.py ../../logs/collect_trace_claude37/successful_logs/
```

#### 3.5 数据量目标

| 数据类型 | 目标数量 | 用途 |
|----------|----------|------|
| SFT 训练数据 | 500-2000 条 | 学习工具调用模式 |
| DPO 训练对 | 200-500 对 | 优化工具选择偏好 |

---

### Phase 4：SFT 后训练（Day 7-14）

**目标**：用收集的轨迹微调基座模型

#### 4.1 训练框架选择

由于 MiroThinker 仓库不包含训练代码，使用外部框架：

| 框架 | 优势 | 劣势 | 推荐度 |
|------|------|------|--------|
| **LLaMA-Factory** | 支持 MoE、LoRA、多卡 | 需要额外安装 | ⭐⭐⭐⭐⭐ |
| **OpenRLHF** | 原生支持 DPO | 配置复杂 | ⭐⭐⭐⭐ |
| **TRL** | HuggingFace 官方 | 对 MoE 支持有限 | ⭐⭐⭐ |

**推荐：LLaMA-Factory**

#### 4.2 环境准备

```bash
# 安装 LLaMA-Factory
pip install llamafactory

# 或从源码安装
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e .
```

#### 4.3 数据格式转换

将收集的 ChatML 数据转换为 LLaMA-Factory 格式：

```python
# 转换脚本
import json

def convert_to_llamafactory(input_file, output_file):
    """将 MiroThinker trace 转换为 LLaMA-Factory SFT 格式"""
    data = []
    with open(input_file) as f:
        for line in f:
            record = json.loads(line)
            # 提取 system prompt 和对话
            messages = record.get("messages", [])
            conversations = []
            for msg in messages:
                if msg["role"] == "system":
                    conversations.append({"from": "system", "value": msg["content"]})
                elif msg["role"] == "user":
                    conversations.append({"from": "human", "value": msg["content"]})
                elif msg["role"] == "assistant":
                    conversations.append({"from": "gpt", "value": msg["content"]})
            data.append({"conversations": conversations})
    
    with open(output_file, 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

convert_to_llamafactory(
    "../../logs/collect_trace_claude37/successful_logs/chatml_data.jsonl",
    "data/sft_train.json"
)
```

#### 4.4 SFT 训练配置

```yaml
# sft_config.yaml
### model
model_name_or_path: /data/home/yizhou/models/Qwen3-30B-A3B-Thinking-2507
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: lora
lora_target: all  # 对 MoE 模型，target all layers
lora_rank: 64
lora_alpha: 128
lora_dropout: 0.05

### dataset
dataset: mirothinker_sft
template: qwen3
cutoff_len: 32768  # 32K context for training
max_samples: 2000
overwrite_cache: true

### output
output_dir: /data/home/yizhou/models/MiroThinker-30B-SFT
logging_steps: 10
save_steps: 500
save_total_limit: 3

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 1.0e-4
num_train_epochs: 3
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000

### multi-GPU
deepspeed: examples/deepspeed/ds_z2_config.json
```

#### 4.5 启动训练

```bash
# 8 卡 DDP 训练
cd LLaMA-Factory

# 方案 1：使用 llamafactory-cli
llamafactory-cli train sft_config.yaml

# 方案 2：使用 torchrun
torchrun --nproc_per_node=8 src/train.py \
    --model_name_or_path /data/home/yizhou/models/Qwen3-30B-A3B-Thinking-2507 \
    --finetuning_type lora \
    --lora_rank 64 \
    --dataset mirothinker_sft \
    --output_dir /data/home/yizhou/models/MiroThinker-30B-SFT \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 3 \
    --learning_rate 1e-4 \
    --bf16 true
```

#### 4.6 合并 LoRA 权重

```bash
# 训练完成后，合并 LoRA 到基座模型
llamafactory-cli export \
    --model_name_or_path /data/home/yizhou/models/Qwen3-30B-A3B-Thinking-2507 \
    --adapter_name_or_path /data/home/yizhou/models/MiroThinker-30B-SFT \
    --export_dir /data/home/yizhou/models/MiroThinker-30B-SFT-merged \
    --export_size 2 \
    --export_legacy_format false
```

---

### Phase 5：DPO 后训练（Day 14-21）

**目标**：用正确/错误轨迹对优化模型偏好

#### 5.1 准备 DPO 数据

```python
# 从 Trace Collection 中提取 chosen/rejected 对
def prepare_dpo_data(successful_traces, failed_traces):
    """
    successful_traces: 正确轨迹列表
    failed_traces: 错误轨迹列表
    按 task_id 配对
    """
    dpo_data = []
    success_by_id = {t["task_id"]: t for t in successful_traces}
    fail_by_id = {t["task_id"]: t for t in failed_traces}
    
    for task_id in success_by_id:
        if task_id in fail_by_id:
            dpo_data.append({
                "conversations": extract_context(success_by_id[task_id]),
                "chosen": extract_response(success_by_id[task_id]),
                "rejected": extract_response(fail_by_id[task_id]),
            })
    
    return dpo_data
```

#### 5.2 DPO 训练配置

```yaml
# dpo_config.yaml
### model
model_name_or_path: /data/home/yizhou/models/MiroThinker-30B-SFT-merged
trust_remote_code: true

### method
stage: dpo
do_train: true
finetuning_type: lora
lora_target: all
lora_rank: 64
lora_alpha: 128

### dataset
dataset: mirothinker_dpo
template: qwen3
cutoff_len: 32768

### output
output_dir: /data/home/yizhou/models/MiroThinker-30B-DPO

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 5.0e-5
num_train_epochs: 2
bf16: true
dpo_beta: 0.1  # DPO 温度参数
```

#### 5.3 启动 DPO 训练

```bash
llamafactory-cli train dpo_config.yaml
```

---

### Phase 6：验证与对比（Day 21-25）

**目标**：评估后训练模型，对比效果

#### 6.1 部署后训练模型

```bash
# 部署 SFT 模型
python3 -m sglang.launch_server \
    --model-path /data/home/yizhou/models/MiroThinker-30B-SFT-merged \
    --tp 2 --quantization fp8 --port 61003 --trust-remote-code

# 部署 DPO 模型
python3 -m sglang.launch_server \
    --model-path /data/home/yizhou/models/MiroThinker-30B-DPO-merged \
    --tp 2 --quantization fp8 --port 61004 --trust-remote-code
```

#### 6.2 对比评估

```bash
cd apps/miroflow-agent

# 基座模型
LLM_MODEL="Qwen3-30B-Base" BASE_URL="http://localhost:61002/v1" \
    NUM_RUNS=3 bash scripts/run_evaluate_multiple_runs_gaia-validation-text-103.sh

# SFT 模型
LLM_MODEL="MiroThinker-30B-SFT" BASE_URL="http://localhost:61003/v1" \
    NUM_RUNS=3 bash scripts/run_evaluate_multiple_runs_gaia-validation-text-103.sh

# DPO 模型
LLM_MODEL="MiroThinker-30B-DPO" BASE_URL="http://localhost:61004/v1" \
    NUM_RUNS=3 bash scripts/run_evaluate_multiple_runs_gaia-validation-text-103.sh
```

#### 6.3 预期结果对比

| 模型 | GAIA-Text-103 | 提升 |
|------|--------------|------|
| Qwen3-30B 基座 | ~30% | baseline |
| + SFT | ~45-50% | +15-20% |
| + SFT + DPO | ~48-53% | +3-5% |

---

## 三、资源分配方案

### 3.1 GPU 分配时间表

| Phase | 时间 | GPU 使用 | 具体分配 |
|-------|------|----------|----------|
| Phase 1 | Day 1-2 | 2 卡 | 2×4090 for SGLang TP=2 |
| Phase 2 | Day 2-4 | 2 卡 | 2×4090 for 推理 + 6 卡空闲 |
| Phase 3 | Day 4-7 | 2 卡 | 2×4090 for 推理（Trace 收集） |
| Phase 4 | Day 7-14 | 8 卡 | 8×4090 for SFT 训练 |
| Phase 5 | Day 14-21 | 8 卡 | 8×4090 for DPO 训练 |
| Phase 6 | Day 21-25 | 2 卡 | 2×4090 for 评估 |

### 3.2 并行优化

如果需要加速，可以并行执行：

```
方案：2 卡推理 + 6 卡训练 并行
├── GPU 0-1: SGLang 推理（TP=2）
├── GPU 2-7: 训练（6 卡 DDP）
└── 注意：需要确保 PCIe 带宽不冲突
```

---

## 四、风险与应对

| 风险 | 概率 | 影响 | 应对方案 |
|------|------|------|----------|
| 显存不足 | 中 | 无法运行 | 降低 context length、使用 Q4 量化 |
| PCIe 带宽瓶颈 | 中 | TP 效率低 | 减少 TP 度数、增加 DP |
| 训练不收敛 | 低 | 浪费时间 | 降低学习率、增加 warmup |
| API 额度不足 | 中 | 无法收集 trace | 使用自部署模型收集、减少数据量 |
| 评估时间过长 | 高 | 进度延迟 | 减少 benchmark 数量、减少运行次数 |

---

## 五、时间线总览

```
Week 1:
├── Day 1: 环境搭建 + 模型下载
├── Day 2: 模型部署 + 验证
├── Day 3-4: Benchmark 评估（GAIA-Text-103）
└── Day 5-7: Trace Collection（SFT 数据）

Week 2:
├── Day 8-10: Trace Collection（DPO 数据）
├── Day 11-14: SFT 训练
└── （并行：继续收集更多 trace）

Week 3:
├── Day 15-18: DPO 训练
├── Day 19-21: 评估后训练模型
└── Day 22-25: 对比分析 + 文档整理
```

---

## 六、快速验证路径（如果时间紧张）

如果只有 1 周时间，可以走快速路径：

```
Day 1: 环境搭建 + 模型部署（SGLang FP8, TP=2）
Day 2: 单个 benchmark 评估（GAIA-Text-103, 10 题子集）
Day 3: 用商业 API 收集 100 条 SFT trace
Day 4-5: LoRA SFT 训练（8 卡，100 条数据，3 epochs）
Day 6: 评估 SFT 模型
Day 7: 对比分析 + 总结
```

**快速验证的核心**：用 100 条数据 + LoRA SFT 就能看到效果，不需要完整的全流程。

---

## 七、成本估算

### 7.1 外部 API 成本

| API | 用途 | 预估用量 | 单价 | 总成本 |
|-----|------|----------|------|--------|
| Serper | Google 搜索 | 50,000 次 | $1/1000次 | ~$50 |
| Jina | 网页抓取 | 50,000 次 | $0.002/次 | ~$100 |
| E2B | 代码沙箱 | 10,000 分钟 | $0.01/分钟 | ~$100 |
| OpenAI (Judge) | 答案验证 | 10,000 次 | $0.01/次 | ~$100 |
| **Trace Collection（Claude）** | SFT 数据 | 2,000 条 | $0.5/条 | ~$1,000 |
| **总计** | | | | **~$1,350** |

### 7.2 本地算力成本

| 阶段 | 时间 | 电费（8×4090, ~3kW） | 总计 |
|------|------|---------------------|------|
| 推理评估 | 7 天 | ~$15/天 | ~$105 |
| SFT 训练 | 7 天 | ~$15/天 | ~$105 |
| DPO 训练 | 7 天 | ~$15/天 | ~$105 |
| **总计** | 21 天 | | **~$315** |

### 7.3 总成本

- **外部 API**：~$1,350
- **本地算力**：~$315
- **总计**：~$1,665

**降低成本的方法**：
1. 使用自部署模型收集 trace（节省 $1,000）
2. 减少 benchmark 数量（节省 API 成本）
3. 使用 LoRA 而非全参训练（节省时间）

---

*最后更新：2026-06-27*
