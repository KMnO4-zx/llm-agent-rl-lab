# Search-R1：PyTRIO 复现方案

> 状态：设计稿。后续代码按照本文的数据、工具、rollout、reward、loss mask 和评估口径实现。

## 1. 方案概览

本项目训练一个能够自主调用搜索工具的多轮问答模型：

1. 模型读取问题并决定直接回答或调用搜索。
2. 搜索 query 由模型生成。
3. 环境调用知乎全局搜索 API。
4. 搜索结果以 `role="tool"` 写回消息历史。
5. 模型结合新信息继续搜索或生成最终答案。
6. 根据最终答案正确性和答案格式计算 reward。
7. 使用 group-relative advantage 和 PyTRIO `importance_sampling` 更新 LoRA。

冻结的技术选择：

- 训练框架：PyTRIO 异步训练
- 基础模型：`Qwen/Qwen3.5-4B`
- 参数更新：LoRA，rank 32
- 工具协议：`tokenizer.apply_chat_template(..., tools=[SEARCH_TOOL])`
- 搜索后端：知乎全局搜索 API
- 搜索范围：`SearchDB=all`
- 每次搜索：top 3
- 最大搜索次数：5
- 最大 assistant turns：6
- 最大轨迹长度：8,192 tokens
- 每个 rollout batch：8 道题
- 每道题采样：8 条轨迹
- Reward：答案 EM + 最终答案格式项
- Loss：`importance_sampling`
- 不部署本地向量索引、embedding 服务或搜索结果缓存库

知乎搜索是实时服务，索引和排序可能变化，因此每次评估结果都需要记录运行时间和 API 错误率。

## 2. 数据集

### 2.1 数据来源

训练和评估使用 [`PeterJinGo/nq_hotpotqa_train`](https://huggingface.co/datasets/PeterJinGo/nq_hotpotqa_train)。

每条样本使用以下字段：

- `question`：问题
- `reward_model.ground_truth` 或 `golden_answers`：一个或多个可接受答案
- `data_source`：数据来源

数据量来自发布版本 `b7d80abfee334a7a91cb377544f09180d58b34f6` 的 parquet 元数据和 `data_source` 列统计。

### 2.2 训练集

`train.parquet` 共 **169,615** 道题，下载大小约 **355.7 MB**：

| 数据来源 | 题目数 | 占比 |
| --- | ---: | ---: |
| NQ（Natural Questions） | 79,168 | 46.67% |
| HotpotQA | 90,447 | 53.33% |
| 合计 | **169,615** | 100% |

训练脚本加载完整训练池，使用固定 seed 打乱，再通过以下参数控制单次实验规模：

- `--max-steps`
- `--max-train-samples`
- `--questions-per-batch`
- `--group-size`
- `--max-micro-batch-items`
- `--max-micro-batch-tokens`

默认 `questions_per_batch=8`、`group_size=8`，即每个 rollout batch 最多生成 64 条轨迹。一次 run 的规模必须显式记录：

```text
question_groups = 实际进入 rollout 的问题数
trajectories ≤ question_groups × group_size
```

发生 API 失败、轨迹异常或整组 reward 相同时，实际进入训练的 trajectories 可能少于上限。

### 2.3 测试集与 benchmark

`test.parquet` 共 **51,713** 道题，下载大小约 **70.4 MB**：

| 类别 | Benchmark | 题目数 |
| --- | --- | ---: |
| General QA | NQ | 3,610 |
| General QA | TriviaQA | 11,313 |
| General QA | PopQA | 14,267 |
| Multi-Hop QA | HotpotQA | 7,405 |
| Multi-Hop QA | 2WikiMultiHopQA | 12,576 |
| Multi-Hop QA | MuSiQue | 2,417 |
| Multi-Hop QA | Bamboogle | 125 |
|  | **合计** | **51,713** |

评估分两档：

1. 开发评估：固定 seed，每个 benchmark 最多抽 100 条，共 700 条。
2. 完整评估：运行全部 51,713 条，报告每项 EM 和七项宏平均。

固定的是问题清单，不缓存搜索结果。

## 3. 搜索工具

### 3.1 工具声明

搜索工具使用 function schema：

```python
SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search",
        "description": "Search the web for information needed to answer the question.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A concise English search query.",
                }
            },
            "required": ["query"],
        },
    },
}
```

每轮采样前都通过模型自带 chat template 渲染消息和工具：

```python
prompt_tokens = tokenizer.apply_chat_template(
    messages,
    tools=[SEARCH_TOOL],
    tokenize=True,
    return_dict=False,
    add_generation_prompt=True,
    enable_thinking=False,
)
```

业务代码只维护结构化的 `messages` 和 `tool_calls`，不手写模型内部使用的 tool-call 特殊文本。

`apply_chat_template()` 只负责声明和渲染工具。模型生成 tool call 后，仍由 rollout 控制器解析参数并执行真实 API 请求。

### 3.2 知乎 API

API 密钥只从环境变量读取：

```bash
export ZHIHU_ACCESS_SECRET="..."
```

密钥不得写入代码、配置文件、SwanLab 日志或轨迹内容。

每次请求使用：

```text
Query=<tool call 中的 query>
Count=3
SearchDB=all
```

API 没有英文结果过滤参数。训练问题为英文，因此工具说明要求生成简洁英文 query，但搜索结果可能包含中文或英文内容。

### 3.3 Tool message

模型生成合法 tool call 后，环境保存 assistant message：

```python
{
    "role": "assistant",
    "content": reasoning_text,
    "tool_calls": [
        {
            "id": call_id,
            "type": "function",
            "function": {
                "name": "search",
                "arguments": {"query": query},
            },
        }
    ],
}
```

搜索完成后追加 tool message：

```python
{
    "role": "tool",
    "tool_call_id": call_id,
    "name": "search",
    "content": (
        "[1] Title: ...\n"
        "    Content: ...\n"
        "    URL: ...\n"
        "[2] ...\n"
        "[3] ..."
    ),
}
```

搜索结果只保留标题、摘要或正文片段、来源和 URL。单次 tool response 最多保留 1,024 tokens，并在完整搜索条目边界截断。

### 3.4 错误处理

搜索客户端使用异步 HTTP 请求，并设置：

- 请求超时
- 对 `429` 和 `5xx` 做有限次数指数退避
- 独立的搜索并发 semaphore
- API 错误转换成明确的 `role="tool"` 错误消息
- 不无限重试
- 不缓存搜索结果

API 错误不直接改变 reward。日志单独记录搜索成功率、超时率、`429` 比例和延迟。

## 4. 多轮 rollout

### 4.1 状态机

```text
question
   ↓
apply_chat_template(messages, tools=[search])
   ↓
sample assistant message
   ├─ search tool call ─→ Zhihu API ─→ append role="tool" result ─┐
   │                                                               │
   ├─ Answer: ... ─→ grade ─→ finish                              │
   │                                                               │
   └─ invalid / EOS ─→ correction message or finish                │
                                                                   │
   └────────────────── render next assistant turn ←────────────────┘
```

终止条件：

- assistant 没有 tool call，并生成合法 `Answer:`
- 已执行 5 次 search tool call
- 已完成 6 个 assistant turns
- 完整消息历史达到 8,192 tokens
- 连续生成无法解析的 tool call
- 模型 EOS 且没有可评分答案

采样只使用 tokenizer EOS / `<|im_end|>` 作为停止边界。

每轮采样前必须根据完整 prompt 的实际 token 数动态收紧生成上限，不能等生成结束后再检查轨迹长度：

```python
remaining_tokens = max_trajectory_tokens - len(prompt_tokens)
if remaining_tokens <= 0:
    return terminate_trajectory()

max_tokens_this_turn = min(
    max_tokens_per_assistant_turn,
    remaining_tokens,
)
```

`SamplingParams.max_tokens` 使用 `max_tokens_this_turn`。assistant 产生 tool call 后，tool response 还要按照 assistant completion 之后的剩余轨迹预算二次截断；若已无空间则终止轨迹。这样完整消息历史不会超过 8,192 tokens。

### 4.2 并发

一条轨迹内的多轮调用必须串行：

```text
generation 1 → search 1 → generation 2 → search 2 → ...
```

不同轨迹之间可以并发：

- 同一道题的 8 条轨迹并发
- 同一个 batch 中不同问题的轨迹并发
- 模型采样和知乎请求分别设置并发上限

每个 rollout batch 包含 8 道题、每道题 8 条轨迹，共最多 64 条轨迹。第一轮对同一道题调用 `sample_async(..., num_samples=8)`；第一次搜索后，各轨迹的 query、tool response 和消息历史已经分叉，后续分别使用 `num_samples=1` 独立推进。

按每条轨迹最多搜索 5 次计算，每个 rollout batch 理论上最多发起 320 次知乎搜索请求；这是整批上限，实际并发量由搜索 semaphore 单独限制。

## 5. Reward

最终 assistant message 必须包含且只包含一个非空 `Answer:`：

```text
correct_answer ∈ {0, 1}
correct_format = 1：恰好有一个非空的 "Answer:"
correct_format = 0：缺少、重复或 "Answer:" 后为空

reward = correct_answer + 0.1 × (correct_format - 1)
```

| 最终结果 | Reward |
| --- | ---: |
| 格式合法且答案正确 | `1.0` |
| 格式合法但答案错误 | `0.0` |
| 格式不合法或未生成最终答案 | `-0.1` |

答案比较前执行：

- 转小写
- 去标点
- 去英文冠词 `a`、`an`、`the`
- 合并多余空白
- 命中任意 ground-truth answer 即为正确

格式项只检查最终答案，不直接奖励搜索次数或 tool-call 格式，避免模型为了 reward 滥用搜索。

## 6. Advantage 与 loss mask

### 6.1 Group-relative advantage

每道题采样 `group_size=8` 条轨迹：

```text
advantage_i = reward_i - mean(group_rewards)
```

如果整组 reward 相同，所有 advantage 都是 0，该组不产生有效梯度，跳过更新并记录 `degenerate_group`。

### 6.2 Token mask

| Token 来源 | 保留在上下文 | 参与 policy loss |
| --- | --- | --- |
| system prompt、工具 schema、user question | 是 | 否 |
| 历史消息和 `role="tool"` 搜索结果 | 是 | 否 |
| 当前 assistant 生成的推理、tool call 或最终答案 | 是 | 是 |

每个 assistant turn 单独构造一个 PyTRIO `Datum`。该轮完整 prompt 作为 observation，采样返回的 completion 作为 action：

```python
observation_len = len(prompt_tokens) - 1
input_tokens = prompt_tokens + completion_tokens[:-1]

target_tokens = [0] * observation_len + completion_tokens
old_logprobs = [0.0] * observation_len + completion_logprobs
advantages = [0.0] * observation_len + [trajectory_advantage] * len(completion_tokens)
```

随后构造：

```python
datum = trio.Datum(
    model_input=trio.ModelInput.from_ints(input_tokens),
    loss_fn_inputs={
        "target_tokens": np.asarray(target_tokens, dtype=np.int64),
        "logprobs": np.asarray(old_logprobs, dtype=np.float32),
        "advantages": np.asarray(advantages, dtype=np.float32),
    },
)
```

约束：

- `input_tokens`、`target_tokens`、`logprobs` 和 `advantages` 必须右移对齐
- old logprobs 必须来自 rollout 时的 student sampler
- 同一轨迹的所有 assistant turns 使用相同的最终 trajectory advantage
- 每个 assistant action token 只训练一次
- tool schema 和 tool response 只作为上下文，advantage 为 0

### 6.3 Token-aware micro-batch 与梯度累计

Qwen3.5-4B 的单次训练请求限制为：单条 `Datum` 最多 16K tokens、一个 micro-batch 最多 32 个 `Datum`、一个 micro-batch 合计最多 57K tokens。本方案进一步把单条 `Datum` 限制为 8,192 tokens，并把 micro-batch token 上限收紧为 55,000，预留协议和边界余量：

```python
MAX_TRAIN_CONTEXT_TOKENS = 8_192
MAX_MICRO_BATCH_ITEMS = 32
MAX_MICRO_BATCH_TOKENS = 55_000
```

逻辑 rollout batch 是 8 道题乘以每题 8 条轨迹，共 64 条轨迹。reward 和 group-relative advantage 必须先在完整的 8 条同题轨迹上计算，再把生成的 `Datum` 拆成 micro-batch；不能在 micro-batch 内重新计算组均值。

即使按照一条轨迹一个 `Datum` 粗略估算，64 条满长度轨迹也有 `64 × 8,192 = 524,288` tokens，远超单次请求上限。当前实现为每个 assistant turn 构造一个 `Datum`，因此一个 rollout batch 最多产生 `64 × 6 = 384` 个 `Datum`，并且后续 turn 会重复包含历史上下文，实际待训练 token 还可能更高。不能用固定的 micro-batch size，也不能只按轨迹数估算训练 token，必须按照每个 `Datum.model_input.length` 动态装箱：

```python
def pack_micro_batches(datums):
    micro_batches = []
    micro_batch_tokens = []

    # First-fit decreasing，减少 micro-batch 数量。
    ordered = sorted(datums, key=lambda x: x.model_input.length, reverse=True)

    for datum in ordered:
        num_tokens = datum.model_input.length
        if num_tokens > MAX_TRAIN_CONTEXT_TOKENS:
            raise ValueError(f"Datum exceeds train context limit: {num_tokens}")

        for index, micro_batch in enumerate(micro_batches):
            fits_items = len(micro_batch) < MAX_MICRO_BATCH_ITEMS
            fits_tokens = (
                micro_batch_tokens[index] + num_tokens
                <= MAX_MICRO_BATCH_TOKENS
            )
            if fits_items and fits_tokens:
                micro_batch.append(datum)
                micro_batch_tokens[index] += num_tokens
                break
        else:
            micro_batches.append([datum])
            micro_batch_tokens.append(num_tokens)

    return micro_batches
```

多个 `forward_backward_async` 在同一个 `TrainingClient` 上累积梯度。所有 micro-batch 都完成后，整个 rollout batch 只执行一次 `optim_step_async`：

```python
micro_batches = pack_micro_batches(datums)

forward_backward_futures = []
for micro_batch in micro_batches:
    future = await training_client.forward_backward_async(
        micro_batch,
        loss_fn="importance_sampling",
    )
    forward_backward_futures.append(future)

forward_backward_results = await asyncio.gather(
    *forward_backward_futures,
)

optim_future = await training_client.optim_step_async(adam_params)
await optim_future
```

这里的异步表示本地可以连续提交远程 micro-batch；不假设同一 `TrainingClient` 上的多个 backward 会在服务端并行执行。若任意 micro-batch 失败，不调用 `optim_step_async`，也不能无状态地重试整个逻辑 batch，否则可能重复累积已经成功的梯度；应停止当前 run，并从最近的 training state 恢复。

## 7. 初始配置

| 配置 | 初始值 |
| --- | ---: |
| Base model | `Qwen/Qwen3.5-4B` |
| `enable_thinking` | `False` |
| LoRA rank | 32 |
| Questions per rollout batch | 8 |
| Group size | 8 |
| Num samples per question | 8 |
| Trajectories per rollout batch | 64 |
| Max search calls | 5 |
| Max assistant turns | 6 |
| Top results | 3 |
| Max tool response tokens | 1,024 |
| Max tokens per assistant turn | 1,024 |
| Max trajectory tokens | 8,192 |
| Max train context per Datum | 8,192（服务上限 16K） |
| Max micro-batch items | 32 |
| Max micro-batch tokens | 55,000（服务上限 57K） |
| Gradient accumulation | 动态 micro-batch，整批更新一次 |
| Temperature | 1.0 |
| Top-p | 1.0 |
| Learning rate | `4e-5` |
| Adam beta1 / beta2 | `0.9 / 0.95` |
| Loss | `importance_sampling` |
| Format coefficient | `0.1` |
| SearchDB | `all` |
| Max steps | 命令行显式传入 |

## 8. PyTRIO 训练循环

1. 创建 `ServiceClient` 和 LoRA `TrainingClient`。
2. 获取 tokenizer，加载并打乱训练数据。
3. 保存当前 LoRA 权重，获得 `SamplingClient`。
4. 取一个包含 8 道题的 rollout batch。
5. 为每道题并发执行 8 条多轮搜索轨迹；每个 rollout batch 最多 64 条轨迹。
6. 计算每条轨迹的 EM + format reward。
7. 计算组内 advantage，过滤 degenerate groups。
8. 把每个 assistant turn 转成带 observation mask 的 `Datum`。
9. 按每个 `Datum` 的真实 token 数动态打包 micro-batch。
10. 为所有 micro-batch 提交 `forward_backward_async(..., loss_fn="importance_sampling")` 并等待完成。
11. 整个 rollout batch 只调用一次 `optim_step_async(...)`。
12. 记录 SwanLab 指标，并按间隔保存用于推理的 sampler weights 和用于断点续训的 training state。

SwanLab 至少记录：

- `reward/mean`
- `reward/correct`
- `reward/format`
- `reward/nq`
- `reward/hotpotqa`
- `rollout/turns`
- `rollout/search_calls`
- `rollout/trajectory_tokens`
- `rollout/valid_tool_call_rate`
- `rollout/degenerate_group_rate`
- `train/datums_per_rollout_batch`
- `train/micro_batches_per_step`
- `train/tokens_per_rollout_batch`
- `train/max_micro_batch_tokens`
- `search/success_rate`
- `search/429_rate`
- `search/latency`
- PyTRIO trainer metrics

训练轨迹可以记录当时实际收到的 tool response，用于调试；这些内容不供后续请求复用。

## 9. 评估协议

base model 和训练后的 checkpoint 使用同一套 evaluator：

- 相同 chat template
- 相同工具 schema 和 tool-call parser
- 相同知乎 API 参数
- `max_search_calls=5`
- `max_assistant_turns=6`
- 每题生成 1 条轨迹

主要指标：

- 七个 benchmark 各自的 EM
- 七项 EM 宏平均
- 答案格式正确率

辅助指标：

- 平均搜索次数
- 无搜索比例
- 无效 tool-call 比例
- 平均轨迹长度
- 搜索 API 成功率和错误率

开发阶段运行固定的 700 道题；正式评估运行 51,713 道题。按每题最多搜索 5 次计算，调用上限分别为 3,500 次和 258,565 次。全量评估前先用开发集确认 API 限流、延迟和调用预算。base model 和 checkpoint 尽量在相近时间窗口交错评估，降低实时搜索变化造成的偏差。

## 10. 代码结构

```text
03-search-r1/
├── dev.md                  # 本设计文档
├── readme.md               # 教程与实验结果
├── data.py                 # 数据下载、解析、抽样和统计
├── zhihu_search.py         # 异步知乎搜索客户端
├── protocol.py             # 工具 schema、tool-call 解析和答案归一化
├── rollout.py              # 单轨迹状态机与 group 并发
├── reward.py               # EM + format reward
├── train.py                # PyTRIO 异步训练
└── eval.py                 # 700 条开发评估与全量评估
```

## 11. 验收条件

1. `apply_chat_template()` 能正确注入 search tool schema。
2. 单条问题能完成至少两轮 `assistant → search → tool response → assistant`。
3. 同一问题的 8 条轨迹能够在第一次搜索后独立分叉。
4. 一个 rollout batch 能生成最多 64 条轨迹，并在训练前完成完整的 group advantage 计算。
5. 每个 micro-batch 同时满足不超过 32 个 `Datum`、不超过 55,000 tokens，单个 `Datum` 不超过 8,192 tokens。
6. 所有 micro-batch 累积梯度后，每个 rollout batch 只执行一次 optimizer step。
7. assistant tool-call token 参与训练，tool schema 和 tool response token 的 advantage 为 0。
8. prompt、completion、old logprobs 和 advantages 的长度与右移对齐有单元测试。
9. 正确、错误和格式非法三类输出分别得到 `1.0`、`0.0` 和 `-0.1`。
10. API 超时或 `429` 不会挂死整个 batch。
11. base model 和 checkpoint 能完成同一份 700 条开发评估。
12. SwanLab 能展示 reward、搜索次数、轨迹长度、micro-batch、工具成功率和 degenerate group 变化。

## 12. 参考资料

- [Search-R1](https://arxiv.org/abs/2503.09516)
- [训练与评估数据](https://huggingface.co/datasets/PeterJinGo/nq_hotpotqa_train)
- [PyTRIO 文档](https://docs.pytrio.cn/)
- [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B)
- [知乎全局搜索 API](https://developer.zhihu.com/docs?key=global_search)
