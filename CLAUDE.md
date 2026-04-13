# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

HMEMS（Hierarchical Memory Enhancement System）是一个分层记忆增强系统，用于长期对话记忆管理。系统采用三层记忆架构：

1. **向量记忆（VecMem）**：原始对话片段的向量存储，使用余弦相似度检索
2. **情节记忆（Episodic Memory）**：通过 LLM 增强的高级记忆，自动合并相关原始记忆
3. **语义记忆（Semantic Memory）**：从情节记忆中提取的事实性知识

## 环境配置

### Conda 环境设置

```bash
conda create -n vecmem python=3.9 -y
conda activate vecmem
pip install -r requirements.txt
```

### 环境变量配置（.env 文件）

必需的环境变量：

```bash
# OpenAI API 配置
OPENAI_API_KEY=<your_api_key>

# 模型配置（三个模型）
MODEL0=<embedding_model>  # 嵌入模型（如 text-embedding-ada-002）
MODEL1=<memory_ops_model>  # 记忆操作模型（记忆增强、语义提取）
MODEL2=<answer_gen_model>  # 回答生成模型

# API 端点配置（可选，用于第三方提供商）
M0_BASE_URL=<base_url_for_embeddings>
M1_BASE_URL=<base_url_for_memory_ops>
M2_BASE_URL=<base_url_for_answer_gen>

# 数据路径
LOCOMO_PATH=<path_to_locomo10.json>
LOCOMO_EMBEDDING_PATH=<path_to_embeddings>
LOCOMO_INDEX_PATH=<path_to_indices>
LOCOMO_RES_PATH=<path_to_answers>
LOCOMO_SCORE_PATH=<path_to_scores>
```

## 常用命令

### 初始化环境（生成嵌入和索引）

```bash
python run_experiments.py --init_env
```

### 运行单个实验

```bash
python run_experiments.py \
    --min_aug_count 3 \
    --min_relevant_score 0.7 \
    --retrieve_raw_topk 5 \
    --retrieve_aug_topk 5 \
    --output_file ${LOCOMO_RES_PATH}/experiment_name.json
```

### 使用 run.sh 批量运行实验

```bash
# 运行所有配置
./run.sh

# 运行特定配置
./run.sh config1 config2

# 日志保存在 logs/ 目录
```

### 仅评估已有结果

```bash
python run_experiments.py --eval_only --output_file <path_to_results.json>
```

## 核心架构

### 文件组织

```
HMEMS/
├── vec_mem.py           # VecMem 核心类，三层记忆系统的主入口
├── conv_loader.py       # Locomo 数据集加载器
├── embed_manager.py     # 嵌入生成和管理
├── pipeline.py          # 数据处理流水线（嵌入生成、索引构建）
├── run_experiments.py   # 实验运行主脚本
├── prompt.py            # 所有提示词模板
├── token_monitor.py     # Token 使用统计
├── aug_methods/         # 情节记忆增强方法
│   ├── naive_aug.py     # NaiveAugMem 实现
│   └── aug_config       # 增强配置
├── vector_store/        # 向量存储实现
│   ├── flat_index.py    # 简单的平面向量索引
│   ├── faiss_index.py   # FAISS 向量索引
│   └── naive_store.py   # 朴素存储实现
├── metrics/             # 评估指标
│   ├── llm_judge.py     # LLM 判别器评估
│   └── utils.py         # BLEU、F1 等指标计算
├── episodic_memory.py   # EpisodicNote 数据类
└── semantic_memory.py   # 语义记忆实现
```

### VecMemConfig 关键参数

```python
@dataclass
class VecMemConfig:
    # 通用设置
    min_aug_count: int = 3           # 触发记忆增强的最小相关记忆数
    min_relevant_score: float = 0.7  # 记忆相关性阈值
    merge_with_aug_thresh: float = 0.85  # 与情节记忆合并的阈值

    # 检索阶段 topk
    retrieve_raw_topk: int = 5       # 从向量存储检索的数量
    retrieve_aug_topk: int = 5       # 从情节记忆检索的数量

    # 迭代回答（可选）
    enable_iter_anwser: bool = False
    iterative_raw_topk: int = 3
    iterative_aug_topk: int = 3
    iter_max_depth: int = 3

    # 语义记忆（可选）
    enable_semantic_memory: bool = False
    semantic_memory_topk: int = 10
    semantic_memory_threshold: float = 0.5

    # 开发测试
    conv_limit: Optional[int] = None  # 限制处理的对话数量
```

### 记忆添加流程

1. **检查情节记忆合并**：新记忆与现有情节记忆比较
2. **搜索相似向量记忆**：从向量存储中查找相关记忆
3. **记忆增强**：当相关记忆数量 >= min_aug_count 时，触发 LLM 生成情节记忆
4. **清理向量存储**：被增强的原始记忆从向量存储中移除
5. **语义提取**（可选）：从情节记忆中提取事实性语义记忆

### 记忆检索流程

1. **并行检索**：同时从向量存储和情节记忆检索
2. **语义记忆检索**（可选）：从语义记忆中检索相关事实
3. **迭代检索**（可选）：基于初始结果生成新查询，迭代优化
4. **答案生成**：使用检索到的上下文生成答案

## 提示词系统

所有提示词定义在 `prompt.py` 中，使用 Jinja2 模板：

- `ANSWER_PROMPT_VECMEM`：使用向量和情节记忆生成答案
- `ANSWER_PROMPT_WITH_SEMANTIC`：包含语义记忆的答案生成
- `ITERATIVE_ANWSER_PROMPT_VECMEM`：迭代检索的提示词
- `MEMORY_AUGMENT_PROMPT`：生成情节记忆的提示词
- `MEMORY_AUGMENT_MERGE_PROMPT`：合并情节记忆的提示词
- `SEMANTIC_EXTRACTION_PROMPT`：从新情节记忆提取语义的提示词
- `SEMANTIC_EXTRACTION_DURING_MERGE_PROMPT`：合并时提取语义的提示词

## 评估指标

系统使用三种评估指标：

1. **BLEU-1**：n-gram 重叠度
2. **F1 Score**：词级别的召回率和精确率
3. **LLM Judge**：使用 GPT-4o-mini 评估答案质量（0-1 分数）

评估按五个类别分别统计：
- Category 1：多跳推理（multihop）
- Category 2：时序推理（temporal）
- Category 3：开放域（open_domain）
- Category 4：单跳推理（singlehop）
- Category 5：对抗性问题（adversarial，默认跳过）

## Token 监控

启用 token 统计（用于分析成本）：

```bash
python run_experiments.py --enable_stat ...
```

统计结果保存在 `_token_stats.json` 文件中，按操作类型分类：
- `MEMORY_AUGMENT`：记忆增强（生成情节记忆）
- `MEMORY_MERGE`：记忆合并（合并现有情节记忆）
- `SEMANTIC_EXTRACTION`：语义提取（从新情节记忆中提取）
- `SEMANTIC_EXTRACTION_DURING_MERGE`：合并时语义提取
- `ITERATIVE_FILTER`：迭代检索过滤
- `ANSWER`：答案生成

## Agent 配置

项目支持使用 Agent 进行记忆添加（可选）。Agent 配置文件格式（YAML）：

```yaml
agent_name: "memory_agent"
model_name: "qwen3-4b"  # 或其他支持的模型
model_path: "/path/to/model"  # 可选，本地模型路径
enable_thinking: true  # 是否启用思维链
max_new_tokens: 2048  # 最大生成 token 数
```

使用 agent 添加记忆：

```bash
python run_experiments.py --agent_config config/qwen3-4b.yaml ...
```

默认配置：`config/qwen3-4b.yaml`（需要手动创建）

## 开发注意事项

### 修改提示词

提示词在 `prompt.py` 中定义。修改后无需重新生成嵌入，直接运行实验即可。

### 调试单个对话

使用 `--conv_limit` 参数限制处理的对话数量：

```bash
python run_experiments.py --conv_limit 1 ...
```

### 检查记忆内容

在 `run_experiments.py` 的 `run_experiment` 函数中，`FINAL_RESULTS` 会保存：
- 向量存储状态
- 情节记忆存储状态
- 语义记忆存储状态
- 每个问题的答案

### 向量存储实现

- 开发/测试：使用 `FlatIndex`（简单、可调试）
- 生产环境：使用 `FAISSIndex`（更快的检索）

切换方式在 `pipeline.py` 和 `vec_mem.py` 中修改导入。

## 数据格式

### Locomo 数据集格式

```json
{
  "sample_id": "conversation_id",
  "conversation": {
    "speaker_a": "Alice",
    "speaker_b": "Bob",
    "session_1": [...],
    "session_1_date_time": "2022-05-04 12:00:00"
  },
  "qa": [
    {
      "question": "...",
      "answer": "...",
      "evidence": "...",
      "category": 1
    }
  ]
}
```

### 输出结果格式

```json
{
  "0": [  // 对话索引
    {"function_calls": [...]},
    {/* vector_store 状态 */},
    {/* aug_mem 状态 */},
    {/* semantic_memory 状态 */},
    {
      "question": "...",
      "answer": "...",
      "category": 1,
      "response": "..."
    }
  ]
}
```

## 常见问题

### API 调用失败

检查 `.env` 文件中的 `M1_BASE_URL` 和 `M2_BASE_URL` 是否正确配置。如果使用官方 OpenAI API，可以省略这些变量。

### 嵌入维度不匹配

系统默认使用 OpenAI `text-embedding-ada-002`（1536 维）。如需更换嵌入模型，需要相应修改 `FlatIndex` 的 `embedding_dim` 参数。

### 评估卡在 LLM Judge

LLM Judge 使用 `MODEL1` 环境变量指定的模型，可以使用更快的模型（如 `gpt-4o-mini`）来加速评估。
