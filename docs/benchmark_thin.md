# 瘦身模型 Benchmark 记录

日期：2026-09-28 · 环境：Windows / Python 3.12 / transformers 5.17.0 / torch 2.14.0+cpu（纯 CPU）/ 32 GB RAM

## 1. 模型文件大小

| 版本 | model.safetensors | 参数量 | 说明 |
|---|---|---|---|
| 原始完整模型 | 5,441,695,216 B (5.44 GB) | 2.72B | 含视觉塔 0.66 GB + lm_head 1.02 GB |
| 瘦身 round 1（去视觉塔） | 4,778,828,400 B (4.78 GB) | 2.39B | `model.visual.*` 移除 |
| **瘦身 round 2（当前）** | **3,762,700,816 B (3.50 GiB / 3.76 GB)** | **1.88B** | 另移除 `lm_head`（logits 头） |

瘦身产物：`outputs/wemm-text-only/`（320 个张量，仅 `model.language_model.*`）
源码实现：`models/WeMM-Embedding-2B/modeling_wemm_embedding.py`（`__init__` 删除 visual + lm_head，`forward` 直接返回 hidden states）

## 2. 内存占用（进程 RSS / Peak Working Set，同一测量脚本、独立进程）

| 加载对象 | dtype | 加载后 RSS | Peak WS |
|---|---|---|---|
| 原始完整模型 | bfloat16 | 5.53 GB | 5.61 GB |
| 瘦身 round 1 | bfloat16 | 4.92 GB | 4.99 GB |
| **瘦身 round 2（当前）** | **bfloat16** | **3.98 GB** | **4.04 GB** |
| 瘦身 round 2（当前） | float32 | 7.48 GB | 7.56 GB |

结论：相对原始模型，bf16 运行内存 **5.53 → 3.98 GB（−28%）**；`encode.py` 默认已改为 bfloat16 + `low_cpu_mem_usage`，CPU 推理走低内存路径。

## 3. 标准 Benchmark（`scripts/benchmark_thin.py`）

- 数据：`data/eval_medical.jsonl`（医疗，4 组检索）、`data/eval_commonsense.jsonl`（常识，8 组检索，自生成）
- 指标：recall@1/5/10 + MRR；Matryoshka 截断维度 2048 / 1024 / 512
- 结果 JSON：`outputs/benchmark_thin.json`

| 数据集 | dim | recall@1 | recall@5 | recall@10 | mrr |
|---|---|---|---|---|---|
| 医疗 | 2048 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 医疗 | 1024 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 医疗 | 512 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 常识 | 2048 | 0.8750 | 1.0000 | 1.0000 | 0.9062 |
| 常识 | 1024 | 0.8750 | 1.0000 | 1.0000 | 0.9062 |
| 常识 | 512 | 0.8750 | 1.0000 | 1.0000 | 0.9062 |

- 模型加载：9.4 s（文件缓存热）；吞吐 batch=8：cold 0.63 / warm 0.61 texts/s（CPU 参考实现的 linear-attention 内核较慢，GPU 可显著提升）
- benchmark 全程 Peak WS 5.55 GB（含多轮重复编码的分配缓存）
- Matryoshka 截断到 512 维指标无损（本样本量下）

## 4. 瘦身对嵌入值的影响（`scripts/compare_thin.py`）

同口径 `model.embedding()` 下，原始 vs 瘦身：逐文本余弦 mean=1.000001（bf16 噪声 ~1e-7），相似度矩阵差 max=0，检索排名完全一致——**瘦身不改变嵌入值**。
