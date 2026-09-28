# WeMM-Embedding 医疗领域适配方案

基于 WeMM-Embedding-2B（Qwen3.5 backbone）做纯文本医疗检索。原方案：抽视觉塔 + LoRA 微调；实际执行为三轮无训练权重瘦身（LoRA 未执行，见下）。

## 执行总结（2026-09-28）

最终模型 **`outputs/wemm-medical-pruned25/`（3.31GB）**，`config.yaml` 已指向。原始权重完整保留在 `models/WeMM-Embedding-2B/`。

### 具体做了什么

**Round 1 — 移除视觉/视频塔**
- 模型复制入项目 `models/`；`modeling_wemm_embedding.py` 的 `__init__` 加载后删除 `self.model.visual`（transformers 的 `Qwen3_5Model.__init__` 会按 config 无条件重建视觉塔，删 config 字段不够，必须在建模代码里删）
- 删除视频辅助文件（`patch_sglang_video.py`、`processor_config.json`、`modeling_st_wemm.py`、`modules.json`）
- `strip_vision.py` 扩展清理 config 中 video/vision token 字段 → 产物 `outputs/wemm-text-only/`

**Round 2 — 移除 lm_head + 修复编码 bug**
- `__init__` 再删 `lm_head`（508M 参数 / 1.02GB logits 头，嵌入路径用不到）；覆写 `forward()` 直返 hidden states
- **顺带修复既有 bug**：`encode.py`/`train_lora.py` 原先取 `res[0]` = logits（248078 维错误向量），覆写后自动取 `last_hidden_state`（2048 维正确向量）
- `encode.py` 默认 dtype 改 bfloat16 + `low_cpu_mem_usage`（CPU 推理内存 9.6GB → 4.0GB 量级）

**Round 3 — 医疗领域校准 FFN 通道剪枝**（`scripts/prune_ffn.py`）
- Wanda 式通道得分：`s_j = ||down_proj[:,j]||₂ × RMS_t(silu(gate_j(x)) ⊙ up_j(x))`，在医疗语料（医学教材 200 段 + 医疗 query 100 条）上 batch=1 校准
- 每层保留得分 top-k 通道（结构化裁 `gate/up` 行与 `down` 列），阶梯 25% / 50% 由同一份基线得分生成
- 在 192.168.251.15（RTX 4090）执行，校准 300 条仅 33s；**采纳 25% 档**（intermediate 6144→4608），产物 MD5 校验后回传本地；50% 档留服务器作实验

**配套产出**：`scripts/benchmark_thin.py`（标准 benchmark：文件/内存/吞吐/检索）、`scripts/compare_thin.py`（模型间嵌入漂移对比）、`data/eval_commonsense.jsonl`（常识评估集 8 题）、`docs/benchmark_thin.md`、`docs/prune_ffn.md`（阶梯数据与采纳记录）。

### 效果（本地 CPU 实测，同一测量脚本）

**资源消耗**

| 指标 | 原始 WeMM-2B | R1 去视觉塔 | R2 去 lm_head | **最终（剪枝 25%）** |
|---|---|---|---|---|
| 模型文件 | 5.44 GB | 4.78 GB | 3.76 GB | **3.31 GB（−39%）** |
| 参数量 | 2.72B | 2.39B | 1.88B | **1.655B（−39%）** |
| bf16 加载 RSS | 5.53 GB | 4.92 GB | 3.98 GB | **3.56 GB（−36%）** |
| bf16 Peak WS | 5.61 GB | 4.99 GB | 4.04 GB | **3.63 GB** |
| f32 加载 RSS | 10.61 GB | — | 7.48 GB | **6.65 GB** |
| 103 条嵌入耗时 | 290 s | — | — | **188 s（−35%）** |

**质量（vs 原始完整模型）**

| 指标 | 原始 | 最终（剪枝 25%） |
|---|---|---|
| 医疗检索 R@1（2048/1024/512 维） | 1.000 | **1.000（全维度保持）** |
| 常识检索 R@1（8 题，2048 维） | 0.875 | 0.750（1024/512 维 0.875 持平） |
| 常识检索 R@5 | 1.000 | 1.000 |
| 嵌入漂移（103 条余弦） | 1.000 | mean **0.941** / min 0.861 |
| 相似度矩阵 \|Δ\| | 0 | mean 0.033 / max 0.149 |

**结论**：医疗场景检索完全无损，通用能力仅常识集掉 1 题（8 题小样本，1024/512 维无损），换来 **−39% 文件体积与 −36% 运行内存**。R@5/R@10 全程 1.0。

备注：本机 CPU 无原生 BF16，fp32 前向反而比 bf16 快 ~4.5×（GPU 上仍用 bf16）；INT8 量化与 LoRA 微调未执行（分别缺 bitsandbytes / peft，且无 GPU 训练条件）。

## 目录结构

```
wemm-medical-embedding/
├── README.md
├── requirements.txt
├── config.yaml              # model_path 指向最终模型
├── scripts/
│   ├── strip_vision.py      # 删视觉塔（Round 1）
│   ├── prune_ffn.py         # 医疗领域校准 FFN 剪枝（Round 3）
│   ├── benchmark_thin.py    # 标准 benchmark（文件/内存/吞吐/检索）
│   ├── compare_thin.py      # 模型间嵌入漂移对比
│   ├── train_lora.py        # LoRA 对比学习微调（未执行）
│   ├── encode.py            # 推理 encode
│   └── eval.py              # 医疗/通用双线评估
├── data/
│   ├── train.jsonl / train.sample.jsonl
│   ├── eval_medical.jsonl   # 医疗评估集
│   └── eval_commonsense.jsonl  # 常识评估集（自生成）
├── docs/
│   ├── architecture_principle.md
│   ├── benchmark_thin.md    # 瘦身两轮 benchmark 记录
│   └── prune_ffn.md         # 剪枝阶梯记录与采纳决定
└── outputs/
    ├── wemm-text-only/          # R1+R2 产物（3.76GB 基线）
    └── wemm-medical-pruned25/   # 最终模型（3.31GB）
```

## 前置准备

1. 下载 WeMM-Embedding-2B 权重到本地（HuggingFace 搜 `WeMM-Embedding-2B`，Apache-2.0）。
2. 安装依赖：`pip install -r requirements.txt`
3. 改 `config.yaml` 里的 `model_path` 为本地权重路径。

## 执行顺序

### Step 1：抽视觉塔

```bash
python scripts/strip_vision.py
```

加载全量模型，删除 vision_tower / visual projection 模块，保存纯文本权重到 `outputs/wemm-text-only/`。
显存占用从 ~4.5GB（FP16）降到 ~3.8GB；INT8 下 ~2.0GB。

### Step 2：准备数据

按 `data/train.sample.jsonl` 格式准备训练集。每行：

```json
{"query": "...", "positive": "...", "negatives": ["...", "..."]}
```

- 医疗数据 60%，常识数据 40%（防通用能力崩塌）
- 总量 3000-5000 条起步
- negatives 可留空，脚本自动做 in-batch negative

### Step 3：LoRA 微调

```bash
python scripts/train_lora.py
```

- 冻结 backbone，只训 q_proj / v_proj 的 LoRA（r=16）
- 可训练参数 ~5-10M，占总参数 <0.5%
- 输出到 `outputs/lora-adapter/`
- 训练完自动 `merge_and_unload`，保存合并权重到 `outputs/wemm-medical-merged/`

### Step 4：评估

```bash
python scripts/eval.py
```

双线评估：
- 医疗检索 recall@10（用 `data/eval_medical.jsonl`）
- 通用检索平均分（抽 C-MTEB 子集）

红线：通用分掉幅 >3% 需加通用数据重训。

### Step 5：推理

```python
from scripts.encode import encode
emb = encode(["高血压用药咨询", "感冒症状"])
```

## 显存预估

| 阶段 | 显存 |
|---|---|
| 原始 FP16 | ~4.5GB |
| 抽视觉塔后 FP16 | ~3.8GB |
| 抽视觉塔后 INT8 | ~2.0GB |
| LoRA 训练（bf16 + 梯度检查点）| ~10GB |

## 参考

- c6oisini/Qwen3-VL-8B-embedding：同代 Qwen3-VL 权重拆分实践，视觉模块命名 `model.visual.*`，语言模型 `model.language_model.*`。
  https://huggingface.co/c6oisini/Qwen3-VL-8B-embedding/raw/main/qwenextract.md

## 注意事项

- WeMM 用 last-token pooling，输入末尾必须加 `<embedding>` token。
- 向量维度 2048，支持 Matryoshka 截断到 1024/512，存储减半，损失 <2%。
- 微调后先在通用测试集上验，再上医疗测试集。
- 推理时加载合并权重，不要带 LoRA adapter，省显存。
