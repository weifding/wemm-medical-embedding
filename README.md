# WeMM-Embedding 医疗领域适配方案

基于 WeMM-Embedding-2B（Qwen3.5 backbone）做纯文本医疗检索。两步：抽视觉塔 + LoRA 微调。

## 目录结构

```
wemm-medical-embedding/
├── README.md
├── requirements.txt
├── config.yaml
├── scripts/
│   ├── strip_vision.py    # 删视觉塔，保存纯文本权重
│   ├── train_lora.py      # LoRA 对比学习微调
│   ├── encode.py          # 推理 encode
│   └── eval.py            # 医疗/通用双线评估
├── data/
│   ├── train.sample.jsonl # 数据样例
│   └── eval_medical.jsonl # 医疗评估集
└── outputs/               # 训练输出
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
