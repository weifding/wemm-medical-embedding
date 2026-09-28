# FFN 领域剪枝记录（Round 3）

日期：2026-09-28 · 执行环境：远程 RTX 4090 服务器（24GB GPU + 32 核 CPU；本地 CPU 太慢，占用 GPU 的常驻推理服务经确认后已停止）
代码：`scripts/prune_ffn.py` · 得分缓存：`outputs/ffn_scores.npy`

## 方法

- 通道得分（Wanda 思路适配 SwiGLU，在医疗语料上校准）：
  `s_j = ||down_proj[:,j]||₂ × RMS_t( silu(gate_j(x)) ⊙ up_j(x) )`
- 校准数据：`data/raw/pretrain/medical_book_zh.json` 随机 200 段 + `data/train.jsonl` 医疗 query 100 条
  （seed=42，max_len=128，batch=1 无 padding 污染；GPU bf16 校准 33s / 300 条）
- 手术：每层保留得分 top-k 通道，切 `gate_proj/up_proj` 行、`down_proj` 列，改 `text_config.intermediate_size`
- 25% 与 50% 两档从**同一份基线得分**生成（不迭代）

## 阶梯结果

| 档位 | 文件大小 | 参数量 | 漂移 mean/min | 医疗 R@1 | 常识 R@1 (2048/1024/512) | 常识 MRR |
|---|---|---|---|---|---|---|
| 基线（瘦身产物） | 3.76 GB | 1881.3M | 1.0 / 1.0 | 1.0（全维度） | 0.875 / 0.875 / 0.875 | 0.906 |
| **剪枝 25%**（6144→4608） | **3.31 GB** | 1654.8M | 0.941 / 0.862 | **1.0（全维度）** | **0.75 / 0.875 / 0.875** | 0.844 / 0.906 / 0.906 |
| **剪枝 50%**（6144→3072） | **2.86 GB** | 1428.3M | 0.853 / 0.712 | **1.0（全维度）** | 0.75 / 0.75 / 0.75 | 0.81–0.83 |

- 漂移 = 103 条基准文本上剪枝前后逐文本余弦（`embedding()` 口径）
- 全部档位 R@5/R@10 = 1.0（positive 始终在前 5）
- 基线为本地 CPU bf16 测得；剪枝档为 4090 GPU 测得（吞吐 warm ≈ 90 texts/s，加载 RSS 1.09GB + VRAM 模型）

## 过闸判定与采纳

| 标准 | 25% | 50% |
|---|---|---|
| 医疗 R@1 保持 1.0 | ✅ | ✅ |
| 常识 R@1 ≥ 0.875 | ⚠️ 2048 维 0.75（1024/512 维 0.875 达标）→ 需人工裁决 | ❌ 全维度 0.75 → 不达标 |

**决定（2026-09-28）：采纳剪枝 25% 档**（用户裁决：医疗无损、1024/512 维常识持平基线）。
- 本地已回传：`outputs/wemm-medical-pruned25/`（MD5 与服务器一致：`4be2307717a3d4df42de18c8534520c8`）
- `config.yaml` 的 `model_path` 已指向该目录
- 50% 档留在服务器作为实验产物

注意：常识集仅 8 题，掉 1 题 = −12.5pp，统计力弱；医疗集仅 4 组。R@10 全部无损。

## 产物位置

- 服务器：`<服务器工作目录>/outputs/wemm-medical-pruned{25,50}/`（25% 档已回传本地）
- 本地（已回传）：`outputs/benchmark_wemm-medical-pruned{25,50}.json`、`outputs/ffn_scores.npy`
- 模型回传：链路 ~3.2MB/s（3.76GB 用时 20min），如需本地使用按需回传单档

## 环境备注

- 服务器 transformers **5.8.0**（save 不自动拷贝 trust_remote_code 建模文件 → `prune_ffn.py` 已加手动 copy）
- 本地 Windows：fp32 前向比 bf16 快 ~4.5×（无原生 BF16）；GPU 上则用 bf16
- 执行前已停止服务器上的 llama-server（Qwen3.8-27B API 服务，PID 3571116），重启命令见会话记录
