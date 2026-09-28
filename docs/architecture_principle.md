# 拆解原理：VLM 模块化双塔结构

## 架构图

Qwen-VL / WeMM 这类模型不是"一个大网络里图文混在一起算"，而是三个物理上独立的模块串起来：

```
┌─────────────┐
│  文本 token  │──► token_embedding ─┐
└─────────────┘                      │
                                     ▼
                              ┌─────────────┐     ┌──────────────┐
                              │  LLM backbone │◄────│  视觉 ViT    │
                              │ (Qwen3.5)    │     │  (~600M)     │
                              │ 所有 Transformer│     └──────┬───────┘
                              │  层都在这     │            │
                              └──────┬───────┘            ▼
                                     │              ┌──────────────┐
                                     │              │ projection /│
                                     │              │ merger 投影层 │
                                     │              └──────────────┘
                                     ▼
                              last-token hidden state
                                     │
                                     ▼
                                 embedding 向量
```

## 数据流决定了能拆

纯文本输入时，前向路径：

```
"高血压用药" + <embedding>
    │
    ▼
token_embedding（查表）
    │
    ▼
Transformer layer 1 → 2 → ... → N
    │
    ▼
<embedding> 位置 hidden state → L2 norm → 向量
```

整条路径上，ViT 和 projection 一次都没被调用。

视觉分支只在输入带图像时才被激活：

```
图像 → ViT → patch tokens → projection → 塞进文本序列当特殊 token → 进同一个 Transformer
```

## 为什么删权重是安全的

1. **权重物理隔离**：视觉参数存在 `model.visual.*` 命名空间下，和 `model.language_model.*` 没有参数共享。删掉前者，后者权重一个数都不变。

2. **前向代码有分支**：`forward()` 里判断 `pixel_values is not None` 才走 ViT。纯文本输入时这个分支根本不进。

3. **Embedding 头在最后一层**：WeMM 取 `<embedding>` token 的最后一层 hidden state。这层 hidden state 只由文本 token 经过所有 Transformer 层计算出来，和视觉无关。

4. **LoRA 挂在注意力层**：`q_proj/v_proj` 是 LLM backbone 内部的线性层，删视觉塔不影响 LoRA 挂载点。

## 拆的时候要小心的两件事

| 坑 | 原因 | 处理 |
|---|---|---|
| config 里残留 `vision_config` | 重新加载时 `from_pretrained` 会按 config 初始化视觉模块，报错或占显存 | 删 config 里 vision 相关字段 |
| `modeling_xxx.py` 里 forward 开头就调 `self.visual` | 即使没图像，代码可能先 `self.visual.dtype` 做类型对齐，属性没了会崩 | 要么改 modeling 代码，要么把 visual 模块留着但置空 |

## Qwen3.5-VL 系权重命名参考

来自 c6oisini/Qwen3-VL-8B-embedding 拆分实践：

视觉编码器：
```
model.visual.patch_embed.*
model.visual.blocks.{0-26}.attn.*
model.visual.blocks.{0-26}.mlp.*
model.visual.merger.*
model.visual.deepstack_merger_list.*
```

语言模型：
```
model.language_model.embed_tokens.weight
model.language_model.layers.{i}.self_attn.{q,k,v,o}_proj.weight
model.language_model.layers.{i}.mlp.{gate,up,down}_proj.weight
model.language_model.norm.weight
```

## 一句话总结

VLM = 独立的视觉编码器 + 独立的文本编码器 + 共享的 Transformer 主干。纯文本推理时视觉编码器是死重，删了不影响文本前向计算，只省显存和加载时间。
