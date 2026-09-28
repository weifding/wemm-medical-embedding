"""
strip_vision.py
加载 WeMM-Embedding-2B，删除视觉塔和 visual projection，保存纯文本权重。

Qwen3.5-VL 系列命名参考（来自 c6oisini/Qwen3-VL-8B-embedding 提取实践）：
  视觉: model.visual.patch_embed.*
        model.visual.blocks.{0-26}.attn.* / mlp.*
        model.visual.merger.*
        model.visual.deepstack_merger_list.*
  语言: model.language_model.embed_tokens.*
        model.language_model.layers.{i}.*
        model.language_model.norm.*
"""
import argparse
import os
import torch
from transformers import AutoModel, AutoTokenizer


# Qwen-VL 系视觉模块关键词（小写匹配）
VISUAL_KEYWORDS = [
    "vision_tower", "visual", "vision_encoder", "image_encoder",
    "vision_model", "vision_backbone", "patch_embed", "merger",
]


def list_top_modules(model):
    """打印顶层模块结构，帮助确认命名。"""
    print("=== 顶层模块 ===")
    for name, child in model.named_children():
        n_params = sum(p.numel() for p in child.parameters())
        print(f"  {name:30s}  {n_params/1e6:8.1f}M params")
        # 二级
        for sub_name, sub in child.named_children():
            n = sum(p.numel() for p in sub.parameters())
            print(f"    {sub_name:28s}  {n/1e6:8.1f}M")


def strip_visual_modules(model):
    """递归删除所有视觉相关子模块。"""
    removed = []

    def _walk(parent, prefix):
        to_delete = []
        for name, child in parent.named_children():
            full = f"{prefix}.{name}" if prefix else name
            low = name.lower()
            if any(k in low for k in VISUAL_KEYWORDS):
                to_delete.append(name)
                removed.append(full)
            else:
                _walk(child, full)
        for name in to_delete:
            delattr(parent, name)

    _walk(model, "")
    return removed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="原始 WeMM-Embedding-2B 权重路径")
    parser.add_argument("--output_path", default="outputs/wemm-text-only")
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--list_only", action="store_true",
                        help="只打印模块结构，不删不存，用来确认命名")
    args = parser.parse_args()

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }

    print(f"[1/3] 加载模型: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=dtype_map[args.dtype],
        low_cpu_mem_usage=True,
    )

    list_top_modules(model)

    if args.list_only:
        print("\n--list_only 模式，退出。确认要删的模块在 VISUAL_KEYWORDS 里能命中后再跑正式。")
        return

    print(f"\n[2/3] 删除视觉模块...")
    removed = strip_visual_modules(model)
    print(f"  已删除 {len(removed)} 个模块:")
    for r in removed:
        print(f"    - {r}")

    # 清理 config 里的视觉配置
    cfg = model.config
    for k in ["vision_config", "vision_tower", "visual", "image_size", "patch_size",
              "vision_tower", "mm_llm_projector", "projector",
              "image_token_id", "video_token_id", "vision_start_token_id", "vision_end_token_id"]:
        if hasattr(cfg, k):
            try:
                delattr(cfg, k)
            except Exception:
                pass

    print(f"\n[3/3] 保存到: {args.output_path}")
    os.makedirs(args.output_path, exist_ok=True)
    model.save_pretrained(args.output_path, safe_serialization=True)
    tokenizer.save_pretrained(args.output_path)

    total = sum(p.numel() for p in model.parameters())
    print(f"完成。纯文本参数量: {total/1e6:.1f}M ({total/1e9:.2f}B)")
    print(f"保存路径: {args.output_path}")


if __name__ == "__main__":
    main()
