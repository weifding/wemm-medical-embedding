"""
train_lora.py
WeMM-Embedding 医疗领域 LoRA 对比学习微调。
输入: 纯文本权重（strip_vision.py 输出）
输出: LoRA adapter + 合并后权重
"""
import argparse
import json
import os
import math
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModel, AutoTokenizer,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model


EMBED_TOKEN = "<embedding>"


# ---------------- 数据 ----------------
class TripletDataset(Dataset):
    def __init__(self, path):
        self.data = [json.loads(l) for l in open(path, encoding="utf-8")]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]


def collate(batch):
    queries = [b["query"] + EMBED_TOKEN for b in batch]
    positives = [b["positive"] + EMBED_TOKEN for b in batch]
    negatives = []
    for b in batch:
        for n in b.get("negatives", []) or []:
            negatives.append(n + EMBED_TOKEN)
    return queries, positives, negatives


# ---------------- 编码 ----------------
def encode_texts(model, tokenizer, texts, device, max_length=512):
    enc = tokenizer(
        texts, padding=True, truncation=True,
        max_length=max_length, return_tensors="pt",
    ).to(device)
    out = model(**enc)
    # 兼容不同返回：优先 last_hidden_state
    hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
    # last-token pooling
    lengths = enc["attention_mask"].sum(dim=1) - 1
    emb = hidden[torch.arange(hidden.size(0), device=device), lengths]
    return F.normalize(emb, p=2, dim=1)


# ---------------- 主流程 ----------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True, help="strip_vision.py 输出的纯文本权重路径")
    p.add_argument("--train_file", default="data/train.jsonl")
    p.add_argument("--output_dir", default="outputs/lora-adapter")
    p.add_argument("--merged_dir", default="outputs/wemm-medical-merged")
    p.add_argument("--r", type=int, default=16)
    p.add_argument("--alpha", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--temperature", type=float, default=0.02)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[1/5] 加载模型: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to(device)
    model.gradient_checkpointing_enable()

    # ---------------- LoRA ----------------
    print(f"[2/5] 配置 LoRA (r={args.r}, alpha={args.alpha})")
    lora_cfg = LoraConfig(
        r=args.r,
        lora_alpha=args.alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type="FEATURE_EXTRACTION",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # ---------------- 数据 ----------------
    print(f"[3/5] 加载数据: {args.train_file}")
    ds = TripletDataset(args.train_file)
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate, drop_last=True,
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01,
    )
    total_steps = len(loader) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, int(total_steps * args.warmup_ratio), total_steps,
    )

    # ---------------- 训练 ----------------
    print(f"[4/5] 开始训练, total_steps={total_steps}")
    model.train()
    global_step = 0
    for epoch in range(args.epochs):
        for queries, positives, negatives in loader:
            q_emb = encode_texts(model, tokenizer, queries, device, args.max_length)
            p_emb = encode_texts(model, tokenizer, positives, device, args.max_length)

            # in-batch contrastive
            logits = q_emb @ p_emb.T / args.temperature
            labels = torch.arange(len(q_emb), device=device)
            loss = F.cross_entropy(logits, labels)

            # hard negatives
            if negatives:
                n_emb = encode_texts(model, tokenizer, negatives, device, args.max_length)
                all_p = torch.cat([p_emb, n_emb], dim=0)
                logits2 = q_emb @ all_p.T / args.temperature
                loss2 = F.cross_entropy(logits2, labels)
                loss = 0.5 * loss + 0.5 * loss2

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0,
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            global_step += 1
            if global_step % 10 == 0:
                print(f"  epoch={epoch} step={global_step}/{total_steps} loss={loss.item():.4f} lr={scheduler.get_last_lr()[0]:.2e}")

        # 每个 epoch 存一次 adapter
        model.save_pretrained(f"{args.output_dir}/epoch{epoch}")
        print(f"  epoch {epoch} 完成, adapter 已存到 {args.output_dir}/epoch{epoch}")

    # ---------------- 合并 ----------------
    print(f"[5/5] 合并 LoRA 并保存到: {args.merged_dir}")
    merged = model.merge_and_unload()
    os.makedirs(args.merged_dir, exist_ok=True)
    merged.save_pretrained(args.merged_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.merged_dir)
    print("完成。")


if __name__ == "__main__":
    main()
