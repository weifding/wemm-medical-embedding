"""
compare_thin.py
对比瘦身前后模型的嵌入值变化（CPU 即可跑）。

数据: data/eval_medical.jsonl + data/train.sample.jsonl (医疗)
      data/eval_commonsense.jsonl (常识，自生成)

A. 同口径 model.embedding()：原始完整模型 vs 瘦身模型 —— 瘦身本身是否改变了嵌入
B. 旧 encode.py 口径（原始模型 forward 的 logits 池化）vs 新口径（瘦身模型 hidden 池化）
   —— bug 修复前后下游实际拿到的向量差异与检索得分变化

用法: python scripts/compare_thin.py [--full_model PATH] [--thin_model PATH] [--skip_legacy]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FULL_MODEL = "E:/ChatGPT/models/WeMM-Embedding-2B"
THIN_MODEL = os.path.join(ROOT, "outputs", "wemm-text-only")
DATA_FILES = {
    "医疗": [os.path.join(ROOT, "data", "eval_medical.jsonl"),
             os.path.join(ROOT, "data", "train.sample.jsonl")],
    "常识": [os.path.join(ROOT, "data", "eval_commonsense.jsonl")],
}
EMBED_TOKEN = "<embedding>"
DEVICE = "cpu"
DTYPE = torch.bfloat16


def load_items(path):
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def candidates(it):
    return it.get("candidates") or it.get("negatives") or []


def collect_texts(items):
    seen, out = set(), []
    for it in items:
        for t in [it["query"], it["positive"]] + candidates(it):
            if t not in seen:
                seen.add(t)
                out.append(t)
    return out


def _encode_batch(tok, texts):
    batch = [t + EMBED_TOKEN for t in texts]
    return tok(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")


@torch.no_grad()
def embed_hidden(model, tok, texts, batch_size=16):
    """model.embedding() 口径（= 修复后 encode.py 的结果）。"""
    out = []
    for i in range(0, len(texts), batch_size):
        enc = _encode_batch(tok, texts[i:i + batch_size])
        e = model.embedding(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
        out.append(e.float().cpu())
    return torch.cat(out)


@torch.no_grad()
def embed_logits_legacy(model, tok, texts, batch_size=16):
    """旧 encode.py 行径: model(**enc) -> res[0]（logits）按最后位置池化 + L2。"""
    out = []
    for i in range(0, len(texts), batch_size):
        enc = _encode_batch(tok, texts[i:i + batch_size])
        res = model(**enc)
        hidden = res.last_hidden_state if hasattr(res, "last_hidden_state") else res[0]
        lengths = enc["attention_mask"].sum(dim=1) - 1
        v = hidden[torch.arange(hidden.size(0)), lengths]
        out.append(F.normalize(v.float(), dim=-1))
    return torch.cat(out)


def retrieval_scores(items, emb, texts, topk=(1, 5)):
    """对每条 item: 在 [positive] + candidates 里按与 query 的余弦排序，统计 positive 命中。"""
    idx = {t: i for i, t in enumerate(texts)}
    hits = {k: 0 for k in topk}
    ranks = []
    for it in items:
        pool = [it["positive"]] + candidates(it)
        q = emb[idx[it["query"]]]
        scores = F.normalize(emb[[idx[t] for t in pool]], dim=-1) @ q
        order = torch.argsort(scores, descending=True)
        rank = int(order.tolist().index(0)) + 1  # positive 位于 pool[0]
        ranks.append(rank)
        for k in topk:
            if rank <= k:
                hits[k] += 1
    n = len(items)
    return {f"R@{k}": hits[k] / n for k in topk} | {"mean_rank": sum(ranks) / n}


def sim_matrix(emb):
    e = F.normalize(emb, dim=-1)
    return (e @ e.T).numpy()


def upper_tri(s):
    iu = np.triu_indices(s.shape[0], k=1)
    return s[iu]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--full_model", default=FULL_MODEL, help="对比基准（左）模型")
    p.add_argument("--thin_model", default=THIN_MODEL, help="对比目标（右）模型")
    p.add_argument("--skip_legacy", action="store_true",
                   help="跳过 B 段（旧 logits 口径），A 段嵌入漂移+检索仍会算")
    args = p.parse_args()
    full_model_path, thin_model_path = args.full_model, args.thin_model

    items_by_ds = {name: [it for p in paths for it in load_items(p)]
                   for name, paths in DATA_FILES.items()}
    all_items = [it for v in items_by_ds.values() for it in v]
    texts = collect_texts(all_items)
    text_ds = {}
    for name, items in items_by_ds.items():
        for t in collect_texts(items):
            text_ds.setdefault(t, name)
    print(f"文本总数: {len(texts)} (医疗 {sum(1 for t in texts if text_ds[t]=='医疗')}, "
          f"常识 {sum(1 for t in texts if text_ds[t]=='常识')})")
    index = {t: i for i, t in enumerate(texts)}

    # ---------- 基准模型 ----------
    t0 = time.time()
    print(f"\n[1/4] 加载基准模型: {full_model_path}")
    tok_full = AutoTokenizer.from_pretrained(full_model_path, trust_remote_code=True)
    full = AutoModel.from_pretrained(full_model_path, trust_remote_code=True,
                                     torch_dtype=DTYPE, low_cpu_mem_usage=True)
    full.eval()
    print(f"  加载完成 {time.time()-t0:.0f}s")
    t0 = time.time()
    e_full = embed_hidden(full, tok_full, texts)
    e_old = None if args.skip_legacy else embed_logits_legacy(full, tok_full, texts)
    print(f"  基准 embedding 完成 {time.time()-t0:.0f}s"
          + ("" if args.skip_legacy else " (含旧logits口径)"))
    del full
    import gc
    gc.collect()

    # ---------- 对比模型 ----------
    t0 = time.time()
    print(f"[2/4] 加载对比模型: {thin_model_path}")
    tok_thin = AutoTokenizer.from_pretrained(thin_model_path, trust_remote_code=True)
    thin = AutoModel.from_pretrained(thin_model_path, trust_remote_code=True,
                                     torch_dtype=DTYPE, low_cpu_mem_usage=True)
    thin.eval()
    print(f"  加载完成 {time.time()-t0:.0f}s")
    t0 = time.time()
    e_thin = embed_hidden(thin, tok_thin, texts)
    print(f"  对比模型 embedding 完成 {time.time()-t0:.0f}s")

    # ---------- A: 同口径对比 ----------
    print("\n" + "=" * 64)
    print("A. 同口径 model.embedding(): 原始完整模型 vs 瘦身模型")
    print("=" * 64)
    cos = F.cosine_similarity(e_full, e_thin, dim=-1)
    print(f"逐文本余弦(原始,瘦身): mean={cos.mean():.6f}  min={cos.min():.6f}  "
          f"max|1-cos|={(1-cos).abs().max():.2e}")
    d = np.abs(sim_matrix(e_full) - sim_matrix(e_thin))
    print(f"成对相似度矩阵 |Δ|: mean={d.mean():.2e}  max={d.max():.2e}")
    for name, items in items_by_ds.items():
        r_full = retrieval_scores(items, e_full, texts)
        r_thin = retrieval_scores(items, e_thin, texts)
        print(f"  [{name}] R@1 原始={r_full['R@1']:.3f} 瘦身={r_thin['R@1']:.3f} | "
              f"R@5 原始={r_full['R@5']:.3f} 瘦身={r_thin['R@5']:.3f} | "
              f"mean_rank 原始={r_full['mean_rank']:.2f} 瘦身={r_thin['mean_rank']:.2f}")

    # ---------- B: 旧口径 vs 新口径 ----------
    if e_old is not None:
        print("\n" + "=" * 64)
        print("B. 旧 encode.py 口径(logits, 基准模型) vs 新口径(hidden, 对比模型)")
        print("=" * 64)
        print(f"向量维度: 旧={e_old.shape[1]}  新={e_thin.shape[1]}")
        s_old, s_new = sim_matrix(e_old), sim_matrix(e_thin)
        corr = np.corrcoef(upper_tri(s_old), upper_tri(s_new))[0, 1]
        print(f"成对相似度矩阵上三角相关系数: {corr:.4f}")
        for name, items in items_by_ds.items():
            r_old = retrieval_scores(items, e_old, texts)
            r_new = retrieval_scores(items, e_thin, texts)
            print(f"  [{name}] R@1 旧={r_old['R@1']:.3f} 新={r_new['R@1']:.3f} | "
                  f"R@5 旧={r_old['R@5']:.3f} 新={r_new['R@5']:.3f} | "
                  f"mean_rank 旧={r_old['mean_rank']:.2f} 新={r_new['mean_rank']:.2f}")

    # ---------- 明细示例 ----------
    print("\n逐文本余弦明细 (原始 vs 瘦身, 按 |1-cos| 降序前5):")
    order = torch.argsort((1 - cos), descending=True)
    for i in order[:5].tolist():
        print(f"  cos={cos[i]:.6f}  [{text_ds[texts[i]]}] {texts[i][:38]}")


if __name__ == "__main__":
    sys.exit(main())
