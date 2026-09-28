"""
eval.py
双线评估：
1. 医疗检索 recall@k（用问答对，每个 query 的 positive 在候选池中排第几）
2. 通用语义相似度 spearman（STS-b 类数据）

用法:
    python eval.py --model_path outputs/wemm-medical-merged \
                   --eval_file data/eval_medical.jsonl
"""
import argparse
import json
import numpy as np
from encode import Embedder


def recall_at_k(ranks, k):
    return sum(1 for r in ranks if r < k) / len(ranks)


def mrr(ranks):
    return sum(1.0 / (r + 1) for r in ranks) / len(ranks)


def eval_retrieval(embedder, samples):
    """samples: list of {query, positive, candidates:[...]}
    每个 query 在 candidates 中检索 positive，看 positive 排第几。"""
    queries = [s["query"] for s in samples]
    candidates = []
    for s in samples:
        candidates.append(s["positive"])
        candidates.extend(s.get("candidates", []))

    q_emb = embedder.encode(queries)
    c_emb = embedder.encode(candidates)

    # 每个 query 的候选池：[positive] + candidates
    ranks = []
    idx = 0
    for s in samples:
        pool_size = 1 + len(s.get("candidates", []))
        pool = c_emb[idx:idx + pool_size]          # [pool_size, D]
        sim = q_emb[len(ranks)] @ pool.T            # [pool_size]
        order = np.argsort(-sim)
        rank = int(np.where(order == 0)[0][0])      # positive 在 index 0
        ranks.append(rank)
        idx += pool_size

    return {
        "recall@1": recall_at_k(ranks, 1),
        "recall@5": recall_at_k(ranks, 5),
        "recall@10": recall_at_k(ranks, 10),
        "mrr": mrr(ranks),
    }


def eval_sts(embedder, pairs):
    """pairs: [{text1, text2, score: 0-5}]
    算 cosine 相似度，和人工打分算 spearman。"""
    from scipy.stats import spearmanr
    t1 = [p["text1"] for p in pairs]
    t2 = [p["text2"] for p in pairs]
    e1 = embedder.encode(t1)
    e2 = embedder.encode(t2)
    sim = (e1 * e2).sum(axis=1)
    gold = [p["score"] for p in pairs]
    corr, _ = spearmanr(sim, gold)
    return {"sts_spearman": corr}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--eval_file", default="data/eval_medical.jsonl",
                   help="医疗检索评估 jsonl")
    p.add_argument("--sts_file", default=None,
                   help="通用 sts 评估 jsonl（可选）")
    p.add_argument("--matryoshka_dim", type=int, default=None)
    args = p.parse_args()

    embedder = Embedder(args.model_path, matryoshka_dim=args.matryoshka_dim)

    # 医疗检索
    samples = [json.loads(l) for l in open(args.eval_file, encoding="utf-8")]
    print(f"医疗检索评估: {len(samples)} 条")
    r = eval_retrieval(embedder, samples)
    for k, v in r.items():
        print(f"  {k}: {v:.4f}")

    # 通用 STS
    if args.sts_file:
        pairs = [json.loads(l) for l in open(args.sts_file, encoding="utf-8")]
        print(f"通用 STS 评估: {len(pairs)} 对")
        s = eval_sts(embedder, pairs)
        for k, v in s.items():
            print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
