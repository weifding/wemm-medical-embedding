"""从公开中文语料清洗采样，生成 data/train.jsonl。

输出格式（与 data/train.sample.jsonl 一致）：
    {"query": "...", "positive": "...", "negatives": []}
negatives 留空，训练脚本走 in-batch negative。

数据来源：
  - 医疗 60%：shibing624/medical finetune/train_zh_0.json（医患问答）
  - 通用 40%：C-MTEB/ATEC  score=1 的同义问对（金融客服，通用 STS 域）
"""
import json
import random
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MED_SRC = ROOT / "data/raw/finetune/train_zh_0.json"
GEN_SRC = ROOT / "data/raw/data/train-00000-of-00001-fe0441739a89f7a1.parquet"
DST = ROOT / "data/train.jsonl"

MEDICAL_N = 3000
GENERAL_N = 2000
SEED = 42
MIN_Q, MAX_Q = 5, 64
MIN_P, MAX_P = 20, 500

_ctrl = re.compile(r"[\x00-\x1f\x7f]")
_ws = re.compile(r"\s+")


def clean(s: str) -> str:
    s = _ctrl.sub(" ", s)
    s = _ws.sub(" ", s).strip()
    return s


def build_medical() -> list[dict]:
    seen = set()
    pool = []
    with MED_SRC.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            q = clean(row.get("instruction", ""))
            inp = clean(row.get("input", ""))
            a = clean(row.get("output", ""))
            if inp:
                continue
            if not (MIN_Q <= len(q) <= MAX_Q):
                continue
            if not (MIN_P <= len(a) <= MAX_P):
                continue
            if q in seen:
                continue
            seen.add(q)
            pool.append({"query": q, "positive": a, "negatives": []})
    print(f"medical pool: {len(pool)}")
    random.seed(SEED)
    random.shuffle(pool)
    return pool[:MEDICAL_N]


def build_general() -> list[dict]:
    df = pd.read_parquet(GEN_SRC)
    pos = df[df["score"] == 1]
    seen = set()
    pool = []
    for _, r in pos.iterrows():
        q = clean(str(r["sentence1"]))
        a = clean(str(r["sentence2"]))
        # ATEC 是短句同义对，放宽 positive 下限到 4 字
        if not (MIN_Q <= len(q) <= MAX_Q):
            continue
        if not (4 <= len(a) <= MAX_Q):
            continue
        key = (q, a)
        if key in seen:
            continue
        seen.add(key)
        pool.append({"query": q, "positive": a, "negatives": []})
    print(f"general pool: {len(pool)}")
    random.seed(SEED)
    random.shuffle(pool)
    return pool[:GENERAL_N]


def main() -> None:
    med = build_medical()
    gen = build_general()
    rows = med + gen
    random.seed(SEED)
    random.shuffle(rows)

    DST.parent.mkdir(parents=True, exist_ok=True)
    with DST.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"medical={len(med)} general={len(gen)} total={len(rows)} -> {DST}")


if __name__ == "__main__":
    main()
