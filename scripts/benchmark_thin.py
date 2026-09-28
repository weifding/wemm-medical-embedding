"""
benchmark_thin.py
瘦身模型标准 benchmark：
  1. 模型文件大小 + 加载/推理内存占用 (RSS / peak working set)
  2. 编码吞吐 (texts/s)
  3. 检索指标 recall@1/5/10 + MRR —— 医疗集与常识集 × Matryoshka 维度 {2048,1024,512}

用法: python scripts/benchmark_thin.py
输出: 打印表格, 并写入 outputs/benchmark_thin.json
"""
import argparse
import json
import os
import platform
import sys
import time

import psutil

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from encode import Embedder          # noqa: E402
from eval import eval_retrieval      # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.path.join(ROOT, "outputs", "wemm-text-only")
SAFETENSORS = os.path.join(MODEL, "model.safetensors")
DATASETS = {
    "医疗(eval_medical)": os.path.join(ROOT, "data", "eval_medical.jsonl"),
    "常识(eval_commonsense)": os.path.join(ROOT, "data", "eval_commonsense.jsonl"),
}
DIMS = [None, 1024, 512]   # None = 全量 2048
BENCH_TEXTS = [
    "高血压能不能吃西柚", "发烧38.5度需要吃退烧药吗", "为什么天空是蓝色的",
    "地球绕太阳一圈要多久", "感冒了多喝水能加快康复吗", "飞机为什么能飞起来",
    "空腹血糖7.2正常吗", "冬天哈气为什么会冒白烟",
]
GB = 1024 ** 3


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default=MODEL, help="待测模型目录")
    args = p.parse_args()
    model_path = args.model_path
    safetensors = os.path.join(model_path, "model.safetensors")

    proc = psutil.Process()
    result = {"model": model_path, "platform": platform.platform()}

    # ---- 文件大小 ----
    size = os.path.getsize(safetensors)
    result["file_bytes"] = size
    print(f"模型文件: {size:,} B ({size/GB:.3f} GiB / {size/1e9:.2f} GB)")

    # ---- 加载 ----
    base = proc.memory_info().rss
    t0 = time.time()
    embedder = Embedder(model_path)     # bf16 + low_cpu_mem_usage (encode.py 现状)
    load_s = time.time() - t0
    loaded = proc.memory_info().rss
    print(f"加载: {load_s:.1f}s   RSS {base/GB:.3f} -> {loaded/GB:.3f} GB "
          f"(Δ {(loaded-base)/GB:.3f} GB)")

    # ---- 吞吐 ----
    t0 = time.time()
    _ = embedder.encode(BENCH_TEXTS)
    cold_s = time.time() - t0
    t0 = time.time()
    for _ in range(3):
        _ = embedder.encode(BENCH_TEXTS)
    warm_s = (time.time() - t0) / 3
    thr_cold = len(BENCH_TEXTS) / cold_s
    thr_warm = len(BENCH_TEXTS) / warm_s
    print(f"吞吐(batch=8, 512tok): cold {thr_cold:.2f} texts/s, "
          f"warm {thr_warm:.2f} texts/s")
    result["load_seconds"] = round(load_s, 1)
    result["rss_baseline_bytes"] = base
    result["rss_loaded_bytes"] = loaded
    result["throughput_cold_texts_per_s"] = round(thr_cold, 2)
    result["throughput_warm_texts_per_s"] = round(thr_warm, 2)

    # ---- 检索 benchmark ----
    rows = []
    print("\n数据集 | dim | recall@1 | recall@5 | recall@10 | mrr")
    print("-" * 66)
    for ds_name, ds_path in DATASETS.items():
        samples = [json.loads(l) for l in open(ds_path, encoding="utf-8")]
        for dim in DIMS:
            embedder.matryoshka_dim = dim
            t0 = time.time()
            r = eval_retrieval(embedder, samples)
            dt = time.time() - t0
            label = dim or 2048
            print(f"{ds_name} | {label} | {r['recall@1']:.4f} | {r['recall@5']:.4f} | "
                  f"{r['recall@10']:.4f} | {r['mrr']:.4f}   ({dt:.1f}s)")
            rows.append({"dataset": ds_name, "dim": label, **r,
                         "seconds": round(dt, 1)})
    embedder.matryoshka_dim = None
    result["retrieval"] = rows

    # ---- 内存 ----
    try:
        peak = proc.memory_info().peak_wset          # Windows
    except AttributeError:
        import resource                                # Linux: ru_maxrss 单位 KB
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    after = proc.memory_info().rss
    result["rss_after_bench_bytes"] = after
    result["peak_working_set_bytes"] = peak
    print(f"\n内存: RSS(加载后) {loaded/GB:.3f} GB  RSS(结束) {after/GB:.3f} GB  "
          f"Peak WS {peak/GB:.3f} GB")

    out = os.path.join(ROOT, "outputs",
                       f"benchmark_{os.path.basename(os.path.normpath(model_path))}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"结果已写入 {out}")


if __name__ == "__main__":
    main()
