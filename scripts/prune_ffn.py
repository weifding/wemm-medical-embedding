"""
prune_ffn.py
医疗领域校准的 FFN 通道结构化剪枝（Wanda 思路适配 SwiGLU）。

得分:  s_j = ||down_proj[:,j]||_2  ×  RMS_t( silu(gate_j(x)) ⊙ up_j(x) )
校准:  医学教材 + 医疗 query，batch=1 前向（无 padding 污染），得分缓存可复用。
注意:  本机 CPU 上 float32 前向比 bfloat16 快 ~4.5x，校准用 fp32；
       剪枝/保存仍用 bf16（保持产物 dtype 不变）。

用法:
  python scripts/prune_ffn.py --ratio 0.25 --output outputs/wemm-medical-pruned25
  python scripts/prune_ffn.py --ratio 0.50 --output outputs/wemm-medical-pruned50   # 复用得分缓存
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_thin import collect_texts, load_items  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL = os.path.join(ROOT, "outputs", "wemm-text-only")
SCORES_CACHE = os.path.join(ROOT, "outputs", "ffn_scores.npy")
DRIFT_TEXTS_NPY = os.path.join(ROOT, "outputs", "drift_texts.npy")
DRIFT_BASE_NPY = os.path.join(ROOT, "outputs", "drift_base_emb.npy")
BOOK = os.path.join(ROOT, "data", "raw", "pretrain", "medical_book_zh.json")
TRAIN = os.path.join(ROOT, "data", "train.jsonl")
DRIFT_FILES = [
    os.path.join(ROOT, "data", "eval_medical.jsonl"),
    os.path.join(ROOT, "data", "train.sample.jsonl"),
    os.path.join(ROOT, "data", "eval_commonsense.jsonl"),
]
EMBED_TOKEN = "<embedding>"
MAX_LEN = 128
N_LAYERS, INTER = 24, 6144


def load_calib_texts(n_book, n_query, seed=42):
    rng = random.Random(seed)
    with open(BOOK, encoding="utf-8") as f:
        lines = f.readlines()
    picks = rng.sample(lines, min(n_book, len(lines)))
    texts = [json.loads(l)["text"] for l in picks]
    queries = [json.loads(l)["query"] for l in open(TRAIN, encoding="utf-8")]
    texts += rng.sample(queries, min(n_query, len(queries)))
    return texts


def pick_device_dtype():
    """GPU 可用时用 cuda+bf16（本地 CPU 上 fp32 比 bf16 快 ~4.5x）。"""
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    return "cpu", torch.float32


class Calibrator:
    """在 gate_proj/up_proj 输出上挂 hook，累加 SwiGLU 通道流量 m^2。"""

    def __init__(self, model, n_layers=N_LAYERS, inter=INTER):
        self.gate_stash = {}
        self.sq = [torch.zeros(inter, dtype=torch.float64) for _ in range(n_layers)]
        self.tok = [0] * n_layers
        self.handles = []
        for i in range(n_layers):
            mlp = model.model.language_model.layers[i].mlp
            self.handles.append(mlp.gate_proj.register_forward_hook(self._gate(i)))
            self.handles.append(mlp.up_proj.register_forward_hook(self._up(i)))

    def _gate(self, i):
        def hook(module, inputs, output):
            self.gate_stash[i] = output.detach()
        return hook

    def _up(self, i):
        def hook(module, inputs, output):
            g = self.gate_stash.pop(i)
            m = F.silu(g) * output.detach()
            self.sq[i] += (m.double() ** 2).sum(dim=tuple(range(m.ndim - 1))).cpu()
            self.tok[i] += m.numel() // m.shape[-1]
        return hook

    def remove(self):
        for h in self.handles:
            h.remove()

    def scores(self, model):
        out = []
        for i in range(len(self.sq)):
            down_w = model.model.language_model.layers[i].mlp.down_proj.weight.detach().float().cpu()
            wnorm = down_w.norm(dim=0)                      # [I]
            rms = torch.sqrt(self.sq[i] / max(self.tok[i], 1))
            out.append((wnorm * rms).numpy())
        return out


@torch.no_grad()
def embed_texts(model, tok, texts, batch_size=16):
    device = next(model.parameters()).device
    out = []
    for i in range(0, len(texts), batch_size):
        batch = [t + EMBED_TOKEN for t in texts[i:i + batch_size]]
        enc = tok(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
        enc = enc.to(device)
        e = model.embedding(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
        out.append(e.float().cpu())
    return torch.cat(out)


def prune_ffn(model, scores, ratio):
    new_inter = None
    for i in range(N_LAYERS):
        mlp = model.model.language_model.layers[i].mlp
        s = torch.as_tensor(scores[i])
        k = int(round(s.numel() * (1 - ratio)))
        keep = torch.topk(s, k).indices.sort().values
        with torch.no_grad():
            mlp.gate_proj.weight = torch.nn.Parameter(mlp.gate_proj.weight.data[keep].clone())
            mlp.up_proj.weight = torch.nn.Parameter(mlp.up_proj.weight.data[keep].clone())
            mlp.down_proj.weight = torch.nn.Parameter(mlp.down_proj.weight.data[:, keep].clone())
        mlp.intermediate_size = k
        new_inter = k
    model.config.text_config.intermediate_size = new_inter
    try:
        model.config.intermediate_size = new_inter
    except Exception:
        pass
    return new_inter


def calibrate(args):
    """校准: GPU 用 bf16 / CPU 用 fp32，batch=1 前向，得分存 npy。"""
    device, dtype = pick_device_dtype()
    print(f"[1/3] 校准 ({device}/{str(dtype).split('.')[-1]}): 加载 {args.model_path}")
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model_path, trust_remote_code=True,
                                      torch_dtype=dtype, low_cpu_mem_usage=True)
    model.to(device)
    model.eval()
    texts = load_calib_texts(args.calib_n_book, args.calib_n_query)
    print(f"  校准文本 {len(texts)} 条 (max_len={MAX_LEN}, batch=1)")
    cal = Calibrator(model)
    t0 = time.time()
    with torch.no_grad():
        for n, t in enumerate(texts, 1):
            enc = tok(t, truncation=True, max_length=MAX_LEN, return_tensors="pt")
            enc = enc.to(device)
            model.embedding(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
            if n % 50 == 0:
                print(f"  {n}/{len(texts)}  {time.time()-t0:.0f}s", flush=True)
    cal.remove()
    scores = cal.scores(model)
    np.save(args.scores, np.stack(scores))
    print(f"  校准完成 {time.time()-t0:.0f}s -> {args.scores}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default=DEFAULT_MODEL)
    p.add_argument("--ratio", type=float, required=True, choices=[0.25, 0.5])
    p.add_argument("--output", required=True)
    p.add_argument("--scores", default=SCORES_CACHE)
    p.add_argument("--calib_n_book", type=int, default=200)
    p.add_argument("--calib_n_query", type=int, default=100)
    args = p.parse_args()

    # ---- 阶段1: 校准（与 ratio 无关，只跑一次）----
    if os.path.exists(args.scores):
        print(f"[1/3] 复用得分缓存 {args.scores}")
    else:
        calibrate(args)
        import gc
        gc.collect()

    # ---- 阶段2: 加载 → 漂移基线 → 剪枝 ----
    device, dtype = pick_device_dtype()
    print(f"[2/3] 加载 {args.model_path} ({device}/{str(dtype).split('.')[-1]})，ratio={args.ratio} 剪枝")
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model_path, trust_remote_code=True,
                                      torch_dtype=dtype, low_cpu_mem_usage=True)
    model.to(device)
    model.eval()

    # 漂移基线：与剪枝后同 dtype、同模型对象
    drift_texts = collect_texts([it for fp in DRIFT_FILES for it in load_items(fp)])
    np.save(DRIFT_TEXTS_NPY, np.array(drift_texts, dtype=object), allow_pickle=True)
    with torch.no_grad():
        base_emb = embed_texts(model, tok, drift_texts)
    np.save(DRIFT_BASE_NPY, base_emb.numpy())
    base_emb = F.normalize(base_emb, dim=-1)

    scores = [np.asarray(a) for a in np.load(args.scores)]
    new_inter = prune_ffn(model, scores, args.ratio)
    n_params = sum(x.numel() for x in model.parameters())
    print(f"  intermediate_size: {INTER} -> {new_inter}   参数量: {n_params/1e6:.1f}M")

    # ---- 阶段3: 漂移自检（GPU 上完成后转 CPU 保存）----
    print("[3/3] 漂移自检 (剪枝前 vs 后)")
    with torch.no_grad():
        new_emb = embed_texts(model, tok, drift_texts)
    new_emb = F.normalize(new_emb, dim=-1)
    cos = (base_emb * new_emb).sum(dim=-1)
    print(f"  余弦: mean={cos.mean():.6f}  min={cos.min():.6f}")

    model.to("cpu")
    os.makedirs(args.output, exist_ok=True)
    model.save_pretrained(args.output, safe_serialization=True)
    tok.save_pretrained(args.output)
    # transformers 5.8(save端) 不会自动拷贝 trust_remote_code 的建模文件，手动补上
    import shutil
    code_src = os.path.join(args.model_path, "modeling_wemm_embedding.py")
    if os.path.exists(code_src):
        shutil.copyfile(code_src, os.path.join(args.output, "modeling_wemm_embedding.py"))
    size = os.path.getsize(os.path.join(args.output, "model.safetensors"))
    print(f"  已保存 {args.output}  ({size:,} B / {size/1e9:.2f} GB)")


if __name__ == "__main__":
    main()
