"""
encode.py
加载（合并后的）WeMM 医疗 embedding 模型，批量编码文本。

用法:
    from encode import Embedder
    emb = Embedder("outputs/wemm-medical-merged")
    vecs = emb.encode(["高血压用药", "感冒怎么办"])
"""
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


EMBED_TOKEN = "<embedding>"


class Embedder:
    def __init__(self, model_path, device=None, dtype=None, matryoshka_dim=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype or torch.bfloat16
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
        ).to(self.device)
        self.model.eval()
        self.matryoshka_dim = matryoshka_dim  # 2048/1024/512

    @torch.no_grad()
    def encode(self, texts, batch_size=32, max_length=512):
        if isinstance(texts, str):
            texts = [texts]
        out = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch = [t + EMBED_TOKEN for t in batch]
            enc = self.tokenizer(
                batch, padding=True, truncation=True,
                max_length=max_length, return_tensors="pt",
            ).to(self.device)
            res = self.model(**enc)
            hidden = res.last_hidden_state if hasattr(res, "last_hidden_state") else res[0]
            lengths = enc["attention_mask"].sum(dim=1) - 1
            emb = hidden[torch.arange(hidden.size(0), device=self.device), lengths]
            emb = F.normalize(emb, p=2, dim=1)
            if self.matryoshka_dim:
                emb = emb[:, :self.matryoshka_dim]
                emb = F.normalize(emb, p=2, dim=1)
            out.append(emb.cpu().float().numpy())
        import numpy as np
        return np.concatenate(out, axis=0)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--text", nargs="+", default=["高血压用药咨询", "感冒发烧怎么办"])
    p.add_argument("--matryoshka_dim", type=int, default=None)
    args = p.parse_args()

    emb = Embedder(args.model_path, matryoshka_dim=args.matryoshka_dim)
    vecs = emb.encode(args.text)
    print(f"输出 shape: {vecs.shape}")
    # 相似度
    sim = vecs @ vecs.T
    print("两两相似度:")
    for i, t in enumerate(args.text):
        print(f"  {t}")
        for j, t2 in enumerate(args.text):
            if i != j:
                print(f"    vs {t2}: {sim[i, j]:.4f}")
