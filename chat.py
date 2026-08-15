from __future__ import annotations
import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
import sentencepiece as spm
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


SPECIAL_SYSTEM = "<|system|>"
SPECIAL_USER = "<|user|>"
SPECIAL_ASSISTANT = "<|assistant|>"
SPECIAL_SEP = "<|sep|>"



# Tokenizer
@dataclass
class ModelConfig:
    vocab_size: int
    max_seq_len: int = 128
    d_model: int = 384
    n_heads: int = 6
    n_layers: int = 8
    dropout: float = 0.1
    mlp_ratio: int = 4


class SPTokenizer:
    def __init__(self, model_file: str):
        self.sp = spm.SentencePieceProcessor(model_file=model_file)
        self.vocab_size = self.sp.vocab_size()
        self.unk_id = self.sp.unk_id()
        self.bos_id = self.sp.bos_id()
        self.eos_id = self.sp.eos_id()
        self.pad_id = self.sp.pad_id() if self.sp.pad_id() >= 0 else 0

    def encode(self, text: str) -> List[int]:
        return list(self.sp.encode(text, out_type=int))

    def decode(self, ids: List[int]) -> str:
        return self.sp.decode(ids)

    def piece_to_id(self, piece: str) -> int:
        return int(self.sp.piece_to_id(piece))



# Model
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * scale * self.weight


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model harus habis dibagi n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        qkv = self.qkv(x)
        qkv = qkv.view(b, t, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, H, T, D]

        if hasattr(F, "scaled_dot_product_attention"):
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
            mask = torch.tril(torch.ones(t, t, device=x.device, dtype=torch.bool))
            scores = scores.masked_fill(~mask, float("-inf"))
            probs = F.softmax(scores, dim=-1)
            if self.training and self.dropout > 0:
                probs = F.dropout(probs, p=self.dropout)
            y = torch.matmul(probs, v)

        y = y.transpose(1, 2).contiguous().view(b, t, c)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, d_model: int, mlp_ratio: int, dropout: float):
        super().__init__()
        hidden = d_model * mlp_ratio
        self.fc1 = nn.Linear(d_model, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return self.dropout(x)


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_heads, cfg.dropout)
        self.norm2 = RMSNorm(cfg.d_model)
        self.mlp = MLP(cfg.d_model, cfg.mlp_ratio, cfg.dropout)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop(self.attn(self.norm1(x)))
        x = x + self.drop(self.mlp(self.norm2(x)))
        return x


class TinyGPT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, cfg.max_seq_len, cfg.d_model))
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm_f = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        _, t = idx.shape
        if t > self.cfg.max_seq_len:
            idx = idx[:, -self.cfg.max_seq_len:]
            t = idx.shape[1]

        x = self.tok_emb(idx) + self.pos_emb[:, :t, :]
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )
        return logits, loss



# Utility
def choose_device(force_cpu: bool = False) -> torch.device:
    if not force_cpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def maybe_cast_model(model: nn.Module, device: torch.device, dtype_name: str) -> nn.Module:
    if device.type != "cuda":
        return model

    dtype_name = dtype_name.lower()
    if dtype_name == "float16":
        return model.half()
    if dtype_name == "bfloat16":
        return model.to(dtype=torch.bfloat16)
    return model.float()


def maybe_cast_input(x: torch.Tensor, device: torch.device) -> torch.Tensor:
    # token ids tetap long
    return x.to(device=device, dtype=torch.long)


def load_model_and_tokenizer(
    checkpoint_path: str,
    tokenizer_path: str,
    config_path: Optional[str],
    device: torch.device,
    dtype_name: str,
) -> Tuple[TinyGPT, SPTokenizer, ModelConfig]:
    tokenizer = SPTokenizer(tokenizer_path)

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(ckpt, dict) and "config" in ckpt:
        cfg_dict = ckpt["config"]
    elif config_path:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg_dict = json.load(f)
    else:
        raise ValueError("Config tidak ditemukan. Berikan --config atau gunakan checkpoint yang menyimpan field 'config'.")

    cfg = ModelConfig(**cfg_dict)
    model = TinyGPT(cfg)

    if isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model.to(device)
    model = maybe_cast_model(model, device, dtype_name)
    return model, tokenizer, cfg


def top_k_filter(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    if top_k is None or top_k <= 0:
        return logits
    k = min(top_k, logits.size(-1))
    v, _ = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float("inf")
    return out


def apply_repetition_penalty(logits: torch.Tensor, generated_ids: List[int], penalty: float) -> torch.Tensor:
    if penalty is None or penalty <= 1.0 or not generated_ids:
        return logits
    out = logits.clone()
    unique_ids = set(int(i) for i in generated_ids)
    for token_id in unique_ids:
        out[:, token_id] = out[:, token_id] / penalty
    return out


@torch.no_grad()
def generate_reply_ids(
    model: TinyGPT,
    tokenizer: SPTokenizer,
    prompt_ids: List[int],
    device: torch.device,
    max_new_tokens: int = 128,
    temperature: float = 0.8,
    top_k: int = 40,
    repetition_penalty: float = 1.05,
) -> List[int]:
    sep_id = tokenizer.piece_to_id(SPECIAL_SEP)
    eos_id = tokenizer.eos_id

    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.cfg.max_seq_len:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :].float()
        logits = apply_repetition_penalty(logits, idx[0].tolist(), repetition_penalty)

        if temperature <= 0:
            next_id = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            logits = logits / max(temperature, 1e-5)
            logits = top_k_filter(logits, top_k)
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)

        idx = torch.cat([idx, next_id], dim=1)
        token = int(next_id.item())
        if token == sep_id or token == eos_id:
            break

    return idx[0].tolist()


def format_history_as_prompt(history: List[Tuple[str, str]], system_prompt: str = "") -> str:
    parts: List[str] = []

    if system_prompt.strip():
        parts.append(f"{SPECIAL_SYSTEM}\n{system_prompt.strip()}\n{SPECIAL_SEP}")

    for role, content in history:
        role = role.strip().lower()
        if role not in {"user", "assistant"}:
            continue
        parts.append(f"<|{role}|>\n{content.strip()}\n{SPECIAL_SEP}")

    parts.append(f"{SPECIAL_ASSISTANT}\n")
    return "\n".join(parts).strip()


def extract_last_assistant_text(decoded_text: str) -> str:
    marker = SPECIAL_ASSISTANT
    idx = decoded_text.rfind(marker)
    if idx >= 0:
        decoded_text = decoded_text[idx + len(marker):]

    if SPECIAL_SEP in decoded_text:
        decoded_text = decoded_text.split(SPECIAL_SEP, 1)[0]

    decoded_text = decoded_text.replace("</s>", "").replace("<s>", "")
    decoded_text = decoded_text.strip()

    # rapikan baris berulang kosong
    decoded_text = re.sub(r"\n{3,}", "\n\n", decoded_text)
    return decoded_text.strip()


def answer_once(
    model: TinyGPT,
    tokenizer: SPTokenizer,
    device: torch.device,
    user_prompt: str,
    system_prompt: str,
    history: Optional[List[Tuple[str, str]]] = None,
    max_new_tokens: int = 128,
    temperature: float = 0.8,
    top_k: int = 40,
    repetition_penalty: float = 1.05,
) -> str:
    history = list(history or [])
    history.append(("user", user_prompt))
    prompt_text = format_history_as_prompt(history, system_prompt=system_prompt)
    prompt_ids = tokenizer.encode(prompt_text)

    out_ids = generate_reply_ids(
        model=model,
        tokenizer=tokenizer,
        prompt_ids=prompt_ids,
        device=device,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
    )
    decoded = tokenizer.decode(out_ids)
    reply = extract_last_assistant_text(decoded)
    return reply


def interactive_chat(
    model: TinyGPT,
    tokenizer: SPTokenizer,
    device: torch.device,
    system_prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    repetition_penalty: float,
) -> None:
    print("=" * 72)
    print("Mode chat interaktif aktif.")
    print("Command:")
    print("  /reset   -> hapus history")
    print("  /exit    -> keluar")
    print("  /history -> lihat history singkat")
    print("=" * 72)

    history: List[Tuple[str, str]] = []

    while True:
        try:
            user_text = input("\nAnda : ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nKeluar.")
            break

        if not user_text:
            continue

        lower = user_text.lower()
        if lower in {"/exit", "/quit"}:
            print("Keluar.")
            break
        if lower == "/reset":
            history = []
            print("History dihapus.")
            continue
        if lower == "/history":
            if not history:
                print("History kosong.")
            else:
                for i, (role, content) in enumerate(history[-10:], start=1):
                    print(f"{i:02d}. {role}: {content[:120]}")
            continue

        reply = answer_once(
            model=model,
            tokenizer=tokenizer,
            device=device,
            user_prompt=user_text,
            system_prompt=system_prompt,
            history=history,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
        )
        if not reply:
            reply = "(model tidak menghasilkan teks yang jelas; coba naikkan --temperature atau ulangi prompt)"

        print(f"\nBot  : {reply}")
        history.append(("user", user_text))
        history.append(("assistant", reply))


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Coba model TinyGPT hasil training.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path ke best.pt atau last.pt")
    parser.add_argument("--tokenizer", type=str, required=True, help="Path ke spm.model")
    parser.add_argument("--config", type=str, default="", help="Path ke config.json (opsional jika ada di checkpoint)")
    parser.add_argument("--prompt", type=str, default="", help="Prompt sekali jalan")
    parser.add_argument("--system", type=str, default="Kamu adalah asisten virtual berbahasa Indonesia yang membantu, jelas, dan aman.")
    parser.add_argument("--interactive", action="store_true", help="Mode chat interaktif")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--repetition_penalty", type=float, default=1.05)
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--cpu", action="store_true", help="Paksa jalan di CPU")
    return parser


def main() -> None:
    args = build_argparser().parse_args()

    device = choose_device(force_cpu=args.cpu)
    model, tokenizer, cfg = load_model_and_tokenizer(
        checkpoint_path=args.checkpoint,
        tokenizer_path=args.tokenizer,
        config_path=args.config or None,
        device=device,
        dtype_name=args.dtype,
    )

    print(f"[Info] device      : {device}")
    print(f"[Info] max_seq_len : {cfg.max_seq_len}")
    print(f"[Info] dtype       : {args.dtype if device.type == 'cuda' else 'float32'}")

    if args.interactive:
        interactive_chat(
            model=model,
            tokenizer=tokenizer,
            device=device,
            system_prompt=args.system,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
        )
        return

    prompt = args.prompt.strip()
    if not prompt:
        prompt = "Jelaskan fotosintesis secara singkat."

    reply = answer_once(
        model=model,
        tokenizer=tokenizer,
        device=device,
        user_prompt=prompt,
        system_prompt=args.system,
        history=[],
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
    )
    print("\n=== HASIL ===")
    print(reply)


if __name__ == "__main__":
    main()
