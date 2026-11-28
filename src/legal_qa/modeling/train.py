"""Training / validation / generation for M1 (scratch) and M2 (pretrained) T5 models.

One training loop serves both systems, so they differ only in the weight initialisation
(`init="scratch"` vs `init="pretrained"`). Everything else (architecture, tokenizer, data, loss,
optimizer family, selection rule) is shared.
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoTokenizer, T5ForConditionalGeneration, get_linear_schedule_with_warmup

from legal_qa.data.preprocessing import MODEL_NAME

SEED = 42


@dataclass
class TrainConfig:
    name: str                      # "m1_scratch" | "m2_finetuned" | ...
    init: str                      # "scratch" | "pretrained"
    lr: float
    batch_size: int = 32
    epochs: int = 10
    eval_every: int = 50           # optimizer steps between validation passes
    warmup_ratio: float = 0.06
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    max_source_len: int = 128
    max_target_len: int = 4
    patience: int = 10             # stop after this many validation passes without improvement
    seed: int = SEED
    model_name: str = MODEL_NAME


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_tokenizer(model_name: str = MODEL_NAME):
    return AutoTokenizer.from_pretrained(model_name)


def build_model(init: str, model_name: str = MODEL_NAME, seed: int = SEED) -> T5ForConditionalGeneration:
    """Same architecture for both systems; only the initial weights differ."""
    if init == "pretrained":
        return T5ForConditionalGeneration.from_pretrained(model_name)
    if init == "scratch":
        set_seed(seed)
        return T5ForConditionalGeneration(AutoConfig.from_pretrained(model_name))  # random init
    raise ValueError(init)


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters())


class QADataset(Dataset):
    """Tokenised (input_text, target_text) records."""

    def __init__(self, records: list[dict], tokenizer, max_source_len: int, max_target_len: int):
        src = tokenizer([r["input_text"] for r in records], max_length=max_source_len, truncation=True)
        tgt = tokenizer([r["target_text"] for r in records], max_length=max_target_len, truncation=True)
        self.src, self.tgt = src["input_ids"], tgt["input_ids"]

    def __len__(self):
        return len(self.src)

    def __getitem__(self, i):
        return self.src[i], self.tgt[i]


def make_collate(pad_id: int):
    def collate(batch):
        ms, mt = max(len(s) for s, _ in batch), max(len(t) for _, t in batch)
        ids = torch.full((len(batch), ms), pad_id, dtype=torch.long)
        att = torch.zeros((len(batch), ms), dtype=torch.long)
        lab = torch.full((len(batch), mt), -100, dtype=torch.long)  # -100 = ignored by the loss
        for i, (s, t) in enumerate(batch):
            ids[i, : len(s)] = torch.tensor(s)
            att[i, : len(s)] = 1
            lab[i, : len(t)] = torch.tensor(t)
        return {"input_ids": ids, "attention_mask": att, "labels": lab}

    return collate


def make_loader(records, tokenizer, cfg: TrainConfig, shuffle: bool, batch_size: int | None = None) -> DataLoader:
    ds = QADataset(records, tokenizer, cfg.max_source_len, cfg.max_target_len)
    g = torch.Generator().manual_seed(cfg.seed)
    return DataLoader(ds, batch_size=batch_size or cfg.batch_size, shuffle=shuffle,
                      collate_fn=make_collate(tokenizer.pad_token_id), generator=g)


@torch.no_grad()
def evaluate_loss(model, loader, device) -> float:
    """Token-level cross-entropy averaged over all target tokens (same quantity as the training loss)."""
    model.eval()
    total, n_tok = 0.0, 0
    for b in loader:
        b = {k: v.to(device) for k, v in b.items()}
        logits = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"], labels=b["labels"]).logits
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)).float(), b["labels"].view(-1), ignore_index=-100, reduction="sum"
        )
        total += loss.item()
        n_tok += (b["labels"] != -100).sum().item()
    return total / n_tok


@torch.no_grad()
def generate(model, tokenizer, texts: list[str], device, max_source_len: int, max_new_tokens: int = 4,
             batch_size: int = 64) -> list[str]:
    """Greedy decoding."""
    model.eval()
    out = []
    for i in range(0, len(texts), batch_size):
        enc = tokenizer(texts[i : i + batch_size], max_length=max_source_len, truncation=True,
                        padding=True, return_tensors="pt").to(device)
        ids = model.generate(**enc, max_new_tokens=max_new_tokens, num_beams=1, do_sample=False)
        out += tokenizer.batch_decode(ids, skip_special_tokens=True)
    return [s.strip() for s in out]


def _accuracy(golds, preds) -> float:
    norm = lambda s: s.strip().lower().rstrip(".")
    return float(np.mean([norm(g) == norm(p) for g, p in zip(golds, preds)]))


def train(cfg: TrainConfig, train_records: list[dict], val_records: list[dict], out_dir: str | Path,
          tokenizer=None, device=None) -> dict:
    """Train, validate every `cfg.eval_every` steps, keep the checkpoint with the lowest validation loss.

    Returns the history dict (also written to `<out_dir>/history.json`); the selected checkpoint is
    saved to `<out_dir>/best`.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = device or get_device()
    tokenizer = tokenizer or load_tokenizer(cfg.model_name)
    set_seed(cfg.seed)

    model = build_model(cfg.init, cfg.model_name, cfg.seed).to(device)
    train_loader = make_loader(train_records, tokenizer, cfg, shuffle=True)
    val_loader = make_loader(val_records, tokenizer, cfg, shuffle=False, batch_size=64)

    steps_per_epoch = math.ceil(len(train_loader.dataset) / cfg.batch_size)
    total_steps = steps_per_epoch * cfg.epochs
    no_decay = ("bias", "layer_norm")
    params = [
        {"params": [p for n, p in model.named_parameters() if not any(x in n for x in no_decay)],
         "weight_decay": cfg.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(x in n for x in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(params, lr=cfg.lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(cfg.warmup_ratio * total_steps), total_steps)
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()  # T5 is unstable in fp16

    val_golds = [r["target_text"] for r in val_records]
    hist = {"config": asdict(cfg), "n_params": count_parameters(model), "n_train": len(train_records),
            "n_val": len(val_records), "steps_per_epoch": steps_per_epoch, "total_steps": total_steps,
            "step": [], "epoch": [], "train_loss": [], "val_loss": [], "val_acc": [],
            "best_step": None, "best_val_loss": float("inf"), "stopped_early": False}
    best_bad, step, run_loss, run_n, t0 = 0, 0, 0.0, 0, time.time()

    def validate():
        nonlocal best_bad
        vl = evaluate_loss(model, val_loader, device)
        va = _accuracy(val_golds, generate(model, tokenizer, [r["input_text"] for r in val_records], device,
                                           cfg.max_source_len, cfg.max_target_len))
        hist["step"].append(step); hist["epoch"].append(step / steps_per_epoch)
        hist["train_loss"].append(run_loss / max(run_n, 1)); hist["val_loss"].append(vl); hist["val_acc"].append(va)
        improved = vl < hist["best_val_loss"]
        if improved:
            hist["best_val_loss"], hist["best_step"], best_bad = vl, step, 0
            model.save_pretrained(out_dir / "best")
            tokenizer.save_pretrained(out_dir / "best")
        else:
            best_bad += 1
        print(f"step {step:5d} ep {step / steps_per_epoch:5.2f} | train {hist['train_loss'][-1]:.4f} | "
              f"val {vl:.4f} | val acc {va:.3f} {'*' if improved else ''} | {time.time() - t0:.0f}s", flush=True)
        model.train()

    model.train()
    done = False
    for epoch in range(cfg.epochs):
        for b in train_loader:
            b = {k: v.to(device) for k, v in b.items()}
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                logits = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"], labels=b["labels"]).logits
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)).float(), b["labels"].view(-1), ignore_index=-100
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
            step += 1
            run_loss += loss.item(); run_n += 1
            if step % cfg.eval_every == 0 or step == total_steps:
                validate()
                run_loss, run_n = 0.0, 0
                if best_bad >= cfg.patience:
                    hist["stopped_early"] = True
                    done = True
                    break
        if done:
            break

    hist["train_seconds"] = time.time() - t0
    (out_dir / "history.json").write_text(json.dumps(hist, indent=2))
    return hist


def load_best(out_dir: str | Path, device=None):
    device = device or get_device()
    model = T5ForConditionalGeneration.from_pretrained(Path(out_dir) / "best").to(device)
    return model, AutoTokenizer.from_pretrained(Path(out_dir) / "best")
