#!/usr/bin/env python3
"""
Single-file Vanilla SFT launcher (GSM8K) - updated with robust training loop,
proper AMP / GradScaler handling, and forward exception handling.

Usage example (quick small run):
  python3 src/vanilla_sft.py --subset 100 --epochs 1 --batch 1 --grad_accum 4 --lr 1e-5 --fp16 --out checkpoints/sft_test_run

Be careful with full runs (large model + full GSM8K) — they require lots of GPU memory.
"""

import argparse
import json
import os
import math
import random
import time
from datetime import datetime
from pathlib import Path
from contextlib import contextmanager, nullcontext

import torch
from torch import optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import datasets
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.cuda.amp import autocast, GradScaler

# ---------------------
# Minimal stubs / helpers (to avoid external dependencies)
# ---------------------
class MemoryTrace:
    """Stub to mimic memory tracing context manager used by the original train()."""
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc, tb):
        pass
    def print_stats(self):
        pass

@contextmanager
def profile(train_config, local_rank=None):
    """Stub profile context manager used in original code."""
    class P:
        def step(self): pass
        def is_done(self): return False
        def get_flops_per_sec(self): return 0.0
    yield P()

def save_to_json(filename, train_step_loss, train_loss, train_step_perplexity, train_prep,
                 val_step_loss, val_loss, val_step_perplexity, val_prep):
    """Save metrics in a compact JSON for plotting later (append/overwrite)."""
    try:
        out = {
            "train_step_loss": train_step_loss,
            "train_loss": train_loss,
            "train_step_perplexity": train_step_perplexity,
            "train_prep": train_prep,
            "val_step_loss": val_step_loss,
            "val_loss": val_loss,
            "val_step_perplexity": val_step_perplexity,
            "val_prep": val_prep,
            "ts": datetime.now().isoformat(),
        }
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
    except Exception as e:
        print("Warning: save_to_json failed:", e)

# ---------------------
# Data prep: create SFT JSONL from GSM8K train
# ---------------------
def prepare_gsm8k_sft(out_path="data/gsm8k_train_sft.jsonl"):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        print(f"[prepare] {out_path} already exists; skipping creation.")
        return out_path
    print("[prepare] Downloading GSM8K train and creating SFT JSONL...")
    ds = datasets.load_dataset("gsm8k", "main")["train"]
    with open(out_path, "w", encoding="utf-8") as f:
        for i, ex in enumerate(ds):
            q = ex.get("question","").strip()
            # choose the best available rationale/solution field if present
            rationale = ex.get("solution") or ex.get("explanation") or ex.get("answer", "")
            ans = ex.get("answer", "").strip()
            j = {"id": i, "question": q, "rationale": rationale, "answer": ans}
            f.write(json.dumps(j, ensure_ascii=False) + "\n")
    print(f"[prepare] Wrote {len(ds)} examples to {out_path}")
    return out_path

# ---------------------
# Tokenize & mask labels for causal LM
# ---------------------
def tokenize_and_mask(tokenizer, question, rationale, answer, prompt_template="Q: {q}\nA: "):
    # Build prompt and target strings
    prompt_text = prompt_template.format(q=question)
    target_text = (rationale.strip() + "\nFinal answer: " + answer.strip()) if rationale and rationale.strip() else ("Final answer: " + answer.strip())

    # encode without adding special tokens automatically (we control layout)
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    target_ids = tokenizer(target_text, add_special_tokens=False)["input_ids"]

    input_ids = prompt_ids + target_ids
    attention_mask = [1] * len(input_ids)
    labels = [-100] * len(prompt_ids) + target_ids
    return input_ids, attention_mask, labels

# ---------------------
# PyTorch Dataset and collator
# ---------------------
class JsonlCausalDataset(Dataset):
    def __init__(self, path, tokenizer, max_examples=None, prompt_template="Q: {q}\nA: "):
        self.rows = []
        with open(path, "r", encoding="utf-8") as f:
            for ln in f:
                if not ln.strip(): continue
                self.rows.append(json.loads(ln))
        if max_examples and max_examples > 0:
            self.rows = self.rows[:max_examples]
        self.tokenizer = tokenizer
        self.prompt_template = prompt_template

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        ex = self.rows[idx]
        input_ids, attn, labels = tokenize_and_mask(self.tokenizer, ex["question"], ex.get("rationale",""), ex.get("answer",""), self.prompt_template)
        return {"input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attn, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long)}

def collate_causal(batch, pad_token_id):
    max_len = max(x["input_ids"].size(0) for x in batch)
    input_ids = []
    attention_mask = []
    labels = []
    for ex in batch:
        pad = max_len - ex["input_ids"].size(0)
        input_ids.append(torch.cat([ex["input_ids"], torch.full((pad,), pad_token_id, dtype=torch.long)]))
        attention_mask.append(torch.cat([ex["attention_mask"], torch.zeros((pad,), dtype=torch.long)]))
        labels.append(torch.cat([ex["labels"], torch.full((pad,), -100, dtype=torch.long)]))
    return {"input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_mask),
            "labels": torch.stack(labels)}

# ---------------------
# evaluation()
# ---------------------
def evaluation(model, train_config, eval_dataloader, local_rank, tokenizer, wandb_run):
    """
    Evaluate the model on eval_dataloader.
    Returns (eval_ppl, eval_epoch_loss, val_step_loss_list, val_step_perplexity_list)
    """
    model.eval()
    val_step_loss = []
    val_step_perplexity = []
    eval_loss = 0.0

    device = next(model.parameters()).device
    with MemoryTrace() as memtrace:
        with torch.no_grad():
            for step, batch in enumerate(tqdm(eval_dataloader, colour="green", desc="evaluating Epoch", dynamic_ncols=True)):
                # move to device
                for k in batch.keys():
                    batch[k] = batch[k].to(device)
                outputs = model(**batch)
                loss = outputs.loss
                if train_config.save_metrics:
                    val_step_loss.append(loss.detach().float().item())
                    val_step_perplexity.append(float(torch.exp(loss.detach().float())))
                eval_loss += loss.detach().float()
    eval_epoch_loss = eval_loss / max(1, len(eval_dataloader))
    eval_ppl = float(torch.exp(eval_epoch_loss))
    if wandb_run:
        wandb_run.log({'eval/perplexity': eval_ppl, 'eval/loss': float(eval_epoch_loss)}, commit=False)
    return eval_ppl, float(eval_epoch_loss), val_step_loss, val_step_perplexity

# ---------------------
# train() - robust loop with AMP and exception handling
# ---------------------
def train(model, train_dataloader, eval_dataloader, tokenizer, optimizer, lr_scheduler, gradient_accumulation_steps, train_config, wandb_run=None):
    """
    Train loop adapted from your provided code but with robust error handling and AMP/GradScaler usage.
    """
    local_rank = None
    rank = None
    use_fp16 = getattr(train_config, "mixed_precision", False)
    scaler = GradScaler(enabled=use_fp16)

    # metric lists
    train_prep = []
    train_loss = []
    val_prep = []
    val_loss = []

    if train_config.save_metrics:
        metrics_filename = f"{train_config.output_dir}/metrics_data_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
        train_step_perplexity = []
        train_step_loss = []
        val_step_loss = []
        val_step_perplexity = []

    epoch_times = []
    checkpoint_times = []
    results = {}
    best_val_loss = float("inf")
    total_train_steps = 0
    max_steps_reached = False

    device = next(model.parameters()).device

    for epoch in range(train_config.num_epochs):
        if max_steps_reached:
            break
        epoch_start_time = time.perf_counter()
        with MemoryTrace() as memtrace:
            model.train()
            total_loss = 0.0
            pbar = tqdm(colour="blue", desc=f"Training Epoch: {epoch+1}", total=math.ceil(len(train_dataloader)/gradient_accumulation_steps), dynamic_ncols=True)
            with profile(train_config, local_rank) as profile_context:
                optimizer.zero_grad()
                for step, batch in enumerate(train_dataloader):
                    total_train_steps += 1
                    # optional max-train-step control
                    if getattr(train_config, "max_train_step", 0) > 0 and total_train_steps > train_config.max_train_step:
                        max_steps_reached = True
                        print("max training steps reached, stopping training")
                        break

                    # move to device
                    for key in batch.keys():
                        batch[key] = batch[key].to(device)

                    # Forward (with try/except so a single bad batch won't crash everything)
                    try:
                        # with autocast(device_type="cuda", enabled=use_fp16):
                        with autocast(enabled=use_fp16):
                            outputs = model(**batch)
                            loss_raw = outputs.loss
                    except Exception as e:
                        # log and save bad batch for inspection, then skip
                        print(f"[WARN] Exception during forward at step {step}: {e}", flush=True)
                        try:
                            bad_path = os.path.join(train_config.output_dir, f"bad_batch_step{step}.pt")
                            torch.save({k: v.detach().cpu() for k, v in batch.items()}, bad_path)
                            print(f"[WARN] Saved bad batch to {bad_path}", flush=True)
                        except Exception as ee:
                            print(f"[WARN] Failed to save bad batch: {ee}", flush=True)
                        optimizer.zero_grad()
                        continue

                    # scale by grad accumulation
                    loss = loss_raw / gradient_accumulation_steps

                    # detect NaN / Inf
                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"[WARN] loss is NaN or Inf at step {step}, skipping batch", flush=True)
                        optimizer.zero_grad()
                        continue

                    # logging metrics
                    if train_config.save_metrics:
                        train_step_loss.append(loss.detach().float().item())
                        # perplexity uses exp(loss) on the un-accumulated loss approx
                        try:
                            train_step_perplexity.append(float(torch.exp(loss.detach().float())))
                        except Exception:
                            train_step_perplexity.append(float("nan"))

                    total_loss += loss_raw.detach().float()

                    # Backward (with scaler if fp16)
                    if use_fp16:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    # Step if accumulation condition met
                    if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                        # unscale before clipping if fp16
                        if use_fp16:
                            try:
                                scaler.unscale_(optimizer)
                            except Exception as e:
                                print(f"[WARN] scaler.unscale_() failed: {e}", flush=True)
                        # gradient clipping
                        if train_config.gradient_clipping and train_config.gradient_clipping_threshold > 0.0:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.gradient_clipping_threshold)

                        # optimizer step
                        if use_fp16:
                            try:
                                scaler.step(optimizer)
                                scaler.update()
                            except Exception as e:
                                print(f"[WARN] scaler.step failed: {e}", flush=True)
                                optimizer.zero_grad()
                                continue
                        else:
                            optimizer.step()

                        optimizer.zero_grad()
                        pbar.update(1)

                    # optional profiling / logging
                    if train_config.use_profiler or train_config.flop_counter:
                        profile_context.step()

                    if wandb_run:
                        wandb_run.log({
                            'train/epoch': epoch + 1,
                            'train/step': epoch * len(train_dataloader) + step,
                            'train/loss': loss.detach().float(),
                        })

                    # update tqdm
                    try:
                        cur_loss = float(loss.detach().float())
                        pbar.set_description(f"Training Epoch: {epoch+1}/{train_config.num_epochs} step {step}/{len(train_dataloader)} (loss: {cur_loss:.4f})")
                    except Exception:
                        pbar.set_description(f"Training Epoch: {epoch+1}/{train_config.num_epochs} step {step}/{len(train_dataloader)}")

                    # periodic validation
                    if step % 50 == 0 and train_config.run_validation and eval_dataloader is not None:
                        eval_ppl, eval_epoch_loss, temp_val_loss, temp_step_perplexity = evaluation(model, train_config, eval_dataloader, local_rank, tokenizer, wandb_run)
                        if train_config.save_metrics:
                            val_step_loss.extend(temp_val_loss)
                            val_step_perplexity.extend(temp_step_perplexity)
                        model.train()

                    # save metrics occasionally
                    if train_config.save_metrics:
                        save_to_json(metrics_filename, train_step_loss, train_loss, train_step_perplexity, train_prep, val_step_loss, val_loss, val_step_perplexity, val_prep)

                pbar.close()

        epoch_end_time = time.perf_counter() - epoch_start_time
        epoch_times.append(epoch_end_time)
        # compute epoch stats
        train_epoch_loss = total_loss / max(1, len(train_dataloader))
        try:
            train_perplexity = float(torch.exp(train_epoch_loss))
        except Exception:
            train_perplexity = float("nan")
        train_prep.append(float(train_perplexity))
        train_loss.append(float(train_epoch_loss))
        memtrace.print_stats()
        try:
            lr_scheduler.step()
        except Exception:
            pass

        # run validation at epoch end
        if train_config.run_validation and eval_dataloader is not None:
            eval_ppl, eval_epoch_loss, temp_val_loss, temp_step_perplexity = evaluation(model, train_config, eval_dataloader, local_rank, tokenizer, wandb_run)
            if train_config.save_metrics:
                val_step_loss.extend(temp_val_loss)
                val_step_perplexity.extend(temp_step_perplexity)

            # save checkpoint
            if train_config.save_model:
                epoch_dir = os.path.join(train_config.output_dir, f"epoch{epoch+1}")
                os.makedirs(epoch_dir, exist_ok=True)
                # Save the model
                model.save_pretrained(epoch_dir)
                print(f"Model saved in {epoch_dir}")

            if eval_epoch_loss < best_val_loss:
                best_val_loss = eval_epoch_loss
                print(f"best eval loss on epoch {epoch+1} is {best_val_loss:.4f}")
            val_loss.append(float(best_val_loss))
            val_prep.append(float(eval_ppl))

        print(f"Epoch {epoch+1}: train_perplexity={train_perplexity:.4f}, train_epoch_loss={train_epoch_loss:.4f}, epoch time {epoch_end_time:.1f}s")

        # save metrics
        if train_config.save_metrics:
            save_to_json(metrics_filename, train_step_loss, train_loss, train_step_perplexity, train_prep, val_step_loss, val_loss, val_step_perplexity, val_prep)

    # aggregate results
    avg_epoch_time = sum(epoch_times) / max(1, len(epoch_times))
    avg_checkpoint_time = sum(checkpoint_times) / max(1, len(checkpoint_times)) if checkpoint_times else 0
    results["avg_train_prep"] = sum(train_prep)/len(train_prep) if train_prep else 0.0
    results["avg_train_loss"] = sum(train_loss)/len(train_loss) if train_loss else 0.0
    if train_config.run_validation:
        results["avg_eval_prep"] = sum(val_prep)/len(val_prep) if val_prep else 0.0
        results["avg_eval_loss"] = sum(val_loss)/len(val_loss) if val_loss else 0.0
    results["avg_epoch_time"] = avg_epoch_time
    results["avg_checkpoint_time"] = avg_checkpoint_time
    if train_config.save_metrics:
        results["metrics_filename"] = metrics_filename
    return results

# ---------------------
# Launcher main
# ---------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct", help="HF model id or local path")
    ap.add_argument("--train_file", default="data/gsm8k_train_sft.jsonl", help="SFT training JSONL")
    ap.add_argument("--subset", type=int, default=500, help="use first N examples (0 = full)")
    ap.add_argument("--out", default="checkpoints/vanilla_sft_direct", help="output dir for checkpoints")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max_seq_len", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fp16", action="store_true", help="use fp16 mixed precision for training")
    args = ap.parse_args()

    # prepare data
    prepare_gsm8k_sft(args.train_file)

    # load tokenizer + model
    print("[launcher] Loading tokenizer + model:", args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_auth_token=True)
    if getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token = tokenizer.eos_token

    # load model as float32 (we use autocast for mixed precision), this avoids some issues
    dtype = torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        torch_dtype=dtype,
        low_cpu_mem_usage=True
    )

    # enable gradient checkpointing to reduce memory
    model.gradient_checkpointing_enable()
    print("[launcher] Model loaded. sample param device:", next(model.parameters()).device)

    # build datasets
    subset = args.subset if args.subset > 0 else None
    ds = JsonlCausalDataset(args.train_file, tokenizer, max_examples=subset)
    # small validation split: last 10% of ds
    n = len(ds)
    n_val = max(1, int(0.1 * n)) if n>1 else 0
    if n_val:
        train_rows = ds.rows[:-n_val]
        val_rows = ds.rows[-n_val:]
        # write temporary jsonls for simplicity
        tmp_train = os.path.join(args.out, "tmp_train.jsonl")
        tmp_val = os.path.join(args.out, "tmp_val.jsonl")
        os.makedirs(args.out, exist_ok=True)
        with open(tmp_train, "w", encoding="utf-8") as f:
            for r in train_rows: f.write(json.dumps(r, ensure_ascii=False)+"\n")
        with open(tmp_val, "w", encoding="utf-8") as f:
            for r in val_rows: f.write(json.dumps(r, ensure_ascii=False)+"\n")
        train_dataset = JsonlCausalDataset(tmp_train, tokenizer, max_examples=None)
        val_dataset = JsonlCausalDataset(tmp_val, tokenizer, max_examples=None)
    else:
        train_dataset = ds
        val_dataset = None

    # dataloaders
    collator = lambda batch: collate_causal(batch, pad_token_id=tokenizer.pad_token_id)
    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, collate_fn=collator, num_workers=2)
    eval_loader = DataLoader(val_dataset, batch_size=args.batch, shuffle=False, collate_fn=collator, num_workers=2) if val_dataset else None

    print(f"[launcher] train size={len(train_dataset)}, val size={len(val_dataset) if val_dataset else 0}")

    # optimizer & scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.85)

    # train_config simple object to pass to train()
    class Cfg:
        pass
    cfg = Cfg()
    cfg.num_epochs = args.epochs
    cfg.gradient_accumulation_steps = args.grad_accum
    cfg.gradient_clipping = True
    cfg.gradient_clipping_threshold = 1.0
    cfg.run_validation = True if eval_loader else False
    cfg.save_model = True
    cfg.save_metrics = True
    cfg.output_dir = args.out
    cfg.max_train_step = 0
    cfg.max_eval_step = 0
    cfg.use_profiler = False
    cfg.flop_counter = False
    cfg.batching_strategy = "padding"
    cfg.context_length = args.max_seq_len
    cfg.num_workers_dataloader = 2
    cfg.per_device_train_batch_size = args.batch
    cfg.lr = args.lr
    cfg.seed = args.seed
    cfg.mixed_precision = args.fp16

    # seed
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)

    # call train()
    print("[launcher] Starting training. Output dir:", cfg.output_dir)
    os.makedirs(cfg.output_dir, exist_ok=True)
    results = train(model, train_loader, eval_loader, tokenizer, optimizer, scheduler, cfg.gradient_accumulation_steps, cfg, None)

    print("[launcher] Training finished. Summary results:")
    for k,v in results.items():
        print(f"  {k}: {v}")
    print("Done.")

if __name__ == "__main__":
    main()
