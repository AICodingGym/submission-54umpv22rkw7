"""LoRA essay scoring with inner epoch selection and a separate outer holdout."""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from sklearn.metrics import cohen_kappa_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

from baseline import integer_scores
from evaluate_qwen import SYSTEM


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU access required")
    torch.set_num_threads(8)
    torch.manual_seed(42)
    np.random.seed(42)
    root = Path(__file__).resolve().parent
    output = root / ("outputs_lora_smoke" if args.smoke else "outputs_lora")
    output.mkdir(exist_ok=True)
    train = pd.read_csv(root / "train.csv", dtype={"essay_id": str})
    test = pd.read_csv(root / "test.csv", dtype={"essay_id": str})
    sample = pd.read_csv(root / "sample_submission.csv", dtype={"essay_id": str})
    fit, outer = train_test_split(train, test_size=0.2, random_state=42, stratify=train.score)
    fit, inner = train_test_split(fit, test_size=0.1, random_state=45, stratify=fit.score)
    if args.smoke:
        fit, inner, outer, test = fit.iloc[:16], inner.iloc[:8], outer.iloc[:8], test.iloc[:8]
        sample = sample[sample.essay_id.isin(test.essay_id)]
    for name, frame in [("fit", fit), ("inner", inner), ("outer", outer)]:
        frame[["essay_id", "score"]].to_csv(output / f"{name}_ids.csv", index=False)
    snapshot = Path((root / "models/qwen_snapshot.txt").read_text().strip())
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    digits = [tokenizer.encode(str(i), add_special_tokens=False) for i in range(1, 7)]
    if any(len(x) != 1 for x in digits):
        raise ValueError("Expected single-token score digits")
    digit_ids = torch.tensor([x[0] for x in digits], device="cuda")
    lengths = {}

    def encode(frame, name):
        encoded, original_lengths = [], []
        for text in frame.full_text:
            prompt = tokenizer.apply_chat_template([
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": "Evaluate this essay:\n<essay>\n" + text + "\n</essay>"},
            ], tokenize=False, add_generation_prompt=True, enable_thinking=False)
            ids = tokenizer.encode(prompt, add_special_tokens=False)
            original_lengths.append(len(ids))
            if len(ids) > args.max_length:
                # Preserve instructions/start and the conclusion/answer position.
                head = args.max_length // 2
                ids = ids[:head] + ids[-(args.max_length - head):]
            encoded.append(ids)
        lengths[name] = {"rows": len(encoded), "truncated": sum(n > args.max_length for n in original_lengths),
                         "mean_tokens": float(np.mean(original_lengths)), "max_tokens": max(original_lengths)}
        return encoded

    fit_tokens = encode(fit, "fit")
    inner_tokens = encode(inner, "inner")
    outer_tokens = encode(outer, "outer")
    test_tokens = encode(test, "test")
    config = {**vars(args), "model_revision": snapshot.name, "gpu": torch.cuda.get_device_name(),
              "lengths": lengths, "seed": 42, "inner_seed": 45,
              "loss": "cross entropy over the six score-digit logits", "lora_rank": 8}
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(json.dumps(config, indent=2), flush=True)

    def load_base():
        return AutoModelForCausalLM.from_pretrained(snapshot, local_files_only=True,
            dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": "cuda"})

    model = get_peft_model(load_base(), LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05, target_modules=["q_proj", "v_proj"],
        bias="none", task_type="CAUSAL_LM",
    ))
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.print_trainable_parameters()

    def batch_inputs(tokens, indices):
        return tokenizer.pad({"input_ids": [tokens[i] for i in indices]}, padding=True,
                             return_tensors="pt").to("cuda")

    def predict(tokens):
        model.eval()
        values = np.zeros(len(tokens))
        # Sort by length to reduce padding; scatter back to original order.
        order = sorted(range(len(tokens)), key=lambda i: len(tokens[i]))
        with torch.inference_mode():
            for start in range(0, len(order), args.batch_size):
                indices = order[start:start + args.batch_size]
                logits = model(**batch_inputs(tokens, indices), use_cache=False, logits_to_keep=1).logits[:, -1, digit_ids].float()
                values[indices] = (logits.softmax(-1) @ torch.arange(1, 7, device="cuda", dtype=torch.float32)).cpu().numpy()
        return values

    batches = math.ceil(len(fit) / args.batch_size)
    steps_per_epoch = math.ceil(batches / args.accumulation)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                 lr=args.learning_rate, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(optimizer, max(1, steps_per_epoch // 10),
                                               steps_per_epoch * args.epochs)
    labels = torch.tensor(fit.score.to_numpy() - 1, device="cuda", dtype=torch.long)
    history, best_qwk = [], -float("inf")
    start_time = time.monotonic()
    for epoch in range(args.epochs):
        model.train()
        order = np.random.default_rng(42 + epoch).permutation(len(fit))
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for batch in range(batches):
            indices = order[batch * args.batch_size:(batch + 1) * args.batch_size].tolist()
            logits = model(**batch_inputs(fit_tokens, indices), use_cache=False, logits_to_keep=1).logits[:, -1, digit_ids].float()
            loss = torch.nn.functional.cross_entropy(logits, labels[indices])
            window = min(args.accumulation, batches - (batch // args.accumulation) * args.accumulation)
            (loss / window).backward()
            losses.append(float(loss.detach()))
            if (batch + 1) % args.accumulation == 0 or batch + 1 == batches:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if batch == 0 or (batch + 1) % 100 == 0:
                progress = {"epoch": epoch + 1, "batch": batch + 1, "batches": batches,
                            "loss_recent": float(np.mean(losses[-100:])),
                            "elapsed_seconds": time.monotonic() - start_time,
                            "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30}
                (output / "progress.json").write_text(json.dumps(progress, indent=2) + "\n")
                print(json.dumps(progress), flush=True)
        raw = predict(inner_tokens)
        qwk = float(cohen_kappa_score(inner.score, integer_scores(raw), weights="quadratic"))
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "inner_qwk": qwk})
        print(json.dumps(history[-1]), flush=True)
        model.save_pretrained(output / f"epoch_{epoch + 1}")
        if qwk > best_qwk:
            best_qwk, best_epoch = qwk, epoch + 1
            model.save_pretrained(output / "adapter")
        (output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    del optimizer, scheduler, model
    torch.cuda.empty_cache()
    model = PeftModel.from_pretrained(load_base(), output / "adapter").eval()
    raw = predict(outer_tokens)
    predictions = integer_scores(raw)
    metrics = {"validation_qwk": float(cohen_kappa_score(outer.score, predictions, weights="quadratic")),
               "validation_mae": float(mean_absolute_error(outer.score, predictions)),
               "selected_epoch": best_epoch, "inner_qwk": best_qwk,
               "elapsed_seconds": time.monotonic() - start_time,
               "fit_rows": len(fit), "inner_rows": len(inner), "outer_rows": len(outer),
               "note": "Adapter trained on fit only; inner selects epoch, outer only evaluates."}
    outer[["essay_id", "score"]].assign(raw_prediction=raw, prediction=predictions).to_csv(output / "validation_predictions.csv", index=False)
    test_raw = predict(test_tokens)
    scored = pd.Series(integer_scores(test_raw), index=test.essay_id)
    submission = sample[["essay_id"]].copy()
    submission["score"] = submission.essay_id.map(scored)
    if submission.score.isna().any() or not submission.score.isin(range(1, 7)).all():
        raise ValueError("Invalid submission")
    submission.to_csv(output / "submission.csv", index=False)
    metrics["total_seconds"] = time.monotonic() - start_time
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
