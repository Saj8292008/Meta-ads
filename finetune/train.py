"""LoRA fine-tune a small chat model to write Meta ad copy, logging every step.

Each run writes runs/<name>/log.jsonl with one JSON event per line:
  {"type": "config", ...}                       once, at the start
  {"type": "step", "step", "loss", "lr", "t"}   every optimizer step
  {"type": "checkpoint", "step", "eval_loss", "format_ok", "samples"}  at step 0 and every --ckpt-every steps
  {"type": "done", "t"}                         at the end
"""
import argparse, json, math, os, random, re, time

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
CTAS = {"Shop Now", "Learn More", "Sign Up", "Order Now", "Book Now", "Get Offer", "Download"}
SAMPLE_PRODUCTS = [  # held out from training; the same four are answered at every checkpoint
    ("Sleep tracking ring", "people who wake up tired"),
    ("Neighborhood pizza place delivery", "families on Friday night"),
    ("Budget travel planning app", "backpackers"),
    ("Home cleaning service", "dual-income couples"),
]


def load(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def prompt_msgs(product, audience):
    return [{"role": "user", "content": f"Write a Meta ad.\nProduct: {product}\nAudience: {audience}"}]


def target(ex):
    return f"Headline: {ex['headline']}\nPrimary text: {ex['primary']}\nCTA: {ex['cta']}"


def encode(tok, ex):
    p = tok.apply_chat_template(prompt_msgs(ex["product"], ex["audience"]), tokenize=False, add_generation_prompt=True)
    p_ids = tok(p, add_special_tokens=False)["input_ids"]
    a_ids = tok(target(ex) + tok.eos_token, add_special_tokens=False)["input_ids"]
    return p_ids + a_ids, [-100] * len(p_ids) + a_ids


def batches(items, size, pad_id):
    for i in range(0, len(items), size):
        chunk = items[i:i + size]
        n = max(len(x) for x, _ in chunk)
        ids = torch.full((len(chunk), n), pad_id)
        lab = torch.full((len(chunk), n), -100)
        att = torch.zeros((len(chunk), n), dtype=torch.long)
        for j, (x, y) in enumerate(chunk):
            ids[j, :len(x)] = torch.tensor(x)
            lab[j, :len(y)] = torch.tensor(y)
            att[j, :len(x)] = 1
        yield ids, lab, att


def format_ok(text):
    m = re.search(r"Headline:\s*(.+)\nPrimary text:\s*(.+)\nCTA:\s*(.+)", text)
    return bool(m) and len(m.group(1).strip()) <= 40 and m.group(3).strip() in CTAS


@torch.no_grad()
def checkpoint(model, tok, evalset, step):
    model.eval()
    losses = []
    for ids, lab, att in batches([encode(tok, e) for e in evalset], 4, tok.pad_token_id):
        losses.append(model(input_ids=ids, attention_mask=att, labels=lab).loss.item())
    samples = []
    for product, audience in SAMPLE_PRODUCTS:
        p = tok.apply_chat_template(prompt_msgs(product, audience), tokenize=False, add_generation_prompt=True)
        enc = tok(p, return_tensors="pt", add_special_tokens=False)
        out = model.generate(**enc, max_new_tokens=90, do_sample=False, pad_token_id=tok.pad_token_id)
        text = tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        samples.append({"product": product, "audience": audience, "output": text, "ok": format_ok(text)})
    model.train()
    return {"type": "checkpoint", "step": step, "eval_loss": round(sum(losses) / len(losses), 4),
            "format_ok": sum(s["ok"] for s in samples) / len(samples), "samples": samples}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-360M-Instruct")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--ckpt-every", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    random.seed(a.seed); torch.manual_seed(a.seed)
    torch.set_num_threads(os.cpu_count() or 4)
    out_dir = os.path.join(HERE, "runs", a.run)
    os.makedirs(out_dir, exist_ok=True)
    log = open(os.path.join(out_dir, "log.jsonl"), "w", buffering=1)
    emit = lambda ev: log.write(json.dumps(ev) + "\n")

    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32)
    model = get_peft_model(model, LoraConfig(r=a.rank, lora_alpha=a.rank * 2, lora_dropout=0.05, task_type="CAUSAL_LM",
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
    train = load(os.path.join(HERE, "data", "ads.jsonl"))
    evalset = load(os.path.join(HERE, "data", "eval.jsonl"))
    steps_per_epoch = math.ceil(len(train) / a.batch)
    total = steps_per_epoch * a.epochs
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    emit({"type": "config", "run": a.run, "model": a.model, "lr": a.lr, "epochs": a.epochs, "rank": a.rank,
          "batch": a.batch, "train_examples": len(train), "total_steps": total, "trainable_params": trainable,
          "started": time.time()})

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 5) * max(0.0, 1 - s / total))
    t0 = time.time()
    emit(checkpoint(model, tok, evalset, 0))
    model.train()
    step = 0
    for _ in range(a.epochs):
        random.shuffle(train)
        for ids, lab, att in batches([encode(tok, e) for e in train], a.batch, tok.pad_token_id):
            loss = model(input_ids=ids, attention_mask=att, labels=lab).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            step += 1
            emit({"type": "step", "step": step, "loss": round(loss.item(), 4), "lr": sched.get_last_lr()[0],
                  "t": round(time.time() - t0, 1)})
            if step % a.ckpt_every == 0 and step != total:
                emit(checkpoint(model, tok, evalset, step))
    emit(checkpoint(model, tok, evalset, step))
    model.save_pretrained(os.path.join(out_dir, "adapter"))
    emit({"type": "done", "t": round(time.time() - t0, 1)})


if __name__ == "__main__":
    main()
