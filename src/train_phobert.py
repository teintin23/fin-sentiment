from __future__ import annotations

import argparse
import io
import json
import random
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns            # noqa: E402
from sklearn.metrics import (accuracy_score, classification_report,  # noqa: E402
                             confusion_matrix, f1_score)

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
CACHE = ROOT / "data" / "interim" / "tokenized_cache.parquet"
MODEL_DIR = ROOT / "models" / "phobert-vnfin"
FIG_DIR = ROOT / "reports" / "figures"
DOC = ROOT / "docs" / "model_phobert.md"

SEED = 42
BASE = "vinai/phobert-base"
LABELS = ["NEGATIVE", "NEUTRAL", "POSITIVE"]
L2I = {l: i for i, l in enumerate(LABELS)}
MAX_LEN = 256


def set_all_seeds() -> None:
    import torch
    from transformers import set_seed
    random.seed(SEED); np.random.seed(SEED)
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    set_seed(SEED)


def load_tokenized() -> dict[str, str]:
    if CACHE.exists():
        c = pd.read_parquet(CACHE)
        return dict(zip(c["id"].astype(str), c["tok"]))
    print("  tokenized_cache.parquet not found, tokenizing with underthesea ...")
    from underthesea import word_tokenize
    frames = [pd.read_parquet(PROC / f"{n}.parquet") for n in ("train", "val", "test")]
    allrows = pd.concat(frames)[["id", "text_input"]].drop_duplicates("id")
    tok = {str(i): word_tokenize(str(t), format="text")
           for i, t in zip(allrows["id"], allrows["text_input"])}
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": list(tok), "tok": list(tok.values())}).to_parquet(CACHE, index=False)
    return tok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    args = ap.parse_args()

    import torch
    import torch.nn as nn
    from datasets import Dataset
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              EarlyStoppingCallback, Trainer, TrainingArguments)

    set_all_seeds()
    has_gpu = torch.cuda.is_available()
    epochs = args.epochs if args.epochs else (4 if has_gpu else 2)
    print("=" * 68)
    print(f"PhoBERT fine-tune  |  device={'GPU ' + torch.cuda.get_device_name(0) if has_gpu else 'CPU'}")
    if not has_gpu:
        print("  WARNING: no GPU detected. Estimated 3-6 hours for 2 epochs on ~4,600 articles.")
        print("  Reduced num_train_epochs to 2. Consider running on Colab/Kaggle with GPU.")
    print("=" * 68)

    tr = pd.read_parquet(PROC / "train.parquet")
    va = pd.read_parquet(PROC / "val.parquet")
    te = pd.read_parquet(PROC / "test.parquet")
    tok_map = load_tokenized()
    for d in (tr, va, te):
        d["tok"] = d["id"].astype(str).map(tok_map)
        d["labels"] = d["label"].map(L2I)
    print(f"  train {len(tr):,} | val {len(va):,} | test {len(te):,}")

    tokenizer = AutoTokenizer.from_pretrained(BASE)

    def enc(batch):
        return tokenizer(batch["tok"], truncation=True, max_length=MAX_LEN,
                         padding="max_length")

    ds = {k: Dataset.from_pandas(d[["tok", "labels"]].reset_index(drop=True))
              .map(enc, batched=True, remove_columns=["tok"])
          for k, d in (("train", tr), ("val", va), ("test", te))}

    model = AutoModelForSequenceClassification.from_pretrained(
        BASE, num_labels=3,
        id2label={i: l for l, i in L2I.items()}, label2id=L2I)

    counts = tr["labels"].value_counts().sort_index().to_numpy()
    weights = torch.tensor(len(tr) / (3 * counts), dtype=torch.float)
    print(f"  class weights: {dict(zip(LABELS, weights.tolist()))}")

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kw):
            labels = inputs.pop("labels")
            out = model(**inputs)
            loss = nn.CrossEntropyLoss(weight=weights.to(out.logits.device))(
                out.logits, labels)
            return (loss, out) if return_outputs else loss

    def metrics(p):
        pred = p.predictions.argmax(-1)
        return {"macro_f1": f1_score(p.label_ids, pred, average="macro"),
                "accuracy": accuracy_score(p.label_ids, pred)}

    targs = TrainingArguments(
        output_dir=str(ROOT / "models" / "_phobert_ckpt"),
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch * 2,
        num_train_epochs=epochs,
        warmup_ratio=0.1,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=2,
        logging_steps=50,
        seed=SEED,
        fp16=has_gpu,
        report_to=[],
    )

    trainer = WeightedTrainer(
        model=model, args=targs,
        train_dataset=ds["train"], eval_dataset=ds["val"],
        compute_metrics=metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    trainer.train()

    hist = [(h["epoch"], h["eval_macro_f1"]) for h in trainer.state.log_history
            if "eval_macro_f1" in h]
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    if hist:
        plt.figure(figsize=(6, 4))
        plt.plot([e for e, _ in hist], [f for _, f in hist], marker="o")
        plt.xlabel("Epoch"); plt.ylabel("macro-F1 on val")
        plt.title("PhoBERT — macro-F1 per epoch"); plt.grid(alpha=.3)
        plt.tight_layout(); plt.savefig(FIG_DIR / "phobert_training.png", dpi=140); plt.close()

    pr = trainer.predict(ds["test"])
    pred = pr.predictions.argmax(-1)
    y = te["labels"].to_numpy()
    mf1 = f1_score(y, pred, average="macro")
    acc = accuracy_score(y, pred)
    wf1 = f1_score(y, pred, average="weighted")
    print(f"\nTEST  macro-F1={mf1:.4f}  accuracy={acc:.4f}  weighted-F1={wf1:.4f}")
    rep = classification_report(y, pred, target_names=LABELS, digits=3, zero_division=0)
    print(rep)

    cm = confusion_matrix(y, pred, labels=list(range(3)))
    plt.figure(figsize=(5.5, 4.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Purples",
                xticklabels=LABELS, yticklabels=LABELS, cbar=False)
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.title(f"PhoBERT — test (macro-F1={mf1:.3f})")
    plt.tight_layout(); plt.savefig(FIG_DIR / "cm_phobert.png", dpi=140); plt.close()

    sub = None
    if "source_detail" in te:
        hm = (te["source_detail"] == "human").to_numpy()
        if hm.sum() > 0:
            sub = {"n": int(hm.sum()),
                   "macro_F1": round(f1_score(y[hm], pred[hm], average="macro"), 4),
                   "accuracy": round(accuracy_score(y[hm], pred[hm]), 4)}
            print(f"Human labels in test (n={sub['n']}): macro-F1={sub['macro_F1']:.4f}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(MODEL_DIR)); tokenizer.save_pretrained(str(MODEL_DIR))
    (MODEL_DIR / "test_predictions.json").write_text(
        json.dumps({"id": te["id"].astype(str).tolist(),
                    "y_true": y.tolist(), "y_pred": pred.tolist()}), encoding="utf-8")
    print(f"Model saved: {MODEL_DIR}")

    write_doc(mf1, acc, wf1, rep, cm, sub, hist, epochs, args, has_gpu, te)
    print(f"Written: {DOC}")
    print(f"\nP2.2 PASS  |  macro-F1 on test = {mf1:.4f}")
    return 0


def write_doc(mf1, acc, wf1, rep, cm, sub, hist, epochs, args, has_gpu, te) -> None:
    DOC.parent.mkdir(parents=True, exist_ok=True)
    with DOC.open("w", encoding="utf-8") as f:
        w = f.write
        w("# PhoBERT fine-tuned\n\n")
        w(f"Base model: `{BASE}`, `num_labels=3`, `max_length={MAX_LEN}`.\n")
        w("Text must be word-segmented with `underthesea.word_tokenize` **before** "
          "tokenization — skipping this step degrades performance significantly.\n\n")
        w("## 1. Hyperparameters\n\n")
        w("| Parameter | Value |\n|---|---|\n")
        w(f"| learning_rate | {args.lr} |\n| batch size | {args.batch} |\n"
          f"| epochs | {epochs} |\n| warmup_ratio | 0.1 |\n| weight_decay | 0.01 |\n"
          f"| early stopping patience | 2 |\n| class weights | yes, inverse label frequency |\n"
          f"| hardware | {'GPU' if has_gpu else 'CPU'} |\n| seed | {SEED} |\n\n")
        w("## 2. Test results\n\n")
        w("| Metric | Value |\n|---|---|\n")
        w(f"| macro-F1 | **{mf1:.4f}** |\n| accuracy | {acc:.4f} |\n"
          f"| weighted-F1 | {wf1:.4f} |\n\n")
        w("## 3. Classification report\n\n```\n" + rep + "```\n\n")
        w("## 4. Confusion matrix\n\n")
        w("| | " + " | ".join(LABELS) + " |\n|---|" + "---|" * 3 + "\n")
        for i, lb in enumerate(LABELS):
            w(f"| **{lb}** | " + " | ".join(str(x) for x in cm[i]) + " |\n")
        w("\n![cm](../reports/figures/cm_phobert.png)\n\n")
        w("## 5. macro-F1 per epoch (val)\n\n")
        if hist:
            w("| Epoch | macro-F1 val |\n|---|---|\n")
            for e, s in hist:
                w(f"| {e:.2f} | {s:.4f} |\n")
        w("\n![training](../reports/figures/phobert_training.png)\n\n")
        w("> Val contains ~94% `weak_model` labels, so this curve reflects how well\n"
          "> the model mimics noisy labels rather than reading news. Use it for checkpoint\n"
          "> selection only, not as a performance figure.\n\n")
        w("## 6. Results on human-labeled subset\n\n")
        if sub:
            w(f"- n = {sub['n']}\n- macro-F1: **{sub['macro_F1']:.4f}**\n"
              f"- accuracy: {sub['accuracy']:.4f}\n\n")
            w("Small sample size; use for qualitative comparison only.\n\n")
        if "source_detail" in te:
            w("### Label source breakdown in test set\n\n| Source | Count |\n|---|---:|\n")
            for k, v in te["source_detail"].value_counts().items():
                w(f"| `{k}` | {v} |\n")
            w("\n")
        w("## 7. How to use the model\n\n")
        w("Input must be `\"[TICKER] Headline. Lead paragraph\"` and **must be word-segmented** "
          "before passing to the tokenizer:\n\n")
        w("```python\n"
          "from transformers import AutoTokenizer, AutoModelForSequenceClassification\n"
          "from underthesea import word_tokenize\n"
          "import torch\n\n"
          "tok = AutoTokenizer.from_pretrained('models/phobert-vnfin')\n"
          "model = AutoModelForSequenceClassification.from_pretrained('models/phobert-vnfin')\n\n"
          "text = '[HPG] Hoa Phat reports record Q2 profit. Net income up 48% YoY.'\n"
          "seg = word_tokenize(text, format='text')\n"
          "x = tok(seg, return_tensors='pt', truncation=True, max_length=256)\n"
          "print(model.config.id2label[model(**x).logits.argmax(-1).item()])\n"
          "```\n\n")
        w("## 8. Limitations\n\n")
        w("- ~89% of train labels come from a weak propagation model; performance ceiling "
          "is set by label quality, not architecture.\n")
        w("- Reads headlines and lead paragraphs only, not full article text.\n")
        w("- Ticker attribution accuracy ~85%; wrong ticker means wrong label target.\n")
        w("- Single source (CafeF), short time window (late 2024 to mid 2026).\n")
        w("- Not for use in real investment decisions.\n")


if __name__ == "__main__":
    sys.exit(main())
