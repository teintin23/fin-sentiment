"""
push_hf.py
----------
Day dataset va model len HuggingFace Hub bang mot lenh.

Chuan bi (lam 1 lan):
    pip install huggingface_hub datasets
    huggingface-cli login        # dan token co quyen write, lay tai hf.co/settings/tokens

Chay:
    python src/push_hf.py --user TEN_HF_CUA_BAN                 # day dataset
    python src/push_hf.py --user TEN_HF_CUA_BAN --with-model    # day ca model (can models/phobert-vnfin/ tren may)

Script tu dong:
- gop train/val/test parquet thanh DatasetDict
- dung docs/dataset_card.md va docs/model_card.md lam README tren Hub
- thay <username> trong README.md local bang ten that (in ra diff, khong tu commit)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
DOCS = ROOT / "docs"
MODEL_DIR = ROOT / "models" / "phobert-vnfin"


def push_dataset(user: str) -> str:
    from datasets import Dataset, DatasetDict
    import pandas as pd

    repo = f"{user}/vn-fin-sentiment"
    dd = DatasetDict({
        split: Dataset.from_pandas(
            pd.read_parquet(PROC / f"{split}.parquet"), preserve_index=False)
        for split in ("train", "val", "test")
    })
    print(f"Day dataset -> {repo} ...")
    dd.push_to_hub(repo)

    card = DOCS / "dataset_card.md"
    if card.exists():
        from huggingface_hub import HfApi
        HfApi().upload_file(path_or_fileobj=str(card), path_in_repo="README.md",
                            repo_id=repo, repo_type="dataset")
        print("Da upload dataset card lam README tren Hub.")
    return repo


def push_model(user: str) -> str:
    from huggingface_hub import HfApi

    repo = f"{user}/phobert-vn-fin-sentiment"
    if not MODEL_DIR.exists() or not any(MODEL_DIR.iterdir()):
        sys.exit(f"Khong thay model tai {MODEL_DIR}. Chay train_phobert.py truoc, "
                 "hoac bo --with-model.")
    api = HfApi()
    api.create_repo(repo, exist_ok=True)
    print(f"Day model -> {repo} ...")
    api.upload_folder(folder_path=str(MODEL_DIR), repo_id=repo)
    card = DOCS / "model_card.md"
    if card.exists():
        api.upload_file(path_or_fileobj=str(card), path_in_repo="README.md",
                        repo_id=repo)
        print("Da upload model card lam README tren Hub.")
    return repo


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True, help="username HuggingFace cua ban")
    ap.add_argument("--with-model", action="store_true")
    args = ap.parse_args()

    ds_repo = push_dataset(args.user)
    md_repo = push_model(args.user) if args.with_model else None

    print("\nXong. Buoc cuoi: sua README.md, thay dong 'chua day' bang link that:")
    print(f"  - Dataset: https://huggingface.co/datasets/{ds_repo}")
    if md_repo:
        print(f"  - Model:   https://huggingface.co/{md_repo}")
    print("Roi commit + push git nhu binh thuong.")


if __name__ == "__main__":
    main()
