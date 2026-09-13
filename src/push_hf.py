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
    print(f"Pushing dataset -> {repo} ...")
    dd.push_to_hub(repo)

    card = DOCS / "dataset_card.md"
    if card.exists():
        from huggingface_hub import HfApi
        HfApi().upload_file(path_or_fileobj=str(card), path_in_repo="README.md",
                            repo_id=repo, repo_type="dataset")
        print("Dataset card uploaded as README on Hub.")
    return repo


def push_model(user: str) -> str:
    from huggingface_hub import HfApi

    repo = f"{user}/phobert-vn-fin-sentiment"
    if not MODEL_DIR.exists() or not any(MODEL_DIR.iterdir()):
        sys.exit(f"Model not found at {MODEL_DIR}. Run train_phobert.py first, "
                 "or omit --with-model.")
    api = HfApi()
    api.create_repo(repo, exist_ok=True)
    print(f"Pushing model -> {repo} ...")
    api.upload_folder(folder_path=str(MODEL_DIR), repo_id=repo)
    card = DOCS / "model_card.md"
    if card.exists():
        api.upload_file(path_or_fileobj=str(card), path_in_repo="README.md",
                        repo_id=repo)
        print("Model card uploaded as README on Hub.")
    return repo


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True, help="your HuggingFace username")
    ap.add_argument("--with-model", action="store_true")
    args = ap.parse_args()

    ds_repo = push_dataset(args.user)
    md_repo = push_model(args.user) if args.with_model else None

    print("\nDone. Next step: update README.md with the real links:")
    print(f"  - Dataset: https://huggingface.co/datasets/{ds_repo}")
    if md_repo:
        print(f"  - Model:   https://huggingface.co/{md_repo}")
    print("Then commit and push with git as usual.")


if __name__ == "__main__":
    main()
