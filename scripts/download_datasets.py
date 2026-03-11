import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download


def parse_args():
    parser = argparse.ArgumentParser(description="Download train/val datasets from Hugging Face")
    parser.add_argument("--repo-id", type=str, default="aryanb1702/awk9")
    parser.add_argument("--repo-type", type=str, default="dataset")
    parser.add_argument("--train-file", type=str, default="awk9_train_10k.pt")
    parser.add_argument("--val-file", type=str, default="awk9_val_1k.pt")
    parser.add_argument("--out-dir", type=str, default="datasets")
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = hf_hub_download(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        filename=args.train_file,
        local_dir=str(out_dir),
        local_dir_use_symlinks=False,
    )
    val_path = hf_hub_download(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        filename=args.val_file,
        local_dir=str(out_dir),
        local_dir_use_symlinks=False,
    )

    print(f"[DONE] train={train_path}")
    print(f"[DONE] val={val_path}")


if __name__ == "__main__":
    main()
