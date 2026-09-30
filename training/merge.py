"""Merge a LoRA adapter into full 16-bit weights.

  CUDA_VISIBLE_DEVICES=1 python training/merge.py --adapter runs/ft_v1/adapter --out runs/ft_v1/merged
"""
import argparse

from unsloth import FastLanguageModel  # noqa: I001


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    model, tokenizer = FastLanguageModel.from_pretrained(model_name=args.adapter, max_seq_length=2048, load_in_4bit=False, load_in_16bit=True)
    model.save_pretrained_merged(args.out, tokenizer, save_method="merged_16bit")
    print("merged ->", args.out)


if __name__ == "__main__":
    main()
