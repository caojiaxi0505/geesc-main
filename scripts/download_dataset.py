"""
从Hugging Face下载GSM8K和MATH500
"""
import argparse
import json
import os
from pathlib import Path
from typing import List, Optional
from datasets import load_dataset, Dataset, DatasetDict, disable_caching, config as ds_config


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def save_split_to_jsonl(ds: Dataset, out_path: Path, fields: Optional[List[str]] = None) -> None:
    """将一个 split 保存为 JSONL。若指定 fields，则只导出这些字段。"""
    ensure_dir(out_path.parent)
    with out_path.open("w", encoding="utf-8") as f:
        for row in ds:
            if fields is not None:
                row = {k: row.get(k) for k in fields if k in row}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def try_load_dataset(
    name: str,
    subset: Optional[str],
    splits_try: List[str],
    token: Optional[str],
    num_proc: int,
):
    loaded = {}
    for split in splits_try:
        try:
            ds = load_dataset(
                name,
                subset,
                split=split,
                use_auth_token=token if token else None,
                num_proc=num_proc if num_proc > 0 else None,
            )
            loaded[split] = ds
        except Exception:
            continue
    if not loaded:
        try:
            dsd: DatasetDict = load_dataset(
                name,
                subset,
                use_auth_token=token if token else None,
                num_proc=num_proc if num_proc > 0 else None,
            )
            for k, v in dsd.items():
                if isinstance(v, Dataset):
                    loaded[k] = v
        except Exception:
            pass
    return loaded


def download_gsm8k(out_dir: Path, token: Optional[str], num_proc: int):
    print("[GSM8K] Loading dataset: gsm8k (config='main')")
    ds_splits = try_load_dataset(
        name="gsm8k",
        subset="main",
        splits_try=["train", "test", "validation"],
        token=token,
        num_proc=num_proc,
    )
    if not ds_splits:
        raise RuntimeError("未能加载 GSM8K（gsm8k, config='main'）。请检查网络或 Hugging Face 访问权限。")
    preferred_fields = ["question", "answer"]
    out_base = out_dir / "gsm8k"
    for split_name, d in ds_splits.items():
        fields = preferred_fields if all(k in d.column_names for k in preferred_fields) else None
        out_path = out_base / f"{split_name}.jsonl"
        print(f"[GSM8K] Saving split '{split_name}' -> {out_path}")
        save_split_to_jsonl(d, out_path, fields=fields)


def download_math500(out_dir: Path, token: Optional[str], num_proc: int):
    candidates = [
        ("lighteval/MATH500", None),
        ("openai/math-500", None),
        ("HuggingFaceH4/MATH-500", None),
        ("hendrycks/competition_math", None),
    ]
    splits_try = ["test", "validation", "dev", "train"]
    last_err = None
    for name, subset in candidates:
        print(f"[MATH500] Trying dataset: {name}" + (f" ({subset})" if subset else ""))
        try:
            ds_splits = try_load_dataset(name, subset, splits_try, token, num_proc)
            if ds_splits:
                out_base = out_dir / "math500"
                preferred_fields = ["problem", "solution", "level", "type"]
                for split_name, d in ds_splits.items():
                    fields = preferred_fields if all(k in d.column_names for k in preferred_fields) else None
                    out_path = out_base / f"{split_name}.jsonl"
                    print(f"[MATH500] Saving split '{split_name}' from '{name}' -> {out_path}")
                    save_split_to_jsonl(d, out_path, fields=fields)
                return
            else:
                print(f"[MATH500] No splits found in '{name}'. Trying next candidate...")
        except Exception as e:
            last_err = e
            print(f"[MATH500] Failed to load '{name}': {e}. Trying next candidate...")
    raise RuntimeError(
        "未能找到可用的 MATH500 数据集镜像（已尝试多种候选命名）。\n"
        "建议手动在 https://huggingface.co/datasets 搜索 MATH500 的具体仓库名，"
        "或将上面代码中的 candidates 列表改为你使用的仓库名。"
        + (f"\n最后错误：{last_err}" if last_err else "")
    )


def main():
    parser = argparse.ArgumentParser(description="Download GSM8K and MATH500 to a target directory as JSONL.")
    parser.add_argument("--out", required=True, help="输出根目录，例如 ./data")
    parser.add_argument("--hf_token", default=None, help="（可选）Hugging Face 访问令牌")
    parser.add_argument("--no_cache", action="store_true", help="禁用本地缓存（如遇缓存损坏可尝试）")
    parser.add_argument("--num_proc", type=int, default=0, help="下载/处理并行度；0 表示由 datasets 自行决定")
    args = parser.parse_args()
    out_dir = Path(args.out).expanduser().resolve()
    ensure_dir(out_dir)
    if args.no_cache:
        disable_caching()
        print("[INFO] Datasets caching disabled.")
    ds_config.DOWNLOADER_MAX_REUSE_CONNECTIONS = 8  # 稍微保守
    ds_config.DOWNLOADER_MAX_RETRIES = 3
    print(f"[INFO] Output directory: {out_dir}")
    print("[STEP] Downloading GSM8K...")
    download_gsm8k(out_dir, token=args.hf_token, num_proc=args.num_proc)
    print("[STEP] Downloading MATH500...")
    download_math500(out_dir, token=args.hf_token, num_proc=args.num_proc)
    print("\n✅ All done.")
    print(f"➡️ 结果位于：{out_dir}/gsm8k/*.jsonl 与 {out_dir}/MATH500/*.jsonl")


if __name__ == "__main__":
    main()
