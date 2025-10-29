"""
从公开源（Wikipedia API / HF Datasets）构建轻量 RAG 语料。
输出为 JSONL：每行 {"title": ..., "text": ...}，兼容 geesc.py 的 FileCorpusRetriever。
"""
import argparse
import json
import os
import time
from pathlib import Path
from typing import List, Dict, Optional

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def save_jsonl(rows: List[Dict], out_path: Path):
    ensure_dir(out_path.parent)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[OK] Saved {len(rows)} docs -> {out_path}")

# ---------------- Wikipedia (REST API) ----------------
WIKI_API_SUMMARY = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
DEFAULT_MATH_TOPICS = [
    "Arithmetic", "Prime number", "Greatest common divisor", "Least common multiple",
    "Fractions", "Percentage", "Ratio", "Proportion", "Exponentiation", "Square root",
    "Order of operations", "Linear equation", "Quadratic equation", "Inequality (mathematics)",
    "System of linear equations", "Function (mathematics)", "Polynomial", "Binomial theorem",
    "Logarithm", "Geometric progression", "Arithmetic progression", "Probability", "Combinatorics",
    "Permutation", "Combination", "Mean", "Median", "Mode (statistics)", "Standard deviation",
    "Variance", "Great-circle distance", "Pythagorean theorem", "Area", "Perimeter",
    "Surface area", "Volume", "Circle", "Triangle", "Rectangle", "Cube", "Sphere", "Cylinder",
    "Distance", "Speed", "Time", "Work (physics)", "Simple interest", "Compound interest",
    "Ratio test", "Harmonic series (mathematics)", "Modular arithmetic", "Remainder", "Congruence relation",
]

def fetch_wikipedia_topics(topics: List[str], sleep_s: float = 0.1, timeout: float = 10.0) -> List[Dict]:
    import requests
    rows = []
    for t in topics:
        url = WIKI_API_SUMMARY.format(title=requests.utils.quote(t))
        try:
            r = requests.get(url, timeout=timeout, headers={"User-Agent":"geesc/0.1 (+RAG download script)"})
            if r.status_code == 200:
                data = r.json()
                text = data.get("extract", "")
                title = data.get("title", t)
                if text:
                    rows.append({"title": title, "text": text, "source": "wikipedia_summary"})
                else:
                    print(f"[WARN] Empty extract for {t}")
            else:
                print(f"[WARN] HTTP {r.status_code} for {t}")
        except Exception as e:
            print(f"[WARN] Failed to fetch {t}: {e}")
        time.sleep(sleep_s)
    return rows

# ---------------- Wikitext (HF Datasets) ----------------
def load_wikitext_2() -> List[Dict]:
    try:
        from datasets import load_dataset
    except Exception as e:
        raise RuntimeError("需要安装 'datasets' 库：pip install datasets") from e
    ds = load_dataset("wikitext", "wikitext-2-raw-v1")
    rows = []
    for split in ["train", "validation", "test"]:
        if split in ds:
            for ex in ds[split]:
                text = (ex.get("text") or "").strip()
                if text:
                    rows.append({"title": f"wikitext-2::{split}", "text": text, "source":"wikitext-2"})
    return rows

def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--out", required=True, help="输出目录，例如 ./rag_corpus")
    parser.add_argument("--source", required=True, choices=["wiki-math-200", "wikitext-2"], help="公开知识库源")
    parser.add_argument("--topics_file", default=None, help="当 source=wiki-math-200 时，可提供一个按行列出的 Wikipedia 标题文件；未提供则使用内置 topics。")
    parser.add_argument("--outfile", default="docs.jsonl", help="输出文件名（默认 docs.jsonl）")
    parser.add_argument("--sleep", type=float, default=0.1, help="Wikipedia 请求间隔秒（防止触发限流）")
    args = parser.parse_args()
    out_dir = Path(args.out).expanduser().resolve()
    ensure_dir(out_dir)
    if args.source == "wiki-math-200":
        topics = DEFAULT_MATH_TOPICS
        if args.topics_file:
            p = Path(args.topics_file)
            if p.exists():
                topics = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
        print(f"[INFO] Fetching {len(topics)} topics from Wikipedia summaries...")
        rows = fetch_wikipedia_topics(topics, sleep_s=args.sleep)
        save_jsonl(rows, out_dir / args.outfile)
    else:
        print("[INFO] Downloading Wikitext-2 via HF Datasets...")
        rows = load_wikitext_2()
        save_jsonl(rows, out_dir / args.outfile)

if __name__ == "__main__":
    main()
