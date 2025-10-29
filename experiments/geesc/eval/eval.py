import argparse
import json
from statistics import mean
import re

from experiments.geesc.utils.dataset_loader import GEESCDataset, load_gsm8k, extract_gsm8k_gt, load_math
from experiments.geesc.utils.math_equivalence import is_equiv


def read_token_file(path: str):
    p_list, c_list, t_list = [], [], []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            a, b, c = [x.strip() for x in line.split(",")]
            p_list.append(int(a)); c_list.append(int(b)); t_list.append(int(c))
    return p_list, c_list, t_list


def evaluate_gsm8k(dataset_path: str, results_prefix: str):
    data = load_gsm8k(dataset_path)
    pred_file = f"{results_prefix}.txt"
    token_file = f"{results_prefix}_token.txt"
    with open(pred_file, 'r', encoding='utf-8') as f:
        preds = [line.strip() for line in f if line.strip()]
    p_list, c_list, t_list = read_token_file(token_file)
    n = min(len(data), len(preds), len(t_list))
    correct = 0
    for i in range(n):
        gt = extract_gsm8k_gt(data[i].get("answer", "")).strip()
        if gt == preds[i]:
            correct += 1
    print("====== GSM8K EVAL ======")
    print(f"Samples: {n}")
    print(f"Accuracy: {100.0*correct/n:.2f}% ({correct}/{n})")
    print(f"Avg Prompt Tokens:     {mean(p_list[:n]):.2f}")
    print(f"Avg Completion Tokens: {mean(c_list[:n]):.2f}")
    print(f"Avg Total Tokens:      {mean(t_list[:n]):.2f}")


def _strip_latex_wrappers(ans: str) -> str:
    a = ans.strip()
    if a.startswith("$") and a.endswith("$"):
        a = a[1:-1].strip()
    m = re.match(r"\\boxed\{(.+)\}$", a)
    if m:
        a = m.group(1).strip()
    return a


def extract_math_pred_from_content(content: str) -> str:
    boxed = re.findall(r"\\boxed\{([^{}]+)\}", content)
    if boxed:
        return _strip_latex_wrappers(boxed[-1])
    m = re.search(r"(Final Answer|Answer)\s*[:：]\s*(.+)", content, flags=re.IGNORECASE)
    if m:
        return _strip_latex_wrappers(m.group(2).strip().splitlines()[0])
    dollars = re.findall(r"\$([^$]+)\$", content)
    if dollars:
        return _strip_latex_wrappers(dollars[-1])
    lines = [ln.strip() for ln in content.strip().splitlines() if ln.strip()]
    return _strip_latex_wrappers(lines[-1]) if lines else ""


def evaluate_math(dataset_path: str, results_prefix: str):
    data = load_math(dataset_path)
    content_file = f"{results_prefix}_content.jsonl"
    token_file = f"{results_prefix}_token.txt"
    contents = []
    with open(content_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            contents.append(obj.get("content", ""))
    preds = [extract_math_pred_from_content(c) for c in contents]
    p_list, c_list, t_list = read_token_file(token_file)
    n = min(len(data), len(preds), len(t_list))
    correct = 0
    for i in range(n):
        gt = data[i].get("answer", "")
        if is_equiv(preds[i], gt, verbose=False):
            correct += 1
    print("====== MATH500 EVAL ======")
    print(f"Samples: {n}")
    print(f"Accuracy: {100.0*correct/n:.2f}% ({correct}/{n})")
    print(f"Avg Prompt Tokens:     {mean(p_list[:n]):.2f}")
    print(f"Avg Completion Tokens: {mean(c_list[:n]):.2f}")
    print(f"Avg Total Tokens:      {mean(t_list[:n]):.2f}")


# python -m experiments.geesc.eval.eval --dataset gsm8k --dataset_path datasets/gsm8k/test.jsonl --results_prefix results/direct_qwen3-8b_gsm8k
# python -m experiments.geesc.eval.eval --dataset math --dataset_path datasets/math500/test.jsonl --results_prefix results/direct_qwen3-8b_math
# python -m experiments.geesc.eval.eval --dataset gsm8k --dataset_path datasets/gsm8k/test.jsonl --results_prefix results/direct_qwen3-4b_gsm8k
# python -m experiments.geesc.eval.eval --dataset math --dataset_path datasets/math500/test.jsonl --results_prefix results/direct_qwen3-4b_math
# python -m experiments.geesc.eval.eval --dataset gsm8k --dataset_path datasets/gsm8k/test.jsonl --results_prefix results/direct_qwen3-1.7b_gsm8k
# python -m experiments.geesc.eval.eval --dataset math --dataset_path datasets/math500/test.jsonl --results_prefix results/direct_qwen3-1.7b_math

# python -m experiments.geesc.eval.eval --dataset gsm8k --dataset_path datasets/gsm8k/test.jsonl --results_prefix results/cot_qwen3-8b_gsm8k
# python -m experiments.geesc.eval.eval --dataset math --dataset_path datasets/math500/test.jsonl --results_prefix results/cot_qwen3-8b_math
# python -m experiments.geesc.eval.eval --dataset gsm8k --dataset_path datasets/gsm8k/test.jsonl --results_prefix results/cot_qwen3-4b_gsm8k
# python -m experiments.geesc.eval.eval --dataset math --dataset_path datasets/math500/test.jsonl --results_prefix results/cot_qwen3-4b_math
# python -m experiments.geesc.eval.eval --dataset gsm8k --dataset_path datasets/gsm8k/test.jsonl --results_prefix results/cot_qwen3-1.7b_gsm8k
# python -m experiments.geesc.eval.eval --dataset math --dataset_path datasets/math500/test.jsonl --results_prefix results/cot_qwen3-1.7b_math


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--results_prefix", required=True)
    args = parser.parse_args()
    if args.dataset == "gsm8k":
        evaluate_gsm8k(args.dataset_path, args.results_prefix)
    else:
        evaluate_math(args.dataset_path, args.results_prefix)


if __name__ == "__main__":
    main()
