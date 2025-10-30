import argparse
import asyncio
import json
import re
import logging
from pathlib import Path
from tqdm import tqdm
from dataclasses import dataclass
from openai import AsyncOpenAI

from experiments.geesc.utils.dataset_loader import GEESCDataset, load_gsm8k, load_math, load_logiqa, load_reclor, load_ruletaker


def _extract_first_json_object(s: str):
    if not s:
        return None
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    try:
        return json.loads(s)
    except Exception:
        pass
    in_str = False
    esc = False
    depth = 0
    start = -1
    for i, ch in enumerate(s):
        if ch == "\\" and not esc:
            esc = True
            continue
        if ch == '"' and not esc:
            in_str = not in_str
        esc = False
        if in_str:
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start != -1:
                candidate = s[start:i+1]
                try:
                    return json.loads(candidate)
                except Exception:
                    start = -1
    m = re.search(r"\{.*?\}", s, flags=re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None


@dataclass
class GEESCConfig:
    model: str
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    presence_penalty: float = 1.5
    enable_thinking: bool = False
    prefix_tokens: int = 96
    prefix_paths: int = 5
    judge_tokens: int = 128
    complete_tokens: int = 32768
    coherence_threshold: float = 0.5


@dataclass
class GEESCOutput:
    question: str
    gt_answer: str


@dataclass
class Prefix:
    text: str
    coherence: float
    reason: str = ""


def build_async_client(base_url: str, api_key: str) -> AsyncOpenAI:
    return AsyncOpenAI(base_url=base_url, api_key=api_key)


async def run():
    client = build_async_client(args.base_url, args.api_key)
    if args.dataset == "gsm8k":
        dataset_path = args.dataset_path
        records = load_gsm8k(dataset_path)
    elif args.dataset == "math":
        dataset_path = args.dataset_path
        records = load_math(dataset_path)
    elif args.dataset == "logiqa":
        dataset_path = args.dataset_path
        records = load_logiqa(dataset_path)
    elif args.dataset == "reclor":
        dataset_path = args.dataset_path
        records = load_reclor(dataset_path)
    elif args.dataset == "ruletaker":
        dataset_path = args.dataset_path
        records = load_ruletaker(dataset_path)
    dataset = GEESCDataset(args.dataset, records)
    config = GEESCConfig(
        model=args.model,
        prefix_tokens=args.prefix_tokens,
        prefix_paths=args.prefix_paths,
        judge_tokens=args.judge_tokens,
        complete_tokens=args.complete_tokens,
        coherence_threshold=args.coherence_threshold)  # 默认使用temperature=0.7, top_p=0.8, top_k=20, presence_penalty=1.5
    result_path = f"results/geesc_{args.dataset}_{args.model}.jsonl"    # 存储单个样本的问题，推理，答案，使用token等等
    log_path = f"logs/geesc_{args.dataset}_{args.model}.log"
    Path(result_path).parent.mkdir(parents=True, exist_ok=True)
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=log_path, level=logging.INFO, encoding='utf-8')
    logging.info("="*200)
    semaphore = asyncio.Semaphore(args.concurrency) # 限制协程数量
    file_lock = asyncio.Lock()  # 串行写文件
    process_bar = tqdm(total=len(dataset.records), desc=f"Processing {args.dataset}")   # 手动更新进度条
    tasks = [
        process_record(rec, client, config, result_path, semaphore, file_lock, process_bar)
        for rec in dataset.records
    ]
    await asyncio.gather(*tasks)
    process_bar.close()


async def process_record(record, client: AsyncOpenAI, config: GEESCConfig, result_path: str, semaphore: asyncio.Semaphore, file_lock: asyncio.Lock, process_bar: tqdm):
    try:
        async with semaphore:
            result = await process_single_record(record, client, config, result_path, semaphore, file_lock, process_bar)
            return result
    except Exception:
        logging.exception("record failed")
        return None
    finally:
        process_bar.update(1)


async def process_single_record(record, client: AsyncOpenAI, config: GEESCConfig, result_path: str, semaphore: asyncio.Semaphore, file_lock: asyncio.Lock, process_bar: tqdm):
    question = record["question"]
    gt_answer = record["gt_answer"]
    output = GEESCOutput(question=question, gt_answer=gt_answer)
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0
    # 前缀可以不用控制协程数量
    gen_tasks = [gen_prefix(question, client, config.model, config) for _ in range(config.prefix_paths)]
    gen_results = await asyncio.gather(*gen_tasks)
    prefixes = []
    filter_prefixes = []
    keep_prefixes = []
    for text, tokens in gen_results:
        prefixes.append(text)
        total_prompt_tokens += tokens["prompt_tokens"]
        total_completion_tokens += tokens["completion_tokens"]
        total_tokens += tokens["total_tokens"]
    # 评估前缀可以不用控制协程数量
    filter_tasks = [judge_prefix(question, prefix, client, config.model, config) for prefix in prefixes]
    filter_results = await asyncio.gather(*filter_tasks)
    for filter_result, tokens in filter_results:
        total_prompt_tokens += tokens["prompt_tokens"]
        total_completion_tokens += tokens["completion_tokens"]
        total_tokens += tokens["total_tokens"]
        filter_prefixes.append(filter_result)
        if filter_result.coherence >= config.coherence_threshold:
            keep_prefixes.append(filter_result)
    if not keep_prefixes:
        logging.info(f"No valid prefixes for question: {question}")
        candidates = [prefix for prefix in filter_prefixes if isinstance(prefix, Prefix)]
        keep_prefixes = [max(candidates, key=lambda p: p.coherence)] if candidates else []
    results = []
    for keep_prefix in keep_prefixes:
        answer, tokens = await answer_with_prefix(question, keep_prefix.text, client, config.model, config)
        total_prompt_tokens += tokens["prompt_tokens"]
        total_completion_tokens += tokens["completion_tokens"]
        total_tokens += tokens["total_tokens"]
        results.append({
            "prefix": keep_prefix.text,
            "prefix_coherence": keep_prefix.coherence,
            "prefix_reason": keep_prefix.reason,
            "answer": answer
        })
    out_lines = {
        "question": output.question,
        "gt_answer": output.gt_answer,
        "results": results,
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "total_tokens": total_tokens,
    }
    # 串行写文件
    async with file_lock:
        with open(result_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(out_lines, ensure_ascii=False) + "\n")
    return out_lines


async def gen_prefix(question: str, client: AsyncOpenAI, model: str, config: GEESCConfig) -> tuple[str, dict]:
    prompt = (
        f"{question}\n\n"
        "Write only the FIRST concise step of a correct solution.\n"
        "- Focus on the key first calculation or deduction.\n"
        "- Do not give the final answer.\n"
        "- Start your line with 'Step 1:' and keep it under 2 sentences."
    )
    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=config.temperature,
        top_p=config.top_p,
        presence_penalty=config.presence_penalty,
        max_tokens=config.prefix_tokens,
        extra_body={
            "top_k": config.top_k,
            "chat_template_kwargs": {"enable_thinking": config.enable_thinking},
        },
    )
    text = (response.choices[0].message.content or "").strip()
    usage = getattr(response, "usage", None)
    tokens = {
        "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
        "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
        "total_tokens": getattr(usage, "total_tokens", 0) or 0,
    }
    return text, tokens


async def judge_prefix(question: str, prefix: str, client: AsyncOpenAI, model: str, config: GEESCConfig) -> tuple[Prefix, dict]:
    system_prompt = (
        "You are a strict judge. Score how coherent and on-topic the given reasoning prefix is for the question. "
        "Return JSON with fields: coherence (0-1, float), reason (short)."
    )
    user_prompt = f"Question:\n{question}\n\nPrefix:\n{prefix}\n\nReturn JSON only."
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt}, 
            {"role": "user", "content": user_prompt}
        ],
        temperature=config.temperature,
        top_p=config.top_p,
        presence_penalty=config.presence_penalty,
        max_tokens=config.judge_tokens,
        extra_body={
            "top_k": config.top_k,
            "chat_template_kwargs": {"enable_thinking": config.enable_thinking},
        },
    )
    content = (response.choices[0].message.content or "").strip()
    try:
        data = json.loads(content)
    except Exception:
        data = _extract_first_json_object(content) or {}
    if not isinstance(data, dict):
        data = {}
    try:
        coherence = float(data.get("coherence", 0.0))
    except Exception:
        coherence = 0.0
    reason = data.get("reason", "") or "parse_error"
    usage = getattr(response, "usage", None)
    tokens = {
        "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
        "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
        "total_tokens": getattr(usage, "total_tokens", 0) or 0,
    }
    return Prefix(text=prefix, coherence=coherence, reason=reason), tokens


async def answer_with_prefix(question: str, prefix: str, client: AsyncOpenAI, model: str, config: GEESCConfig) -> tuple[str, dict]:
    prompt = (
        f"Question:\n{question}\n\n"
        f"You already wrote the first step:\n{prefix}\n\n"
        "Now continue the reasoning step by step and provide a final answer.\n"
        "Format your last line exactly as: Final Answer: <answer>\n"
        "Do not include anything after the Final Answer line."
    )
    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=config.temperature,
        top_p=config.top_p,
        presence_penalty=config.presence_penalty,
        max_tokens=config.complete_tokens,
        extra_body={
            "top_k": config.top_k,
            "chat_template_kwargs": {"enable_thinking": config.enable_thinking},
        },
    )
    content = (response.choices[0].message.content or "").strip()
    usage = getattr(response, "usage", None)
    tokens = {
        "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
        "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
        "total_tokens": getattr(usage, "total_tokens", 0) or 0,
    }
    return content, tokens


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--base_url", required=True)
    parser.add_argument("--api_key", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--prefix_tokens", type=int, default=None)  # 前缀最大token数
    parser.add_argument("--prefix_paths", type=int, default=None)   # 前缀路径数量
    parser.add_argument("--judge_tokens", type=int, default=None)   # 判别前缀的最大token数
    parser.add_argument("--complete_tokens", type=int, default=None)   # 使用前缀回答问题的最大token数
    parser.add_argument("--concurrency", type=int, default=1)   # 并发协程数量
    parser.add_argument("--coherence_threshold", type=float, default=0.5)   # 前缀一致性阈值
    args = parser.parse_args()
    asyncio.run(run())
    # python -m experiments.geesc.launch.main --base_url http://localhost:8000/v1 --api_key anykey --model qwen3-8b --dataset gsm8k --dataset_path datasets/gsm8k/test.jsonl --prefix_tokens 96 --prefix_paths 5 --judge_tokens 128 --complete_tokens 16384 --concurrency 1 --coherence_threshold 0.5
