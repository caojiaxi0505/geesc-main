import os
import re
import json
import math
import time
import argparse
import logging
import asyncio
from tqdm.asyncio import tqdm
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter

from openai import OpenAI, AsyncOpenAI
from experiments.geesc.utils.extract_answer import extract_gsm8k_answer
from experiments.geesc.utils.dataset_loader import GEESCDataset, load_gsm8k, load_math, load_logiqa, load_reclor, load_ruletaker
from experiments.geesc.utils.math_equivalence import is_equiv

def build_client(base_url: str, api_key: str) -> OpenAI:
    return OpenAI(base_url=base_url, api_key=api_key)

def build_async_client(base_url: str, api_key: str) -> AsyncOpenAI:
    return AsyncOpenAI(base_url=base_url, api_key=api_key)

def ensure_dir(p: str):
    Path = __import__("pathlib").Path
    Path(p).parent.mkdir(parents=True, exist_ok=True)

def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())

def extract_math_answer(text: str) -> str:
    if not text:
        return ""
    m = re.search(r"Final Answer\s*[:：]\s*([^\n]+)", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()
    boxed = re.findall(r"\\boxed\{([^{}]+)\}", text)
    if boxed:
        return boxed[-1].strip()
    nums = re.findall(r"[-+]?\d+(?:\.\d+)?(?:\/\d+)?", text)
    if nums:
        return nums[-1]
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    return lines[-1] if lines else ""

def normalize_ans(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip()
    s = s.replace(",", "").strip()
    if s.endswith("."):
        s = s[:-1]
    s = re.sub(r"\s+", " ", s)
    return s

def majority_vote(items: List[str]) -> Tuple[str, Dict[str, int]]:
    normed = [normalize_ans(x) for x in items if x is not None]
    c = Counter(normed)
    if not c:
        return "", {}
    w, _ = c.most_common(1)[0]
    return w, dict(c)

class NullRetriever:
    def retrieve(self, query: str, k: int = 3):
        return []

class FileCorpusRetriever:
    def __init__(self, corpus_path: str):
        self.docs = []
        self.idf = {}
        self._load(corpus_path)
        self._compute_idf()

    def _load(self, path: str):
        if not path or not os.path.exists(path):
            return
        if path.endswith(".jsonl"):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        if isinstance(obj, dict) and "text" in obj:
                            self.docs.append(str(obj["text"]))
                        else:
                            self.docs.append(str(obj))
                    except Exception:
                        self.docs.append(line)
        else:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.docs.append(line)

    @staticmethod
    def _tok(s: str) -> List[str]:
        return re.findall(r"[a-zA-Z0-9]+", s.lower())

    def _compute_idf(self):
        from collections import Counter as C
        N = max(1, len(self.docs))
        df = C()
        for d in self.docs:
            for t in set(self._tok(d)):
                df[t] += 1
        idf = {}
        for t, dfi in df.items():
            idf[t] = math.log((N - dfi + 0.5) / (dfi + 0.5) + 1.0)
        self.idf = idf

    def _score(self, q: str, d: str, k1: float = 1.2, b: float = 0.75) -> float:
        q_toks = self._tok(q)
        d_toks = self._tok(d)
        tf = Counter(d_toks)
        dl = len(d_toks) or 1
        avgdl = max(1.0, sum(len(self._tok(x)) for x in self.docs) / max(1, len(self.docs))) if self.docs else 1.0
        score = 0.0
        for t in q_toks:
            if t not in self.idf:
                continue
            fi = tf.get(t, 0)
            denom = fi + k1 * (1 - b + b * (dl / avgdl))
            score += self.idf[t] * (fi * (k1 + 1)) / (denom + 1e-8)
        return score

    def retrieve(self, query: str, k: int = 3):
        if not self.docs:
            return []
        scored = [(i, self._score(query, d)) for i, d in enumerate(self.docs)]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [{"text": self.docs[i], "score": s} for i, s in scored[:k]]

def arithmetic_sanity(prefix: str) -> bool:
    if not prefix:
        return True
    eqs = re.findall(r"(\d+)\s*([\+\-\*/x×])\s*(\d+)\s*=\s*(\-?\d+)", prefix)
    for a, op, b, c in eqs:
        a, b, c = int(a), int(b), int(c)
        if op in ["x", "×"]:
            op = "*"
        try:
            ok = (a + b == c) if op == "+" else \
                 (a - b == c) if op == "-" else \
                 (a * b == c) if op == "*" else \
                 (b != 0 and a // b == c and a % b == 0) if op == "/" else True
        except Exception:
            ok = True
        if not ok:
            return False
    return True

async def llm_json_score(client: AsyncOpenAI, model: str, system: str, user: str, max_tokens: int = 128, temperature: float = 0.0) -> Dict[str, Any]:
    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=[{"role":"system","content":system},{"role":"user","content":user}],
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type":"json_object"}
        )
        content = resp.choices[0].message.content or ""
        try:
            data = json.loads(content)
        except Exception:
            m = re.search(r"\{.*\}", content, flags=re.S)
            data = json.loads(m.group(0)) if m else {}
        return data
    except Exception:
        return {}

@dataclass
class PrefixRecord:
    text: str
    coherence: float
    verified: float
    arith_ok: bool
    corrected_text: Optional[str] = None
    rationales: Dict[str, str] = field(default_factory=dict)

async def filter_and_correct_prefix(client: AsyncOpenAI, model: str, question: str, prefix: str, retrieved: List[Dict[str, Any]]):
    sys1 = "You are a strict judge. Score how coherent and on-topic the given reasoning prefix is for the question. Return JSON with fields: coherence (0-1, float), reason (short)."
    user1 = f"Question:\n{question}\n\nPrefix:\n{prefix}\n\nReturn JSON only."
    ev_txt = "\n\n".join([f"[Doc {i+1} | score={doc.get('score',0):.3f}]\n{doc.get('text','')}" for i, doc in enumerate(retrieved)]).strip()
    sys2 = "You verify factual claims using the provided evidence snippets. Return JSON with fields: verified (0-1), verdict ('pass'|'fail'|'fixable'), rationale (short), correction (string or null)."
    user2 = f"Question:\n{question}\n\nPrefix:\n{prefix}\n\nEvidence:\n{ev_txt if ev_txt else '(no evidence)'}\n\nReturn JSON only."
    score_task = llm_json_score(client, model, sys1, user1)
    ver_task = llm_json_score(client, model, sys2, user2)
    score, ver = await asyncio.gather(score_task, ver_task)
    coherence = float(score.get("coherence", 0.0))
    r1 = str(score.get("reason", ""))
    verified = float(ver.get("verified", 0.0))
    correction = ver.get("correction", None)
    r2 = str(ver.get("rationale", ""))
    a_ok = arithmetic_sanity(prefix if correction is None else str(correction))
    return PrefixRecord(
        text=prefix, coherence=coherence, verified=verified, arith_ok=a_ok,
        corrected_text=str(correction) if correction else None,
        rationales={"coherence_reason": r1, "verify_rationale": r2}
    )

@dataclass
class GEESCConfig:
    temperature: float = 0.7
    top_p: float = 0.8
    presence_penalty: float = 1.0
    prefix_tokens: int = 96
    complete_tokens: int = 256
    max_paths: int = 8
    early_accept_k: int = 2
    good_coherence: float = 0.6
    good_verified: float = 0.4
    require_arith_ok: bool = False
    corpus_path: Optional[str] = None
    retrieve_k: int = 3
    log_dir: str = "logs"
    results_dir: str = "results"

def build_retriever(conf: GEESCConfig):
    if conf.corpus_path and os.path.exists(conf.corpus_path):
        return FileCorpusRetriever(conf.corpus_path)
    return NullRetriever()

async def gen_prefix(client: AsyncOpenAI, model: str, question: str, conf: GEESCConfig):
    prompt = (
        f"{question}\n\n"
        "Write only the FIRST concise step of a correct solution.\n"
        "- Focus on the key first calculation or deduction.\n"
        "- Do not give the final answer.\n"
        "- Start your line with 'Step 1:' and keep it under 2 sentences."
    )
    resp = await client.chat.completions.create(
        model=model,
        messages=[{"role":"user","content":prompt}],
        temperature=conf.temperature,
        top_p=conf.top_p,
        presence_penalty=conf.presence_penalty,
        max_tokens=conf.prefix_tokens,
    )
    prefix = resp.choices[0].message.content.strip()
    usage = getattr(resp, "usage", None)
    tok = {"prompt_tokens":0,"completion_tokens":0,"total_tokens":0}
    if usage is not None:
        tok = {
            "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
            "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
            "total_tokens": getattr(usage, "total_tokens", 0) or 0,
        }
    return prefix, tok

async def continue_to_answer(client: AsyncOpenAI, model: str, question: str, prefix: str, conf: GEESCConfig):
    prompt = (
        f"Question:\n{question}\n\n"
        f"You already wrote the first step:\n{prefix}\n\n"
        "Now continue the reasoning step by step and provide a final answer.\n"
        "Format your last line exactly as: Final Answer: <answer>\n"
        "Do not include anything after the Final Answer line."
    )
    resp = await client.chat.completions.create(
        model=model,
        messages=[{"role":"user","content":prompt}],
        temperature=conf.temperature,
        top_p=conf.top_p,
        presence_penalty=conf.presence_penalty,
        max_tokens=conf.complete_tokens,
    )
    content = resp.choices[0].message.content.strip()
    usage = getattr(resp, "usage", None)
    tok = {"prompt_tokens":0,"completion_tokens":0,"total_tokens":0}
    if usage is not None:
        tok = {
            "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
            "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
            "total_tokens": getattr(usage, "total_tokens", 0) or 0,
        }
    return content, tok

def extract_final_answer_for_dataset(dataset: str, content: str) -> str:
    if dataset == "gsm8k":
        return extract_gsm8k_answer(content)
    return normalize_ans(extract_math_answer(content))

@dataclass
class GEESCOutputs:
    question: str
    prefixes: List[PrefixRecord] = field(default_factory=list)
    kept_indices: List[int] = field(default_factory=list)
    completions: List[str] = field(default_factory=list)
    final_answers: List[str] = field(default_factory=list)
    votes: Dict[str, int] = field(default_factory=dict)
    majority: str = ""
    gt_answer: Optional[str] = None
    correct: Optional[bool] = None
    token_stats: Dict[str, int] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

async def geesc_single(client: AsyncOpenAI, model: str, dataset: str, sample: Dict[str, Any], conf: GEESCConfig, retriever):
    q = sample["question"]
    gt = sample.get("gt_answer")
    outputs = GEESCOutputs(question=q, gt_answer=gt, meta={"timestamp": now_iso()})
    total_prompt_tok, total_comp_tok, total_tok = 0, 0, 0
    prefix_tasks = [gen_prefix(client, model, q, conf) for _ in range(conf.max_paths)]
    prefix_results = await asyncio.gather(*prefix_tasks)
    
    generated_prefixes = []
    for px, tok in prefix_results:
        generated_prefixes.append(px)
        total_prompt_tok += tok.get("prompt_tokens", 0)
        total_comp_tok += tok.get("completion_tokens", 0)
        total_tok += tok.get("total_tokens", 0)
    docs = retriever.retrieve(q, k=conf.retrieve_k)
    filter_tasks = [filter_and_correct_prefix(client, model, q, px, docs) for px in generated_prefixes]
    prefixes = await asyncio.gather(*filter_tasks)
    kept = []
    for i, rec in enumerate(prefixes):
        good = (rec.coherence >= conf.good_coherence) and (rec.verified >= conf.good_verified)
        if conf.require_arith_ok:
            good = good and rec.arith_ok
        if good:
            kept.append(i)
        if len(kept) >= conf.early_accept_k:
            break
    if not kept and prefixes:
        best_idx = max(range(len(prefixes)), key=lambda j: prefixes[j].coherence * (0.5 + 0.5*prefixes[j].verified))
        kept = [best_idx]
    completions, answers = [], []
    if kept:
        continue_tasks = []
        for idx in kept:
            eff = prefixes[idx].corrected_text if prefixes[idx].corrected_text else prefixes[idx].text
            continue_tasks.append(async_continue_to_answer(client, model, q, eff, conf))
        continue_results = await asyncio.gather(*continue_tasks)
        for comp, tok2 in continue_results:
            completions.append(comp)
            answers.append(extract_final_answer_for_dataset(dataset, comp))
            total_prompt_tok += tok2.get("prompt_tokens", 0)
            total_comp_tok += tok2.get("completion_tokens", 0)
            total_tok += tok2.get("total_tokens", 0)
    maj, votes = majority_vote(answers)
    correct = is_equiv(str(maj), str(gt)) if gt is not None else None
    outputs.prefixes = prefixes; outputs.kept_indices = kept
    outputs.completions = completions; outputs.final_answers = answers
    outputs.votes = votes; outputs.majority = maj; outputs.correct = correct
    outputs.token_stats = {"prompt_tokens": total_prompt_tok, "completion_tokens": total_comp_tok, "total_tokens": total_tok}
    return outputs

async def process_sample(sample, client, model, dataset, conf, retriever, pbar, semaphore):
    async with semaphore:
        try:
            result = await geesc_single(client, model, dataset, sample, conf, retriever)
        except Exception as e:
            logging.error(f"Error processing question: {sample.get('question', 'N/A')}. Error: {e}")
            result = GEESCOutputs(question=sample.get('question', 'N/A'), gt_answer=sample.get('gt_answer'))
            result.meta["error"] = str(e)
        pbar.update(1)
        return result

async def run():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--base_url", required=True)
    parser.add_argument("--api_key", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--prefix_tokens", type=int, default=96)
    parser.add_argument("--complete_tokens", type=int, default=256)
    parser.add_argument("--max_paths", type=int, default=8)
    parser.add_argument("--early_accept_k", type=int, default=2)
    parser.add_argument("--good_coherence", type=float, default=0.6)
    parser.add_argument("--good_verified", type=float, default=0.4)
    parser.add_argument("--require_arith_ok", action="store_true")
    parser.add_argument("--corpus_path", default=None, help="路径指向 .jsonl 或 .txt，单行一文档；jsonl 需包含字段 text。")
    parser.add_argument("--retrieve_k", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--presence_penalty", type=float, default=1.0)
    parser.add_argument("--results_prefix", default=None)
    parser.add_argument("--concurrency", type=int, default=8, help="并发请求数量")
    args = parser.parse_args() 
    conf = GEESCConfig(
        temperature=args.temperature, top_p=args.top_p, presence_penalty=args.presence_penalty,
        prefix_tokens=args.prefix_tokens, complete_tokens=args.complete_tokens, max_paths=args.max_paths,
        early_accept_k=args.early_accept_k, good_coherence=args.good_coherence,
        good_verified=args.good_verified, require_arith_ok=args.require_arith_ok,
        corpus_path=args.corpus_path, retrieve_k=args.retrieve_k,
    )
    client = build_async_client(args.base_url, args.api_key)
    retriever = build_retriever(conf)
    if args.dataset == "gsm8k": records = load_gsm8k(args.dataset_path)
    elif args.dataset == "math": records = load_math(args.dataset_path)
    dataset = GEESCDataset(args.dataset, records)
    results_prefix = args.results_prefix or f"results/geesc_{args.model}_{args.dataset}"
    content_path = f"{results_prefix}_content.jsonl"; token_path = f"{results_prefix}_token.txt"
    ensure_dir(content_path); ensure_dir(token_path)
    logging.basicConfig(filename=f'logs/geesc_{args.model}_{args.dataset}.log', level=logging.INFO, encoding='utf-8')
    logging.info("="*200)
    semaphore = asyncio.Semaphore(args.concurrency)
    pbar = tqdm(total=len(dataset.records), desc=f"Processing {args.dataset}")
    tasks = [process_sample(sample, client, args.model, args.dataset, conf, retriever, pbar, semaphore) for sample in dataset.records]
    all_outputs = await asyncio.gather(*tasks)
    pbar.close()
    correct, total = 0, 0
    P_all, C_all, T_all = [], [], []
    for out in all_outputs:
        if "error" in out.meta: continue
        total += 1
        if out.correct is True: correct += 1
        with open(content_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(out), ensure_ascii=False) + "\n")
        P = out.token_stats.get("prompt_tokens", 0)
        C = out.token_stats.get("completion_tokens", 0)
        T = out.token_stats.get("total_tokens", 0)
        P_all.append(P); C_all.append(C); T_all.append(T)
        with open(token_path, "a", encoding="utf-8") as f: f.write(f"{P}, {C}, {T}\n")
        logging.info(f"Q: {out.question}\n" + "-"*100 + f"\nKept: {out.kept_indices} | Votes: {out.votes} | Majority: {out.majority} | GT: {out.gt_answer} | Correct: {out.correct}\n" + f"Token: Prompt={P} | Completion={C} | Total={T}\n" + "="*200)
    def mean(xs): return sum(xs)/max(1, len(xs))
    acc = (correct / max(1, total)) * 100.0
    print(f"Accuracy on {args.dataset}: {acc:.2f}%  ({correct}/{total})")
    print(f"Avg Prompt Tokens per Q:     {mean(P_all):.2f}")
    print(f"Avg Completion Tokens per Q: {mean(C_all):.2f}")
    print(f"Avg Total Tokens per Q:      {mean(T_all):.2f}")

if __name__ == "__main__":
    asyncio.run(run())