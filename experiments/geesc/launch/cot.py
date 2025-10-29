import json
import argparse
import logging
import numpy as np
from tqdm import tqdm
from openai import OpenAI

from experiments.geesc.utils.extract_answer import extract_gsm8k_answer
from experiments.geesc.utils.dataset_loader import GEESCDataset, load_gsm8k, load_math, load_logiqa, load_reclor, load_ruletaker


def build_client(base_url, api_key):
    return OpenAI(
        base_url=base_url,
        api_key=api_key
    )


def reasoning(client, model, context, temperature=0.7, top_p=0.8, top_k=20, presence_penalty=1.5, max_tokens=None):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": context}
        ],
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        presence_penalty=presence_penalty,
        extra_body={
            "top_k": top_k,
            "chat_template_kwargs": {"enable_thinking": False},
        }
    )
    return response


def result_math(client, model, context, reasoning, temperature=0.7, top_p=0.8, top_k=20, presence_penalty=1.5, max_tokens=None):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": context},
            {"role": "assistant", "content": reasoning},
            {"role": "user", "content": "Directly output the final answer."}
        ],
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        presence_penalty=presence_penalty,
        extra_body={
            "top_k": top_k,
            "chat_template_kwargs": {"enable_thinking": False},
        }
    )
    return response


def inference_gsm8k(client, sample, temperature=0.7, top_p=0.8, top_k=20, presence_penalty=1.5, max_tokens=None):
    question = sample['question']
    context = f"{question}\nLet's think step by step."
    reasoning_result = reasoning(client, args.model, context, temperature, top_p, top_k, presence_penalty, max_tokens)
    result = result_math(client, args.model, context, reasoning_result.choices[0].message.content, temperature, top_p, top_k, presence_penalty, max_tokens)
    return reasoning_result, result


def inference_math(client, sample, temperature=0.7, top_p=0.8, top_k=20, presence_penalty=1.5, max_tokens=None):
    question = sample['question']
    context = f"{question}\nLet's think step by step."
    reasoning_result = reasoning(client, args.model, context, temperature, top_p, top_k, presence_penalty, max_tokens)
    result = result_math(client, args.model, context, reasoning_result.choices[0].message.content, temperature, top_p, top_k, presence_penalty, max_tokens)
    return reasoning_result, result


# def inference_logiqa(client, sample, temperature=0.7, top_p=0.8, top_k=20, presence_penalty=1.5, max_tokens=None):
#     context = sample['context']
#     question = sample['question']
#     choices = sample['choices']
#     options = ""
#     for i, choice in enumerate(choices):
#         options += f"{chr(65 + i)}. {choice} \n"
#     prompt = f"Context: {context}\nQuestion: {question}\nChoices:\n{options}\nDirectly output the symbol of the most suitable option A/B/C/D:"
#     return inference(client, args.model, prompt, temperature, top_p, top_k, presence_penalty, max_tokens)


# def inference_reclor(client, sample, temperature=0.7, top_p=0.8, top_k=20, presence_penalty=1.5, max_tokens=None):
#     context = sample['context']
#     question = sample['question']
#     choices = sample['choices']
#     options = ""
#     for i, choice in enumerate(choices):
#         options += f"{chr(65 + i)}. {choice} \n"
#     prompt = f"Context: {context}\nQuestion: {question}\nChoices:\n{options}\nDirectly output the symbol of the most suitable option A/B/C/D:"
#     return inference(client, args.model, prompt, temperature, top_p, top_k, presence_penalty, max_tokens)


# def inference_ruletaker(client, sample, temperature=0.7, top_p=0.8, top_k=20, presence_penalty=1.5, max_tokens=None):
#     context = sample['context']
#     conclusion = sample['conclusion']
#     prompt = f"Context: {context}\nConclusion:\n{conclusion}\nDirectly give a judgment on whether the conclusion is correct: output 1 for correct, 0 for incorrect; output only 0 or 1:"
#     return inference(client, args.model, prompt, temperature, top_p, top_k, presence_penalty, max_tokens)


def run():
    client = build_client(args.base_url, args.api_key)
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
    logging.basicConfig(filename=f'logs/cot_{args.model}_{args.dataset}.log', level=logging.INFO, encoding='utf-8')
    logging.info("="*200)
    for sample in tqdm(dataset.records, total=len(dataset.records), desc=f"{args.model} · {args.dataset}"):
        if args.dataset == "gsm8k":
            reasoning_result, response = inference_gsm8k(client, sample)
        elif args.dataset == "math":
            reasoning_result, response = inference_math(client, sample, max_tokens=args.max_tokens)
        # elif args.dataset == "logiqa":
        #     response = inference_logiqa(client, sample)
        # elif args.dataset == "reclor":
        #     response = inference_reclor(client, sample)
        # elif args.dataset == "ruletaker":
        #     response = inference_ruletaker(client, sample)
        logging.info(f"Question:\n{sample.get('question')}")
        logging.info("-"*200)
        logging.info(f"Model Response:\n{response.choices[0].message.content}")
        logging.info("-"*200)
        logging.info(f"Token:\nPrompt Token: {response.usage.prompt_tokens} | Completion Token: {response.usage.completion_tokens} | Total Token: {response.usage.total_tokens}")
        logging.info("="*200)
        if args.dataset == "gsm8k":
            extracted = extract_gsm8k_answer(response.choices[0].message.content)
            with open(f"results/cot_{args.model}_{args.dataset}_reasoning.txt", "a", encoding="utf-8") as fout:
                fout.write(f"{reasoning_result.choices[0].message.content}\n")
            with open(f"results/cot_{args.model}_{args.dataset}.txt", "a", encoding="utf-8") as fout:
                fout.write(f"{extracted}\n")
            with open(f"results/cot_{args.model}_{args.dataset}_token.txt", "a", encoding="utf-8") as fout:
                fout.write(f"{response.usage.prompt_tokens+reasoning_result.usage.prompt_tokens}, {response.usage.completion_tokens+reasoning_result.usage.completion_tokens}, {response.usage.total_tokens+reasoning_result.usage.total_tokens}\n")
        if args.dataset == "math":
            with open(f"results/cot_{args.model}_{args.dataset}_content.jsonl", "a", encoding="utf-8") as fout:
                fout.write(json.dumps({
                    "question": sample.get("question"),
                    "reasoning": reasoning_result.choices[0].message.content,
                    "content": response.choices[0].message.content
                }, ensure_ascii=False) + "\n")
            with open(f"results/cot_{args.model}_{args.dataset}_token.txt", "a", encoding="utf-8") as fout:
                fout.write(f"{response.usage.prompt_tokens+reasoning_result.usage.prompt_tokens}, {response.usage.completion_tokens+reasoning_result.usage.completion_tokens}, {response.usage.total_tokens+reasoning_result.usage.total_tokens}\n")



# qwen3-8b
# python -m experiments.geesc.launch.cot --base_url http://localhost:8000/v1 --api_key any --model qwen3-8b --dataset gsm8k --dataset_path datasets/gsm8k/test.jsonl
# python -m experiments.geesc.launch.cot --base_url http://localhost:8000/v1 --api_key any --model qwen3-8b --dataset math --dataset_path datasets/math500/test.jsonl --max_tokens 1024


# qwen3-4b
# python -m experiments.geesc.launch.cot --base_url http://localhost:8001/v1 --api_key any --model qwen3-4b --dataset gsm8k --dataset_path datasets/gsm8k/test.jsonl
# python -m experiments.geesc.launch.cot --base_url http://localhost:8001/v1 --api_key any --model qwen3-4b --dataset math --dataset_path datasets/math500/test.jsonl --max_tokens 1024


# qwen3-1.7b
# python -m experiments.geesc.launch.cot --base_url http://localhost:8002/v1 --api_key any --model qwen3-1.7b --dataset gsm8k --dataset_path datasets/gsm8k/test.jsonl
# python -m experiments.geesc.launch.cot --base_url http://localhost:8002/v1 --api_key any --model qwen3-1.7b --dataset math --dataset_path datasets/math500/test.jsonl --max_tokens 1024

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--base_url", required=True)
    parser.add_argument("--api_key", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--max_tokens", type=int, default=None)
    args = parser.parse_args()
    run()
