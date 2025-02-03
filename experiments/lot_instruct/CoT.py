from openai import OpenAI
import json
import re
import numpy as np
import logging
model = client = OpenAI(api_key="")
dataset_path = ''


def LLM_response(content):
    chat_completion = model.completions.create(
    prompt=content,
    model="gpt-3.5-turbo-instruct",
    max_tokens=1000
    )
    return chat_completion.choices[0].text

def format_answers(answers):
    formatted_answers = ""
    for i, answer in enumerate(answers):
        formatted_answers += f"{chr(65 + i)}. {answer} \n"
    return formatted_answers.strip()

def extract_answer(answer):
    match = re.search(r'[A-D]', answer)
    if match:
        return ord(match.group()) - 65
    else:
        return 4

def test(path):
    with open(path, 'r', encoding='utf-8') as file:
        dataset = json.load(file)
    answer = np.zeros(len(dataset))
    count = 0
    logging.basicConfig(filename='cot.log', level=logging.INFO)

    dataset = dataset[count:]
    for sample in dataset:
        
        context = sample['context']
        question = sample['question']
        option = format_answers(sample['options'])
        content = f"Context: {context}\nQuestion: {question}\nOptions:\n{option}\nChoose the most suitable option, let's think step by step: "
        answer_res = LLM_response(content)
        response = LLM_response(answer_res + "\n" + "Please output the symbol of the answer A/B/C/D: ")
        answer[count] = int(extract_answer(response))
        print(count)
        print(content)
        print(answer_res)
        print(response)
        print(answer)
        logging.info(f"count: {count}")
        logging.info(f"content: {content}")
        logging.info(f"answer_res: {answer_res}")
        logging.info(f"response: {response}")
        logging.info(f"answer: {answer}")
        count += 1
    np.save("cot.npy", answer)
        
        
def main():
    test(dataset_path)

if __name__ == '__main__':
    main()