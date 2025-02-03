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
    logging.basicConfig(filename='direct.log', level=logging.INFO, encoding='utf-8')
    dataset = dataset[count:]
    for sample in dataset:
        context = sample['context']
        question = sample['question']
        option = format_answers(sample['options'])
        content = f"Context: {context}\nQuestion: {question}\nOptions:\n{option}\nDirectly output the symbol of the most suitable option A/B/C/D:"
        answer_res = LLM_response(content)
        answer[count] = int(extract_answer(answer_res))
        print(count)
        print(content)
        print(answer_res)
        print(answer)
        logging.info(f"count: {count}")
        logging.info(f"content: {content}")
        logging.info(f"answer_res: {answer_res}")
        logging.info(f"answer: {answer}")
        count += 1
    np.save("direct.npy", answer)
        
        
def main():
    test(dataset_path)

if __name__ == '__main__':
    main()