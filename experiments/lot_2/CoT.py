from openai import OpenAI
import json
import re
import numpy as np
import logging
model = client = OpenAI(api_key="")



dataset_path = ''


def LLM_response(content):
    chat_completion = model.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": content,
        }
    ],
    model="gpt-4",
    )
    return chat_completion.choices[0].message.content


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
        context = sample['premises']
        question = "Based on the given context, determine if the following conclusion is true: "
        inference = sample['conclusion']
        content2 = "Let's think step by step: "
        content = f"Context: {context}\nQuestion: {question}\nConclusion: {inference}\n{content2}"
        answer_res = LLM_response(content)
        response = LLM_response(answer_res  + "\n" + "The conclusion: " + inference + ", is <True/False>:")
        if "true" in response.lower():
            answer[count] = 1
        else:
            answer[count] = 0
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