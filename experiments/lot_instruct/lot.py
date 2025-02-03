from extend import Logic_extend
from extract import Logic_extract
from negation import Logic_negation
from manual import load_dataset, format_answers, dataset_path, LLM_model, LLM_response, extract_answer
import numpy as np
import json
import logging
def context_extend(context, model):
    context1 = LLM_response("Context:"+ context+"\n"+"Please pick out sentences with conditional relationships from the context:", model)
    print("context1: ", context1)
    logging.info(f"context1: {context1}")
    propositions, expression_extracted = Logic_extract(context1, model)
    logging.info(f"propositions: {propositions}")
    logging.info(f"expression_extracted: {expression_extracted}")
    if "none" in expression_extracted.lower():
        return context
    expression_extended = Logic_extend(expression_extracted)
    logging.info(f"expression_extended: {expression_extended}")
    context_extended = Logic_negation(context, propositions, expression_extended, model)

    return context_extended


def Logic_Reasoner(context, question, option, model):
    context_extended = context_extend(context, model)
    content = f"Context: {context_extended}\nQuestion: {question}\noptions: \n{option}\nDirectly output the symbol of the most suitable option A/B/C/D: "
    print(content)
    response = LLM_response(content, model)
    print(response)
    logging.info(f"context: {context}")
    logging.info(f"content: {content}")
    logging.info(f"response: {response}")
    return response


def run(path, model):
    dataset = load_dataset(path)
    answer = np.zeros(len(dataset))
    count = 0
    logging.basicConfig(filename='logic.log', level=logging.INFO, encoding='utf-8')
    dataset = dataset[count:]
    for sample in dataset:
        context = sample['context']
        question = sample['question']
        option = format_answers(sample['options'])
        answer_res = Logic_Reasoner(context, question, option, model)
        answer[count] = int(extract_answer(answer_res))
        print(count)
        print(answer[count])
        print(answer)
        logging.info(f"count: {count}")
        logging.info(f"answer: {answer[count]}")
        logging.info(f"answers: {answer}")
        count += 1
    np.save("logic.npy", answer)

if __name__ == '__main__':
    run(dataset_path, LLM_model)

