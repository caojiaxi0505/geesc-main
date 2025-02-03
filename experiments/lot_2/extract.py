from manual import extract_prompt_zero, extract_prompt_zero_parser, extract_prompt_few, filter_expression, \
    extract_filter_expression, extract_filter_proposition, LLM_response, filter_expression_code

def zero_extract(context, model):
    chain = extract_prompt_zero | model | extract_prompt_zero_parser
    while True:
        try:
            answer = chain.invoke({"context": context})
            print(answer['propositions'])
            print(answer['expressions'])
            break  
        except (TypeError, KeyError) as e:
            print(f"An exception occurred: {e}")

    return answer['propositions'], filter_expression_code(answer['expressions'])

def few_extract(context, model):
    few_answer = model.invoke(extract_prompt_few.format(context=context)).content
    expression_content = few_answer + "\n" + extract_filter_expression
    proposition_content = few_answer + "\n" + extract_filter_proposition
    expressions = filter_expression_code(LLM_response(expression_content, model))
    propositions = LLM_response(proposition_content, model)
    return propositions, expressions


def Logic_extract(context, model):
    return zero_extract(context, model)