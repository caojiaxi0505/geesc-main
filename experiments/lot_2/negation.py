from manual import LLM_response, negation_zero_prompt, negation_prompt_few


def zero_negation(context, propositions, expressions, model):
    content = f"propositions:{propositions}\nexpressions:{expressions}\n{negation_zero_prompt}"
    response = LLM_response(content, model)
    context_extended = f"{context}{response}"
    return context_extended

def few_negation(context, propositions, expressions, model):
    answer = model.invoke(negation_prompt_few.format(propositions=propositions, expressions=expressions)).content
    context_extended = f"{context}{answer}"
    return context_extended

def Logic_negation(context, propositions, expression_extended, model):
    if expression_extended:
        context_extended = zero_negation(context, propositions, expression_extended, model)
    else:
        context_extended = context
    return context_extended
