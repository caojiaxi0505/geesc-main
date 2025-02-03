import json
import re
#from openai import OpenAI
from langchain_openai import ChatOpenAI, OpenAI
from typing import List
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate

LLM_model = OpenAI(model_name="gpt-3.5-turbo-instruct", openai_api_key="", max_tokens=-1)

dataset_path = ''


def LLM_response(content, model):
    chat_completion = LLM_model.invoke(content)
    return chat_completion

def load_dataset(path):
    with open(path, 'r', encoding='utf-8') as file:
        dataset = json.load(file)
    return dataset

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

##################################################################################################################
##################################################################################################################
##################################################################################################################
##################################################################################################################

# Define your desired data structure.
class extract(BaseModel):
    propositions: str = Field(description="propositions in the format: A:*, B:*, etc.")
    expressions: str = Field(description="expressions in the format: A->B, B->¬C, etc.")
extract_manual_zero = """
Please use uppercase English letters such as A, B, C, etc. to identify all possible propositions. Do not include negative tones such as "not" in the propositions. For example, if the sentence is "It is not bored," you should use "A: bored" to represent it.
Next, for each proposition, use the symbol to represent its negative form. For example, the negative form of proposition A can be expressed as ¬A.
Now, please carefully analyze the context and find causal relationship between propositions. A causal expression is only established when the context directly supports this relationship. Use arrows (->) to indicate causal relationships, for example, "If A, then B", "B if A" and "A causes B" etc. can be represented as A->B.
Finally, output propositions and causal expressions.
"""

# Set up a parser + inject instructions into the prompt template.
extract_prompt_zero_parser = JsonOutputParser(pydantic_object=extract)

extract_prompt_zero = PromptTemplate(
    template="{extract_manual_zero_prompt}\n{format_instructions}\ncontext:{context}\n",
    input_variables=["context"],
    partial_variables={"format_instructions": extract_prompt_zero_parser.get_format_instructions(), "extract_manual_zero_prompt": extract_manual_zero},
)

extract_examples = [
    {
        "context": """
     If you have no keyboarding skills at all, you will not be able to use a computer. 
     And if you are not able to use a computer, you will not be able to write your essays using a word processing program.
     """,
        "logical_extract": """
     propositions:
     A: You have keyboarding skills;
     B: be able to use a computer;
     C: be able to write your essays using a word processing program;
     expressions: ¬A->¬B, ¬B->¬C;
    """,
    },
]

extract_example_prompt = PromptTemplate(
    input_variables=["context", "logical_extract"], template="context: {context}\nlogical_extract:{logical_extract}"
)

# print(extract_example_prompt.format(**extract_examples[2]))
extract_manual_few = """
    Please parse all propositions in the context, with each proposition represented by a letter(A, B, C, ...).
    The negation of a proposition can be represented by ¬, for example, the negation of A Is ¬A.
    Then find possible causal relationships between propositions in context, represent them as expressions, for example, represent "If A, then B" as "A->B".
    If there is no causal relationship, there is no need to form an expression.
    Finally output all propositions and expressions in the Context.
"""

extract_prompt_few = FewShotPromptTemplate(
    examples=extract_examples,
    example_prompt=extract_example_prompt,
    suffix="context: {context}\nlogical_extract:",
    input_variables=["context"],
)

extract_filter_expression = """
Please only output the expressions:
"""

extract_filter_proposition = """
Please only output all propositions and their meanings:
"""

def filter_expression(expressions, model):
    content = "expressions: " + expressions + """
    Identify all expressions separated by comma that meet the form of A->B. 
    Only output the expressions that strictly meet the form requirements!
    If there is no expression that meets the requirements, output none.
    """
    return LLM_response(content, model)

def filter_expression_code(text):
    expressions = []
    parts = re.split(r',|;', str(text))
    valid_parts = [part for part in parts if '^' not in part]
    valid_parts = [part for part in parts if '&' not in valid_parts]
    valid_parts = [part for part in parts if '∧' not in valid_parts]
    valid_parts = [part for part in parts if '(' not in valid_parts]
    valid_parts = [part for part in valid_parts if len(re.findall(r'[A-Z]', part)) < 3]
    valid_parts = [part for part in valid_parts if part.count('¬') < 3]

    for part in valid_parts:
        if '->' in part:
            matches = re.findall(r'(\b(¬?[a-zA-Z])->(¬?[a-zA-Z])\b)', part)
            for match in matches:
                expressions.append(part)

    if len(expressions) == 0:
        return "none"
    else:
        return ",".join(expressions)
##################################################################################################################
##################################################################################################################
##################################################################################################################
##################################################################################################################


negation_zero_prompt = """
Please use the provided propositions to translate each expression into a complete sentence. 
¬A represents the negation of proposition A, the arrow (->) represents the causal relationship, and A->B represents if A, then B.
Only output the sentences in a paragraph! 
"""
negation_examples = [
    {
     "propositions":"""
     A: You have keyboarding skills;
     B: be able to use a computer;
     C: be able to write your essays using a word processing program;
     """,
     "expressions": """
     ¬A->¬B, ¬B->¬C;
     """,
     "negation":"""
    If you do not have keyboarding skills, then you will not be able to use a computer; 
    If you are not able to use a computer, then you will not be able to write your essays using a word processing program.
     """,
    },
    {
     "propositions":"""
    A: economic benefits from forests, mountains, or wetlands that no longer exist;
    B: nature has intrinsic value;
    C: destroy features of the environment;
     """,
     "expressions": """
     ¬A->¬C, B->¬C;
     """,
     "negation":"""
    If there are no economic benefits from forests, mountains, or wetlands that no longer exist, then features of the environment will not be destroyed; 
    If nature has intrinsic value, then features of the environment will not be destroyed.
     """,
    },
    {
     "propositions": """
      A: Paula will visit the dentist tomorrow morning, 
      B: Bill goes golfing in the morning, 
      C: Damien agrees to go golfing, 
     expressions: 
     """,
     "expressions": """
     B->A, C->B;
     """,
     "negation": """
     If Bill goes golfing in the morning, then Paula will visit the dentist tomorrow morning.
     if Damien agrees to go golfing, then Bill goes golfing in the morning.
    """,
    },

]

negation_example_prompt = PromptTemplate(
    input_variables=["propositions", "expressions", "negation"], template="propositions: {propositions}\nexpressions:{expressions}\nnegation:{negation}"
)

negation_manual_few = """
Please use the provided propositions to translate each expression into a complete sentence. Out put all sentences in a paragraph.
¬A represents the negation of proposition A, the arrow (->) represents the causal relationship, and A->B represents if A, then B.

"""
negation_prompt_few = FewShotPromptTemplate(
    examples=negation_examples,
    example_prompt=negation_example_prompt,
    suffix="{negation_manual_few}\npropositions: {propositions}\nexpressions:{expressions}\nnegation:",
    input_variables=["propositions", "expressions"],
    partial_variables={"negation_manual_few": negation_manual_few},
)


