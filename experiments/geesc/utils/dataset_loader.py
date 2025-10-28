"""
支持gsm8k，math500，logiqa，reclor数据集的加载
"""


import re
import json
from typing import List, Dict, Any


class GEESCDataset:
    """GEESC数据集基类"""
    """
    所有数据集的问题都在.records[<idx>]['question']字段中
    所有数据集的答案都在.records[<idx>]['gt_answer']字段中
    对于选择题，所有选项都在.records[<idx>]['choices']字段中
    选择题和判断题（logiqa，reclor，ruletaker）都有字段'records[<idx>]['context']
    """
    def __init__(self, type, records: List[Dict[str, Any]]):
        self.type = type
        self.records = records
        if self.type == "gsm8k":
            self._update_records_gsm8k()
        elif self.type == "math":
            self._update_records_math()
        elif self.type == "logiqa":
            self._update_records_logiqa()
        elif self.type == "reclor":
            self._update_records_reclor()
        elif self.type == "ruletaker":
            self._update_records_ruletaker()

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.records[idx]
    
    def _update_records_gsm8k(self):
        for record in self.records:
            record['gt_answer'] = extract_gsm8k_gt(record['answer'])
    
    def _update_records_math(self):
        for record in self.records:
            record['gt_answer'] = record['answer']
            record['question'] = record['problem']

    def _update_records_logiqa(self):
        for record in self.records:
            record['gt_answer'] = record['answer']
            record['choices'] = process_logiqa_options(record['options'])

    def _update_records_reclor(self):
        for record in self.records:
            record['gt_answer'] = record['label']
    
    def _update_records_ruletaker(self):
        for record in self.records:
            record['gt_answer'] = record['answer']
            record['question'] = record['conclusion']


def load_gsm8k(filepath: str) -> List[Dict[str, Any]]:
    """加载GSM8K数据集（JSONL，每行一个JSON）"""
    records: List[Dict[str, Any]] = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for ln, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except Exception as e:
                    print(f"GSM8K解析失败：文件 {filepath} 第 {ln} 行：{e}")
        return records
    except Exception as e:
        print(f"加载GSM8K数据集失败: {e}")
        return []


def extract_gsm8k_gt(answer: str) -> str:
    """从GSM8K的answer字段中提取最终答案"""
    match = re.search(r"####\s*(.+)", answer)
    if match:
        return match.group(1).strip()
    return answer.strip()


def load_math(filepath: str) -> List[Dict[str, Any]]:
    """加载MATH数据集（JSONL，每行一个JSON）"""
    records: List[Dict[str, Any]] = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for ln, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except Exception as e:
                    print(f"MATH解析失败：文件 {filepath} 第 {ln} 行：{e}")
        return records
    except Exception as e:
        print(f"加载MATH数据集失败: {e}")
        return []


def load_logiqa(filepath: str) -> List[Dict[str, Any]]:
    """加载LogiQA数据集（JSON数组，整个文件为一个JSON list）"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except Exception as e:
                print(f"LOGIQA解析失败：文件 {filepath} 不是合法JSON：{e}")
                return []
        if isinstance(data, list):
            return data
        else:
            print(f"LOGIQA解析失败：文件 {filepath} 顶层不是数组。")
            return []
    except Exception as e:
        print(f"加载LOGIQA数据集失败: {e}")
        return []
    

def process_logiqa_options(options: str) -> List[str]:
    """将LogiQA的options多行字符串转为列表，并去除前缀选项标记（A./A)/A: 等）。"""
    if not isinstance(options, str):
        return []
    items: List[str] = []
    for line in options.splitlines():
        s = line.strip()
        if not s:
            continue
        s = re.sub(r'^\s*\(?\s*[A-Da-d](?:[\.\)）:：、\-]|(?=\s))\s*', '', s)
        items.append(s)
    return items    


def load_reclor(filepath: str) -> List[Dict[str, Any]]:
    """加载ReClor数据集（JSON数组，整个文件为一个JSON list）"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except Exception as e:
                print(f"RECLOR解析失败：文件 {filepath} 不是合法JSON：{e}")
                return []
        if isinstance(data, list):
            return data
        else:
            print(f"RECLOR解析失败：文件 {filepath} 顶层不是数组。")
            return []
    except Exception as e:
        print(f"加载RECLOR数据集失败: {e}")
        return []


def load_ruletaker(filepath: str) -> List[Dict[str, Any]]:
    """加载RuleTaker数据集（JSON数组，整个文件为一个JSON list）"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except Exception as e:
                print(f"RULETAKER解析失败：文件 {filepath} 不是合法JSON：{e}")
                return []
        if isinstance(data, list):
            return data
        else:
            print(f"RULETAKER解析失败：文件 {filepath} 顶层不是数组。")
            return []
    except Exception as e:
        print(f"加载RULETAKER数据集失败: {e}")
        return []


def module_test():
    gsm8k_data = load_gsm8k('datasets/gsm8k/test.jsonl')
    # gsm8k_data[0].keys()
    # dict_keys(['question', 'answer'])
    # gsm8k_data[0]['question']
    # "Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"
    # gsm8k_data[0]['answer']
    # 'Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.\nShe makes 9 * 2 = $<<9*2=18>>18 every day at the farmer’s market.\n#### 18'
    print(f"GSM8K数据集加载了 {len(gsm8k_data)} 条记录。")

    math_data = load_math('datasets/math500/test.jsonl')
    # math_data[0].keys()
    # dict_keys(['problem', 'solution', 'answer', 'subject', 'level', 'unique_id'])
    # math_data[0]['problem']
    # 'Convert the point $(0,3)$ in rectangular coordinates to polar coordinates.  Enter your answer in the form $(r,\\theta),$ where $r > 0$ and $0 \\le \\theta < 2 \\pi.$'
    # math_data[0]['solution']
    # 'We have that $r = \\sqrt{0^2 + 3^2} = 3.$  Also, if we draw the line connecting the origin and $(0,3),$ this line makes an angle of $\\frac{\\pi}{2}$ with the positive $x$-axis.\n\n[asy]\nunitsize(0.8 cm);\n\ndraw((-0.5,0)--(3.5,0));\ndraw((0,-0.5)--(0,3.5));\ndraw(arc((0,0),3,0,90),red,Arrow(6));\n\ndot((0,3), red);\nlabel("$(0,3)$", (0,3), W);\ndot((3,0), red);\n[/asy]\n\nTherefore, the polar coordinates are $\\boxed{\\left( 3, \\frac{\\pi}{2} \\right)}.$'
    # math_data[0]['answer']
    # '\\left( 3, \\frac{\\pi}{2} \\right)'
    # math_data[0]['level']
    # 2
    # math_data[0]['unique_id']
    # 'test/precalculus/807.json'
    print(f"MATH500数据集加载了 {len(math_data)} 条记录。")

    logiqa_data = load_logiqa('datasets/logicqa/json/text.json')
    # logiqa_data[0].keys()
    # dict_keys(['context', 'question', 'options', 'answer'])
    # logiqa_data[0]['context']
    # 'In the planning of a new district in a township, it was decided to build a special community in the southeast, northwest, centered on the citizen park.These four communities are designated as cultural area, leisure area, commercial area and administrative service area.It is known that the administrative service area is southwest of the cultural area, and the cultural area is southeast of the leisure area.'
    # logiqa_data[0]['question']
    # 'Based on the above statement, which of the following can be derived?'
    # logiqa_data[0]['options']
    # 'A.Civic Park is north of the administrative service area\nB.The leisure area is southwest of the cultural area\nC.The cultural district is in the northeast of the business district\nD.The business district is southeast of the leisure area'
    # logiqa_data[0]['answer']
    # 0
    print(f"LogiQA数据集加载了 {len(logiqa_data)} 条记录。")

    reclor_data = load_reclor('datasets/reclor_data/arlsat_test.json')
    # reclor_data[0].keys()
    # dict_keys(['context', 'question', 'choices', 'question_type', 'id', 'label'])
    # reclor_data[0]['context']
    # 'The nature of English literature reflects the rich and diverse vocabulary of the English language, which resulted from the dual influence of the Anglo-Saxon and, later, French languages. The French language, though, is a direct descendant of Latin, with few traces of the Celtic language spoken by the preRoman inhabitants of the area: the hallmark of French literature is its simplicity and clarity.'
    # reclor_data[0]['question']
    # 'Which one of the following can be most reasonably inferred from the information above?'
    # reclor_data[0]['choices']
    # ['Simple and clear literature cannot be written in a language with a rich and diverse vocabulary.', 'The origin of English played a role in shaping English literature.', 'The vocabulary of English is larger than the vocabulary of French.', 'The vocabulary of the Anglo-Saxon language was richer than that of the French language.']
    # reclor_data[0]['question_type']
    # 5
    # reclor_data[0]['id']
    # 'test_13'
    # reclor_data[0]['label']
    # 1
    print(f"ReClor数据集加载了 {len(reclor_data)} 条记录。")

    ruletaker_data = load_ruletaker('datasets/ruletaker/json/val.json')
    # ruletaker_data[0].keys()
    # dict_keys(['Qid', 'context', 'conclusion', 'answer'])
    # ruletaker_data[0]['Qid']
    # 'RelNeg-D5-736-3'
    # ruletaker_data[0]['context']
    # 'The cat sees the rabbit. If something does not visit the rabbit then it is big. The mouse does not visit the lion. The lion visits the rabbit. If something chases the cat then the cat sees the lion. If something is kind then it chases the cat. If something sees the cat then it is not kind. The lion is green. If something is green then it sees the rabbit. The mouse does not see the cat. The cat chases the rabbit. If something sees the lion and it is not blue then it is kind. The mouse visits the cat. If something is red and kind then it does not visit the cat. If something chases the rabbit and it sees the mouse then the mouse sees the lion. The cat visits the mouse. If the rabbit visits the mouse and the rabbit is big then the mouse visits the lion. The mouse sees the lion. The cat is red. The rabbit sees the cat.'
    # ruletaker_data[0]['conclusion']
    # 'The cat is big.'
    # ruletaker_data[0]['answer']
    # 1
    print(f"RuleTaker数据集加载了 {len(ruletaker_data)} 条记录。")

    gsm8k_dataset = GEESCDataset(type="gsm8k", records=gsm8k_data)
    math_dataset = GEESCDataset(type="math", records=math_data)
    logiqa_dataset = GEESCDataset(type="logiqa", records=logiqa_data)
    reclor_dataset = GEESCDataset(type="reclor", records=reclor_data)
    ruletaker_dataset = GEESCDataset(type="ruletaker", records=ruletaker_data)
    

if __name__ == "__main__":
    module_test()
