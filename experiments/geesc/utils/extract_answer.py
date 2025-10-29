import re


def extract_gsm8k_answer(s: str) -> str:
    """
    从文本中提取出“Final Answer”部分的数值，返回字符串形式。
    自动去掉末尾的点、无意义的小数（如 .00）、千分位逗号等。
    """
    # 优先匹配 Final Answer 后的数字
    match = re.search(r'Final Answer[^0-9]*([0-9][0-9,\.]*)', s, flags=re.IGNORECASE)
    if match:
        val = match.group(1)
    else:
        nums = re.findall(r'([0-9][0-9,\.]*)', s)
        if not nums:
            return ''
        val = nums[-1]
    # 去掉千分位逗号
    val = val.replace(',', '')
    # 去掉结尾的点（例如 "694." -> "694"）
    if val.endswith('.'):
        val = val[:-1]
    # 去掉无意义的小数（如 16.00 -> 16, 42.0 -> 42）
    if re.fullmatch(r'\d+\.(0+)$', val):
        val = val.split('.')[0]
    # 去掉末尾的多余零（如 12.50 -> 12.5）
    val = re.sub(r'(\.\d*?)0+$', r'\1', val)
    # 如果最后仍是类似 "12." 这种形式，去掉最后的点
    if val.endswith('.'):
        val = val[:-1]
    return val