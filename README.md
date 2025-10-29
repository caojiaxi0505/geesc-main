# 环境配置

```bash
# 创建虚拟环境
python3.10 -m venv .venv
# 激活虚拟环境
source .venv/bin/activate
# 安装环境
pip install -r requirements.txt
```

# 实验启动命令

## 下载gsm8k和math
```
python scripts/download_dataset.py --out datasets
```

## 进行GSM8K的基线实验
运行：
```
python -m experiments.geesc.launch.direct --base_url http://localhost:8000/v1 --api_key any --model qwen3-8b --dataset gsm8k --dataset_path datasets/gsm8k/test.jsonl
python -m experiments.geesc.launch.direct --base_url http://localhost:8001/v1 --api_key any --model qwen3-4b --dataset gsm8k --dataset_path datasets/gsm8k/test.jsonl
python -m experiments.geesc.launch.direct --base_url http://localhost:8002/v1 --api_key any --model qwen3-1.7b --dataset gsm8k --dataset_path datasets/gsm8k/test.jsonl
python -m experiments.geesc.launch.cot --base_url http://localhost:8000/v1 --api_key any --model qwen3-8b --dataset gsm8k --dataset_path datasets/gsm8k/test.jsonl
python -m experiments.geesc.launch.cot --base_url http://localhost:8001/v1 --api_key any --model qwen3-4b --dataset gsm8k --dataset_path datasets/gsm8k/test.jsonl
python -m experiments.geesc.launch.cot --base_url http://localhost:8002/v1 --api_key any --model qwen3-1.7b --dataset gsm8k --dataset_path datasets/gsm8k/test.jsonl
python -m experiments.geesc.launch.sc --base_url http://localhost:8000/v1 --api_key any --model qwen3-8b --dataset gsm8k --dataset_path datasets/gsm8k/test.jsonl --max_tokens 1024
python -m experiments.geesc.launch.sc --base_url http://localhost:8001/v1 --api_key any --model qwen3-4b --dataset gsm8k --dataset_path datasets/gsm8k/test.jsonl --max_tokens 1024
python -m experiments.geesc.launch.sc --base_url http://localhost:8002/v1 --api_key any --model qwen3-1.7b --dataset gsm8k --dataset_path datasets/gsm8k/test.jsonl --max_tokens 1024
```

获取性能：
```
python -m experiments.geesc.eval.eval --dataset gsm8k --dataset_path datasets/gsm8k/test.jsonl --results_prefix results/direct_qwen3-8b_gsm8k
python -m experiments.geesc.eval.eval --dataset gsm8k --dataset_path datasets/gsm8k/test.jsonl --results_prefix results/direct_qwen3-4b_gsm8k
python -m experiments.geesc.eval.eval --dataset gsm8k --dataset_path datasets/gsm8k/test.jsonl --results_prefix results/direct_qwen3-1.7b_gsm8k
python -m experiments.geesc.eval.eval --dataset gsm8k --dataset_path datasets/gsm8k/test.jsonl --results_prefix results/cot_qwen3-8b_gsm8k
python -m experiments.geesc.eval.eval --dataset gsm8k --dataset_path datasets/gsm8k/test.jsonl --results_prefix results/cot_qwen3-4b_gsm8k
python -m experiments.geesc.eval.eval --dataset gsm8k --dataset_path datasets/gsm8k/test.jsonl --results_prefix results/cot_qwen3-1.7b_gsm8k
python -m experiments.geesc.eval.eval_sc --dataset gsm8k --dataset_path datasets/gsm8k/test.jsonl --results_prefix results/sc_qwen3-8b_gsm8k
python -m experiments.geesc.eval.eval_sc --dataset gsm8k --dataset_path datasets/gsm8k/test.jsonl --results_prefix results/sc_qwen3-4b_gsm8k
python -m experiments.geesc.eval.eval_sc --dataset gsm8k --dataset_path datasets/gsm8k/test.jsonl --results_prefix results/sc_qwen3-1.7b_gsm8k


```

## 进行MATH500的基线实验
```
python -m experiments.geesc.launch.direct --base_url http://localhost:8000/v1 --api_key any --model qwen3-8b --dataset math --dataset_path datasets/math500/test.jsonl
python -m experiments.geesc.launch.direct --base_url http://localhost:8001/v1 --api_key any --model qwen3-4b --dataset math --dataset_path datasets/math500/test.jsonl
python -m experiments.geesc.launch.direct --base_url http://localhost:8002/v1 --api_key any --model qwen3-1.7b --dataset math --dataset_path datasets/math500/test.jsonl
python -m experiments.geesc.launch.cot --base_url http://localhost:8000/v1 --api_key any --model qwen3-8b --dataset math --dataset_path datasets/math500/test.jsonl --max_tokens 1024
python -m experiments.geesc.launch.cot --base_url http://localhost:8001/v1 --api_key any --model qwen3-4b --dataset math --dataset_path datasets/math500/test.jsonl --max_tokens 1024
python -m experiments.geesc.launch.cot --base_url http://localhost:8002/v1 --api_key any --model qwen3-1.7b --dataset math --dataset_path datasets/math500/test.jsonl --max_tokens 1024
python -m experiments.geesc.launch.sc --base_url http://localhost:8000/v1 --api_key any --model qwen3-8b --dataset math --dataset_path datasets/math500/test.jsonl --max_tokens 1024
python -m experiments.geesc.launch.sc --base_url http://localhost:8001/v1 --api_key any --model qwen3-4b --dataset math --dataset_path datasets/math500/test.jsonl --max_tokens 1024
python -m experiments.geesc.launch.sc --base_url http://localhost:8002/v1 --api_key any --model qwen3-1.7b --dataset math --dataset_path datasets/math500/test.jsonl --max_tokens 1024
```

获取性能：
```
python -m experiments.geesc.eval.eval --dataset math --dataset_path datasets/math500/test.jsonl --results_prefix results/direct_qwen3-8b_math
python -m experiments.geesc.eval.eval --dataset math --dataset_path datasets/math500/test.jsonl --results_prefix results/direct_qwen3-4b_math
python -m experiments.geesc.eval.eval --dataset math --dataset_path datasets/math500/test.jsonl --results_prefix results/direct_qwen3-1.7b_math
python -m experiments.geesc.eval.eval --dataset math --dataset_path datasets/math500/test.jsonl --results_prefix results/cot_qwen3-8b_math
python -m experiments.geesc.eval.eval --dataset math --dataset_path datasets/math500/test.jsonl --results_prefix results/cot_qwen3-4b_math
python -m experiments.geesc.eval.eval --dataset math --dataset_path datasets/math500/test.jsonl --results_prefix results/cot_qwen3-1.7b_math
python -m experiments.geesc.eval.eval_sc --dataset math --dataset_path datasets/math500/test.jsonl --results_prefix results/sc_qwen3-8b_math
python -m experiments.geesc.eval.eval_sc --dataset math --dataset_path datasets/math500/test.jsonl --results_prefix results/sc_qwen3-4b_math
python -m experiments.geesc.eval.eval_sc --dataset math --dataset_path datasets/math500/test.jsonl --results_prefix results/sc_qwen3-1.7b_math
```


# 性能记录

<table>
    <tr>
        <th>Model</th>
        <td colspan="2">Qwen3-8B</td>
        <td colspan="2">Qwen3-4B</td>
        <td colspan="2">Qwen3-1.7B</td>
    </tr>
    <tr>
        <th>Dataset</th>
        <td>GSM8K</td>
        <td>MATH</td>
        <td>GSM8K</td>
        <td>MATH</td>
        <td>GSM8K</td>
        <td>MATH</td>
    </tr>
    <tr>
        <th>Direct</th>
        <td>91.28 / 335.37</td>
        <td>64.60 / 1076.88</td>
        <td>89.23 / 326.82</td>
        <td>66.80 / 1117.81</td>
        <td>80.36 / 341.67</td>
        <td>60.20 / 896.57</td>
    </tr>
    <tr>
        <th>CoT</th>
        <td>90.83 / 756.89</td>
        <td>76.40 / 1523.05</td>
        <td>91.43 / 765.46</td>
        <td>74.00 / 1567.31</td>
        <td>63.35 / 783.34</td>
        <td>59.00 / 1833.61 </td>
    </tr>
    <tr>
        <th>Self-Consistency (5)</th>
        <td>93.10 / 3785.86</td>
        <td>80.20 / 7697.72</td>
        <td>92.34 / 3830.10</td>
        <td>78.40 / 7846.26</td>
        <td>84.00 / 3883.09</td>
        <td>62.80 / 9200.10</td>
    </tr>
</table>
