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