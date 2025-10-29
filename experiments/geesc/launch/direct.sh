echo "Running qwen3-8b on gsm8k test set"
python -m experiments.geesc.launch.direct --base_url http://localhost:8000/v1 --api_key any --model qwen3-8b --dataset gsm8k --dataset_path datasets/gsm8k/test.jsonl
echo "Running qwen3-8b on math500 test set"
python -m experiments.geesc.launch.direct --base_url http://localhost:8000/v1 --api_key any --model qwen3-8b --dataset math --dataset_path datasets/math500/test.jsonl
echo "Running qwen3-4b on gsm8k test set"
python -m experiments.geesc.launch.direct --base_url http://localhost:8001/v1 --api_key any --model qwen3-4b --dataset gsm8k --dataset_path datasets/gsm8k/test.jsonl
echo "Running qwen3-4b on math500 test set"
python -m experiments.geesc.launch.direct --base_url http://localhost:8001/v1 --api_key any --model qwen3-4b --dataset math --dataset_path datasets/math500/test.jsonl
echo "Running qwen3-1.7b on gsm8k test set"
python -m experiments.geesc.launch.direct --base_url http://localhost:8002/v1 --api_key any --model qwen3-1.7b --dataset gsm8k --dataset_path datasets/gsm8k/test.jsonl
echo "Running qwen3-1.7b on math500 test set"
python -m experiments.geesc.launch.direct --base_url http://localhost:8002/v1 --api_key any --model qwen3-1.7b --dataset math --dataset_path datasets/math500/test.jsonl