NUM_DEV=${ND:-"-1"}

ARG_N_EVAL=${N_EVAL:-"-1"}

EVAL=${1:-"test"}
NUM_EVAL_SAM=5
NUM_SAM=5
TEMPR=0.05
main_exp()
{
    OPENAI_API_KEY=${KEY} python3 D:/LLM-Logic-Reasoner/SAT-LM/run_manual.py --num_eval_samples ${NUM_EVAL_SAM} --num_samples ${NUM_SAM} --task proofd5 --num_dev ${NUM_DEV} --run_pred --manual_prompt_id cot --temperature ${TEMPR}  --eval_split ${EVAL} --batch_size 2 --style_template cot ${FLAG}
    OPENAI_API_KEY=${KEY} python3 D:/LLM-Logic-Reasoner/SAT-LM/run_manual.py --num_eval_samples ${NUM_EVAL_SAM} --num_samples ${NUM_SAM} --task proofd5 --num_dev ${NUM_DEV} --run_pred --manual_prompt_id proglm --temperature ${TEMPR} --eval_split ${EVAL} --batch_size 2 --style_template proglm ${FLAG}
    OPENAI_API_KEY=${KEY} python3 D:/LLM-Logic-Reasoner/SAT-LM/run_manual.py --num_eval_samples ${NUM_EVAL_SAM} --num_samples ${NUM_SAM} --task proofd5 --num_dev ${NUM_DEV} --run_pred --manual_prompt_id satlm --temperature ${TEMPR} --eval_split ${EVAL} --batch_size 2 --style_template satlm ${FLAG}

python3 SAT-LM/run_manual.py --num_eval_samples 5 --num_samples 5 --task proofd5 --num_dev -1 --run_pred --manual_prompt_id satlm --temperature 0.05 --eval_split test --batch_size 2 --style_template satlm ${FLAG}
}

main_exp