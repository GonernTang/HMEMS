
#!/bin/bash

if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
    export PYTHONPATH="$PYTHONPATH:$PWD"
else
    echo ".env not found"
    echo "Please create a .env file with the following variables:"
    echo "LOCOMO_PATH=/path/to/your/dataset"
    echo "LOCOMO_RES_PATH=/path/to/your/results"
    exit 1
fi


declare -A CONFIGS

# Config 1: 
CONFIGS["config1"]="
    --min_aug_count 4 \
    --min_relevant_score 0.7 \
    --retrieve_raw_topk 5 \
    --retrieve_aug_topk 5 \
    --output_file \${LOCOMO_RES_PATH}/vecmem_c4v5a5t7_partial.json"

# Config 2:
CONFIGS["config2"]="
    --min_aug_count 3 \
    --min_relevant_score 0.7 \
    --retrieve_raw_topk 5 \
    --retrieve_aug_topk 5 \
    --output_file \${LOCOMO_RES_PATH}/vecmem_c3v5a5t7_partial.json"

# Config 3: 
CONFIGS["config3"]="
    --min_aug_count 3 \
    --min_relevant_score 0.75 \
    --retrieve_raw_topk 5 \
    --retrieve_aug_topk 5 \
    --output_file \${LOCOMO_RES_PATH}/vecmem_c3v5a5t75_partial.json"

# Config 4:
CONFIGS["config4"]="
    --min_aug_count 4 \
    --min_relevant_score 0.7 \
    --retrieve_raw_topk 3 \
    --retrieve_aug_topk 5 \
    --enable_iter_anwser \
    --output_file \${LOCOMO_RES_PATH}/vecmem_c4v3a5t7_iter_partial.json"

# Config 5 (Pure Rag):
CONFIGS["config_rag"]="
    --min_aug_count 4 \
    --min_relevant_score 0.99 \
    --retrieve_raw_topk 10 \
    --retrieve_aug_topk 0 \
    --itr_raw_topk 5 \
    --itr_aug_topk 0 \
    --enable_iter_anwser \
    --output_file \${LOCOMO_RES_PATH}/vecmem_rag_iter_partial.json"

# Config 6 (Iterative Config 3)
CONFIGS["config6"]="
    --min_aug_count 3 \
    --min_relevant_score 0.75 \
    --retrieve_raw_topk 5 \
    --retrieve_aug_topk 5 \
    --enable_iter_anwser \
    --output_file \${LOCOMO_RES_PATH}/vecmem_c3v5a5t75_iter_partial.json"

# Config 7 (Iterative Config 2):
CONFIGS["config7"]="
    --min_aug_count 3 \
    --min_relevant_score 0.7 \
    --retrieve_raw_topk 5 \
    --retrieve_aug_topk 5 \
    --enable_iter_anwser \
    --output_file \${LOCOMO_RES_PATH}/vecmem_c3v5a5t7_iter_partial.json"


### 4.1 mini versions
CONFIGS["config1_4.1_mini"]="
    --min_aug_count 4 \
    --min_relevant_score 0.7 \
    --retrieve_raw_topk 5 \
    --retrieve_aug_topk 5 \
    --model gpt-4.1-mini \
    --output_file \${LOCOMO_RES_PATH}/vecmem_c4v5a5t7_partial_4.1_mini.json"

# Config 2:
CONFIGS["config2_4.1_mini"]="
    --min_aug_count 3 \
    --min_relevant_score 0.7 \
    --retrieve_raw_topk 5 \
    --retrieve_aug_topk 5 \ 
    --model gpt-4.1-mini \
    --output_file \${LOCOMO_RES_PATH}/vecmem_c3v5a5t7_partial_4.1_mini.json"

# Config 3: 
CONFIGS["config3_4.1_mini"]="
    --min_aug_count 3 \
    --min_relevant_score 0.75 \
    --retrieve_raw_topk 5 \
    --retrieve_aug_topk 5 \
    --model gpt-4.1-mini \
    --enable_stat \
    --output_file \${LOCOMO_RES_PATH}/vecmem_c3v5a5t75_partial_4.1_mini.json"

# Config 4:
CONFIGS["config4_4.1_mini"]="
    --min_aug_count 4 \
    --min_relevant_score 0.7 \
    --retrieve_raw_topk 3 \
    --retrieve_aug_topk 5 \ 
    --enable_iter_anwser \
    --model gpt-4.1-mini \
    --output_file \${LOCOMO_RES_PATH}/vecmem_c4v3a5t7_iter_partial_4.1_mini.json"

# Config 5 (Pure Rag):
CONFIGS["config_rag_4.1_mini"]="
    --min_aug_count 4 \
    --min_relevant_score 0.99 \
    --retrieve_raw_topk 10 \
    --retrieve_aug_topk 0 \
    --itr_raw_topk 5 \
    --itr_aug_topk 0 \
    --enable_iter_anwser \
    --model gpt-4.1-mini \
    --output_file \${LOCOMO_RES_PATH}/vecmem_rag_iter_partial_4.1_mini.json"

# Config 6 (Iterative Config 3)
CONFIGS["config6_4.1_mini"]="
    --min_aug_count 3 \
    --min_relevant_score 0.75 \
    --retrieve_raw_topk 5 \
    --retrieve_aug_topk 5 \
    --enable_iter_anwser \
    --model gpt-4.1-mini \
    --output_file \${LOCOMO_RES_PATH}/vecmem_c3v5a5t75_iter_partial_4.1_mini.json"

# Config 7 (Iterative Config 2):
CONFIGS["config7_4.1_mini"]="
    --min_aug_count 3 \
    --min_relevant_score 0.7 \
    --retrieve_raw_topk 5 \
    --retrieve_aug_topk 5 \
    --enable_iter_anwser \
    --model gpt-4.1-mini \
    --output_file \${LOCOMO_RES_PATH}/vecmem_c3v5a5t7_iter_partial_4.1_mini.json"


CONFIGS["config_c3v5a5t75_iter_partial_4.1_mini"]="
    --min_aug_count 3 \
    --min_relevant_score 0.75 \
    --retrieve_raw_topk 5 \
    --retrieve_aug_topk 5 \
    --enable_iter_anwser \
    --merge_with_aug_thresh 0.85 \
    --model gpt-4.1-mini \
    --output_file \${LOCOMO_RES_PATH}/vecmem_c3v5a5t75_iter_partial_4.1_mini_improved_prompt.json"


CONFIGS["config_c3v5a5t8_iter_4.1_mini"]="
    --min_aug_count 3 \
    --min_relevant_score 0.8 \
    --retrieve_raw_topk 5 \
    --retrieve_aug_topk 5 \
    --enable_iter_anwser \
    --merge_with_aug_thresh 0.8 \
    --model gpt-4.1-mini \
    --enable_stat \
    --output_file \${LOCOMO_RES_PATH}/vecmem_c3v5a5t8_iter_4.1_mini_improved_prompt.json"

CONFIGS["config_c3v5a5t8m1_iter_partial_4.1_mini"]="
    --min_aug_count 3 \
    --min_relevant_score 0.8 \
    --retrieve_raw_topk 5 \
    --retrieve_aug_topk 5 \
    --enable_iter_anwser \
    --merge_with_aug_thresh 1.0 \
    --model gpt-4.1-mini \
    --enable_stat \
    --output_file \${LOCOMO_RES_PATH}/vecmem_c3v5a5t8m1_iter_partial_4.1_mini_improved_prompt.json"


CONFIGS["config_c3v5a5t8m75_iter_partial_4.1_mini"]="
    --min_aug_count 3 \
    --min_relevant_score 0.8 \
    --retrieve_raw_topk 5 \
    --retrieve_aug_topk 5 \
    --enable_iter_anwser \
    --merge_with_aug_thresh 0.75 \
    --model gpt-4.1-mini \
    --enable_stat \
    --output_file \${LOCOMO_RES_PATH}/vecmem_c3v5a5t8m75_iter_partial_4.1_mini_improved_prompt.json"

CONFIGS["config_c3v6a4t8m8_iter_partial_4.1_mini"]="
    --min_aug_count 3 \
    --min_relevant_score 0.8 \
    --retrieve_raw_topk 6 \
    --retrieve_aug_topk 4 \
    --enable_iter_anwser \
    --merge_with_aug_thresh 0.8 \
    --model gpt-4.1-mini \
    --enable_stat \
    --output_file \${LOCOMO_RES_PATH}/vecmem_c3v6a4t8m8_iter_partial_4.1_mini_improved_prompt.json"

CONFIGS["config_c3v5a5t7m75_iter_partial_4.1_mini"]="
    --min_aug_count 3 \
    --min_relevant_score 0.7 \
    --retrieve_raw_topk 5 \
    --retrieve_aug_topk 5 \
    --enable_iter_anwser \
    --merge_with_aug_thresh 0.75 \
    --model gpt-4.1-mini-ca \
    --enable_stat \
    --conv_limit 4 \
    --output_file \${LOCOMO_RES_PATH}/vecmem_c3v5a5t7m75_iter_partial_4.1_mini_improved_prompt.json"


CONFIGS["config_c3v5a5t8m75_iter_4.1_mini"]="
    --min_aug_count 3 \
    --min_relevant_score 0.7 \
    --retrieve_raw_topk 5 \
    --retrieve_aug_topk 5 \
    --enable_iter_anwser \
    --merge_with_aug_thresh 0.75 \
    --model gpt-4.1-mini-ca \
    --enable_stat \
    --output_file \${LOCOMO_RES_PATH}/vecmem_c3v5a5t8m75_iter_4.1_mini_improved_prompt.json"


CONFIGS["config_c4v5a5t65m75_partial_iter_4.1_mini"]="
    --min_aug_count 4 \
    --min_relevant_score 0.66 \
    --retrieve_raw_topk 5 \
    --retrieve_aug_topk 5 \
    --enable_iter_anwser\
    --merge_with_aug_thresh 0.75 \
    --model gpt-4.1-mini-ca \
    --enable_stat\
    --conv_limit 4 \
    --output_file \${LOCOMO_RES_PATH}/vecmem_c4v5a5t65m75_partial_iter_4.1_mini_improved_prompt.json"

# Semantic Memory Testing

# t75m75 no iter s10 t05
CONFIGS["config_c3v5a5t75m75_partial_semantic_s10t05_4.1_mini"]="
    --min_aug_count 3 \
    --min_relevant_score 0.75 \
    --retrieve_raw_topk 5 \
    --retrieve_aug_topk 5 \
    --merge_with_aug_thresh 0.75 \
    --model gpt-4.1-mini-ca \
    --enable_stat\
    --enable_semantic_memory\
    --semantic_memory_topk 10 \
    --semantic_memory_threshold 0.5 \
    --conv_limit 4 \
    --output_file \${LOCOMO_RES_PATH}/vecmem_c3v5a5t75m75_partial_semantic_s10t05_4.1_mini.json"

# t75m75 no iter s10 t05 Full
CONFIGS["config_c3v5a5t75m75_semantic_s10t05_4.1_mini"]="
    --min_aug_count 3 \
    --min_relevant_score 0.75 \
    --retrieve_raw_topk 5 \
    --retrieve_aug_topk 5 \
    --merge_with_aug_thresh 0.75 \
    --model gpt-4.1-mini-ca \
    --enable_stat\
    --enable_semantic_memory\
    --semantic_memory_topk 10 \
    --semantic_memory_threshold 0.5 \
    --output_file \${LOCOMO_RES_PATH}/vecmem_c3v5a5t75m75_semantic_s10t05_4.1_mini.json"

# t75m75 no iter s10 t0 Full
CONFIGS["config_c3v5a5t75m75_semantic_s10t0_4.1_mini"]="
    --min_aug_count 3 \
    --min_relevant_score 0.75 \
    --retrieve_raw_topk 5 \
    --retrieve_aug_topk 5 \
    --merge_with_aug_thresh 0.75 \
    --model gpt-4.1-mini-ca \
    --enable_stat\
    --enable_semantic_memory\
    --semantic_memory_topk 10 \
    --semantic_memory_threshold 0.0 \
    --output_file \${LOCOMO_RES_PATH}/vecmem_c3v5a5t75m75_semantic_s10t0_4.1_mini.json"
    
# t75m75 no iter s10 t0
CONFIGS["config_c3v5a5t75m75_partial_semantic_s10t0_4.1_mini"]="
    --min_aug_count 3 \
    --min_relevant_score 0.75 \
    --retrieve_raw_topk 5 \
    --retrieve_aug_topk 5 \
    --merge_with_aug_thresh 0.75 \
    --model gpt-4.1-mini-ca \
    --enable_stat\
    --enable_semantic_memory\
    --semantic_memory_topk 10 \
    --semantic_memory_threshold 0.0 \
    --conv_limit 4 \
    --output_file \${LOCOMO_RES_PATH}/vecmem_c3v5a5t75m75_partial_semantic_s10t0_4.1_mini.json"

# t75m75 iter s10 t05
CONFIGS["config_c3v5a5t75m75_partial_iter_semantic_s10t05_4.1_mini"]="
    --min_aug_count 3 \
    --min_relevant_score 0.75 \
    --retrieve_raw_topk 5 \
    --retrieve_aug_topk 5 \
    --merge_with_aug_thresh 0.75 \
    --model gpt-4.1-mini-ca \
    --enable_stat\
    --enable_semantic_memory\
    --enable_iter_anwser \
    --semantic_memory_topk 10 \
    --semantic_memory_threshold 0.5 \
    --conv_limit 4 \
    --output_file \${LOCOMO_RES_PATH}/vecmem_c3v5a5t75m75_partial_iter_semantic_s10t05_4.1_mini.json"

# t8m75 no iter s10 t05
CONFIGS["config_c3v5a5t8m75_partial_semantic_s10t05_4.1_mini"]="
    --min_aug_count 3 \
    --min_relevant_score 0.8 \
    --retrieve_raw_topk 5 \
    --retrieve_aug_topk 5 \
    --merge_with_aug_thresh 0.75 \
    --model gpt-4.1-mini-ca \
    --enable_stat\
    --enable_semantic_memory\
    --semantic_memory_topk 10 \
    --semantic_memory_threshold 0.5 \
    --conv_limit 4 \
    --output_file \${LOCOMO_RES_PATH}/vecmem_c3v5a5t8m75_partial_semantic_s10t05_4.1_mini.json"


CONFIGS["config_c3v5a5t8m8_partial_semantic_s10t0_4.1_mini_no_filter"]="
    --min_aug_count 3 \
    --min_relevant_score 0.8 \
    --retrieve_raw_topk 5 \
    --retrieve_aug_topk 5 \
    --merge_with_aug_thresh 0.8 \
    --model gpt-4.1-mini-ca \
    --enable_stat\
    --enable_semantic_memory\
    --semantic_memory_topk 10 \
    --semantic_memory_threshold 0.0 \
    --output_file \${LOCOMO_RES_PATH}/vecmem_c3v5a5t8m8_partial_semantic_s10t0_4.1_mini.json"

# t8m75 iter s10 t05
CONFIGS["config_c3v5a5t8m75_partial_iter_semantic_s10t05_4.1_mini"]="
    --min_aug_count 3 \
    --min_relevant_score 0.8 \
    --retrieve_raw_topk 5 \
    --retrieve_aug_topk 5 \
    --merge_with_aug_thresh 0.75 \
    --model gpt-4.1-mini-ca \
    --enable_stat\
    --enable_semantic_memory\
    --enable_iter_anwser \
    --semantic_memory_topk 10 \
    --semantic_memory_threshold 0.5 \
    --conv_limit 4 \
    --output_file \${LOCOMO_RES_PATH}/vecmem_c3v5a5t8m75_partial_iter_semantic_s10t05_4.1_mini.json"

# t78m78 no iter s10 t0
CONFIGS["config_c3v5a5t78m78_semantic_s10t0_4.1_mini"]="
    --min_aug_count 3 \
    --min_relevant_score 0.78 \
    --retrieve_raw_topk 5 \
    --retrieve_aug_topk 5 \
    --merge_with_aug_thresh 0.78 \
    --model gpt-4.1-mini-ca \
    --enable_stat\
    --enable_semantic_memory\
    --semantic_memory_topk 10 \
    --semantic_memory_threshold 0.0 \
    --output_file \${LOCOMO_RES_PATH}/vecmem_c3v5a5t78m78_semantic_s10t0_4.1_mini.json"
    
# t78m78 no iter s10 t05
CONFIGS["config_c3v5a5t78m78_semantic_s10t05_4.1_mini"]="
    --min_aug_count 3 \
    --min_relevant_score 0.78 \
    --retrieve_raw_topk 5 \
    --retrieve_aug_topk 5 \
    --merge_with_aug_thresh 0.78 \
    --model gpt-4.1-mini-ca \
    --enable_stat\
    --enable_semantic_memory\
    --semantic_memory_topk 10 \
    --semantic_memory_threshold 0.5 \
    --output_file \${LOCOMO_RES_PATH}/vecmem_c3v5a5t78m78_semantic_s10t0_4.1_mini.json"

run_config() {
    local config_name=$1
    local config_args=${CONFIGS[$config_name]}
    
    if [[ -z "$config_args" ]]; then
        echo "Config '$config_name' not found"
        return 1
    fi
    
    # Expand environment variables
    config_args=$(eval echo "$config_args")
    
    echo "Starting $config_name..."
    nohup python src/run_experiments.py $config_args > logs/${config_name}.log 2>&1 &
    echo "$config_name 已启动 (PID: $!)"
}

# Create logs directory
mkdir -p logs

# If no parameters, run all configs
if [[ $# -eq 0 ]]; then
    for config in "${!CONFIGS[@]}"; do
        run_config "$config"
        sleep 1
    done
else
    # Run specified configs
    for config in "$@"; do
        run_config "$config"
        sleep 1
    done
fi

echo "All tasks started, logs in logs/ directory"