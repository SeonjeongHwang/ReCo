MODEL_NAME=$1   ## [gemma-s|gemma-m|mistral-s|mistral-m|qwen-s|qwen-m|qwen3-s|qwen3-m]
NUM_TRIAL=$2    ## 1 is greedy decoding
SUBTASK=$3      ## [falsify|cnt_evidence|inference|paraphrasing|transformation]
DEMO=$4         ## [zero|few]

echo $MODEL_NAME
echo $NUM_TRIAL
echo $SUBTASK
echo $DEMO

if [[ "$SUBTASK" == "falsify" || "$SUBTASK" == "cnt_evidence" ]]; then
    python es_fine_inference.py \
        --num_trial "$NUM_TRIAL" \
        --model_nickname "$MODEL_NAME" \
        --ablation_stage "$SUBTASK" \
        --demo "$DEMO"
elif [[ "$SUBTASK" == "inference" || "$SUBTASK" == "paraphrasing" || "$SUBTASK" == "transformation" ]]; then
    python tl_fine_inference.py \
        --num_trial "$NUM_TRIAL" \
        --model_nickname "$MODEL_NAME" \
        --ablation_stage "$SUBTASK" \
        --demo "$DEMO"
else
    echo "Invalid subtask name: $SUBTASK"
    exit 1
fi