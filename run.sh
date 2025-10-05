MODEL_NAME=$1   ## [gemma-s|gemma-m|mistral-s|mistral-m|qwen-s|qwen-m|qwen3-s|qwen3-m]
NUM_TRIAL=$2    ## 1 is greedy decoding
STRATEGY=$3     ## [es.sp|es.cot|tl.sp|tl.cot]
DEMO=$4         ## [zero|one|few]

echo $MODEL_NAME
echo $NUM_TRIAL
echo $STRATEGY

if [ "$STRATEGY" = "es.sp" ] || [ "$STRATEGY" = "es.cot" ]; then
    python es_inference.py --num_trial "$NUM_TRIAL" --model_nickname "$MODEL_NAME" --strategy "$STRATEGY" --demo "$DEMO"
elif [ "$STRATEGY" = "tl.sp" ] || [ "$STRATEGY" = "tl.cot" ]; then
    python tl_inference.py --num_trial "$NUM_TRIAL" --model_nickname "$MODEL_NAME" --strategy "$STRATEGY" --demo "$DEMO"
else
    echo "Invalid strategy: $STRATEGY"
    exit 1
fi