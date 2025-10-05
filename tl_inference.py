from utils.generator import Generator, GPT
import torch, os, json, tqdm, random, time, re
import numpy as np
import argparse
from collections import Counter
from evaluation import tl_evaluate

#import torch._dynamo 
#torch._dynamo.config.cache_size_limit = 64

args = None
NICKNAME2NAME = {"gemma-s": "google/gemma-2-9b-it",
                 "gemma-m": "google/gemma-2-27b-it",
                 "mistral-s": "mistralai/Mistral-7B-Instruct-v0.3",
                 "mistral-m": "mistralai/Mistral-Small-24B-Instruct-2501",
                 "qwen-s": "Qwen/Qwen2.5-7B-Instruct",
                 "qwen-m": "Qwen/Qwen2.5-32B-Instruct",
                 "qwen3-s": "Qwen/Qwen3-8B",
                 "qwen3-m": "Qwen/Qwen3-32B"}

def parse_args():
    parser = argparse.ArgumentParser(description="TL")
    parser.add_argument('--model_nickname', type=str, required=True, help='model nickname')
    parser.add_argument('--test_data_file', type=str, default="data/ReCo.test.json")
    parser.add_argument('--output_dir', type=str, default="output")

    parser.add_argument('--es_type', type=str, choices=["single", "multi", "all"], default="all", help='evidence scope')
    parser.add_argument('--strategy', type=str, help="tl.cot")
    parser.add_argument('--demo', type=str, choices=["zero", "one", "few"], default="zero")
    parser.add_argument('--is_reasoning', action="store_true")
    parser.add_argument('--num_trial', type=int, default=1)
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--batch_size', type=int, default=8)
    args = parser.parse_args()
    return args

def set_random_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_examples(data_file):
    global prompt_template
    
    prompt_template = open(f"prompts/{args.strategy}.txt").read()
    if args.demo == "one":
        oneshot = open(f"prompts/{args.strategy}_oneshot.txt").read()
        prompt_template = prompt_template.replace("{ few_shot_examples }", oneshot)
    elif args.demo == "few":
        fewshot = open(f"prompts/{args.strategy}_fewshot.txt").read()
        prompt_template = prompt_template.replace("{ few_shot_examples }", fewshot)
    else:
        prompt_template = prompt_template.replace("{ few_shot_examples }", "")

    id2examples = dict()
    for data in json.load(open(data_file)):
        if data["reasoning_complexity"] == "NEI":
            continue
        if args.es_type != "all" and args.es_type not in data["reasoning_complexity"]:
            continue
        
        passage = ""
        for idx, p in enumerate(data["passage"]):
            passage += f"({idx+1}) {p}\n"
        passage = passage.strip()
        
        if data["factuality"] == "True":
            statement = data["statement"]
        else:
            statement = data["true_version"]           
    
        input_prompt = prompt_template.replace("{ passage }", passage).replace("{ statement }", statement)   
        id2examples[data["id"]] = {"id": data["id"],
                                    "es_type": data["reasoning_complexity"].split("_")[0],
                                    "passage": passage,
                                    "statement": statement,
                                    "input_prompt": input_prompt,
                                    "responses": [],
                                    "preds": None,
                                    "final_pred": None,
                                    "gold": data["reasoning_complexity"].split("_")[-1]}
                        
    return id2examples
    
def generate(model, id2examples, do_greedy=False):
    map = {"Word Matching": 0,
           "Paraphrasing": 1,
           "Inference": 2,
           "Transformed Word Matching": 3,
           "Transformed Paraphrasing": 4}
    
    fail = 0
    id_list, input_prompts = [], []
    id2pred = dict()
    for id, example in id2examples.items():
        id_list.append(id)
        input_prompts.append(example["input_prompt"])
        
    if "gpt" not in args.model_nickname:        
        responses = model.model_generate(input_prompts, do_greedy)
    else:
        responses = model.model_generate(id_list, input_prompts)
        
    for id, res in zip(id_list, responses):
        preds = []
        for r in res:
            pred = extract_output(r)
            if pred is None:
                fail += 1
            else:
                preds.append(pred)
                
        if len(preds) == 0:
            major_pred = "Word Matching"
        else:
            major_pred = Counter(preds).most_common(1)[0][0]
        id2pred[id] = major_pred
        
        id2examples[id]["responses"] = res
        id2examples[id]["preds"] = preds
        id2examples[id]["final_pred"] = major_pred
        
    es_types, predictions, references = [], [], []
    for example in id2examples.values():
        es_types.append(example["es_type"])
        predictions.append(example["final_pred"])
        references.append(example["gold"])
    scores = tl_evaluate(es_types, predictions, references)

    return id2examples, scores
    
def extract_output(response):
    response = response.lower().replace("**", "").replace("```", "")
    
    labels = ["Transformed Word Matching", "Transformed Paraphrasing", "Word Matching", "Paraphrasing", "Inference"]
    for label in labels:
        if response.strip().split("\n")[0] == label.lower():
            return label
        
    pattern = r'answer:\s+(transformed word matching|transformed paraphrasing|word matching|paraphrasing|inference)'
    match = re.search(pattern, response)

    if match:
        for label in labels:
            if label.lower() in match.group(0):
                return label
    else:
        return None
        
def main():
    global args
    args = parse_args()        
    print(args)
    set_random_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    exp_dir = f"{args.output_dir}/tl_output/{args.model_nickname}-{args.strategy}-{args.demo}shot"

    if args.num_trial == 1:
        exp_dir += "-greedy"
    else:
        exp_dir += f"-{args.num_trial}_trial"
    
    if args.is_reasoning:
        exp_dir += f"-reasoning"
    
    print("EXP Dir:", exp_dir)
    if os.path.isdir(exp_dir) and os.listdir(exp_dir):
        print("Already Run.")
        return

    id2examples = get_examples(args.test_data_file)
    
    if "gpt" not in args.model_nickname:
        model_name = NICKNAME2NAME[args.model_nickname]
        print(f'{model_name} is used')
            
        model = Generator(model_name, device, args)
    else:
        print(f'{args.model_nickname} is used')
        
        model = GPT(args.model_nickname, args.strategy, args)

    now = time.localtime()
    print("current time:", time.strftime("%Y-%m-%d %H:%M:%S", now))
    
    id2examples, scores = generate(model, id2examples, do_greedy=args.num_trial == 1)
    
    os.makedirs(exp_dir, exist_ok=True)
    with open(f"{exp_dir}/result.json", "w") as fout:
        json.dump({"score": scores, "config": vars(args), "result": id2examples}, fout, indent=3)
        
if __name__ == '__main__':
    #main()
    main()