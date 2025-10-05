from utils.generator import Generator, GPT
import torch, os, json, tqdm, random, time, re
import numpy as np
from torch.utils.data import DataLoader
import numpy as np
import argparse
from collections import Counter
from evaluation import es_evaluate, es_falsify, es_cnt_evidence

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
    parser = argparse.ArgumentParser(description="ES-fine")
    parser.add_argument('--model_nickname', type=str, required=True, help='model nickname')
    parser.add_argument('--test_data_file', type=str, default="data/ReCo.test.json")
    parser.add_argument('--output_dir', type=str, default="output")

    parser.add_argument('--demo', type=str, choices=["zero", "few"], default="zero")
    parser.add_argument('--is_reasoning', action="store_true")
    parser.add_argument('--num_trial', type=int, default=1)
    parser.add_argument('--ablation_stage', type=str, required=True, choices=["falsify", "cnt_evidence", "none"])
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
    id2examples = dict()
    for data in json.load(open(data_file)):
        if args.ablation_stage == "falsify" and data["factuality"] == "True":
            continue
        if args.ablation_stage == "cnt_evidence" and data["reasoning_complexity"] == "NEI":
            continue
        
        passage = ""
        for idx, p in enumerate(data["passage"]):
            passage += f"({idx+1}) {p}\n"
        passage = passage.strip()
        statement = data["statement"]
        factuality = data["factuality"]
        
        if data["reasoning_complexity"] == "NEI":
            gold = "Insufficient"
        elif "single" in data["reasoning_complexity"]:
            gold = "Single"
        else:
            gold = "Inter"
        
        id2examples[data["id"]] = {"id": data["id"],
                                    "passage": passage,
                                    "statement": statement,
                                    "factuality": factuality,
                                    "responses_per_stage": {"falsify": None, "cnt_evidence": None},
                                    "pred": None,
                                    "pred_per_stage": {"falsify": None, "cnt_evidence": None},
                                    "preds_per_stage": {"falsify": [], "cnt_evidence": []},
                                    "gold": gold}
                        
    return id2examples
    
def generate(model, id2examples, do_greedy=False):
    stage2prompt = {"falsify": open(f"prompts/fine/falsify.txt").read(),
                    "cnt_evidence": open(f"prompts/fine/count_evidence.txt").read()}

    if args.demo == "few":
        stage2fewshot = {"falsify": open(f"prompts/fine/falsify_fewshot.txt").read(),
                         "cnt_evidence": open(f"prompts/fine/count_evidence_fewshot.txt").read()}
        for k in stage2prompt.keys():
            stage2prompt[k] = stage2prompt[k].replace("{ few_shot_examples }", stage2fewshot[k])

    else:
        for k in stage2prompt.keys():
            stage2prompt[k] = stage2prompt[k].replace("{ few_shot_examples }", "")      
    
    for stage in stage2prompt.keys():
        
        if args.ablation_stage != "none" and args.ablation_stage not in stage:
            continue      

        prompt_template = stage2prompt[stage]
        id_list, input_prompts = [], []
        id2pred = dict()
        for id, example in id2examples.items():
            if example["pred"] is not None:
                continue
                    
            id_list.append(id)
            input_prompt = prompt_template.replace("{ passage } ", example["passage"]).replace("{ statement }", example["statement"]).replace("{ factuality }", example["factuality"])
            input_prompts.append(input_prompt)
            
        fail = 0                    
        if "gpt" not in args.model_nickname:        
            responses = model.model_generate(input_prompts, do_greedy)
        else:
            responses = model.model_generate(id_list, input_prompts)
        for id, res in zip(id_list, responses):
            sampling_preds = []
            preds = []
            for r in res:
                pred = extract_output(stage, r)
                if pred is not None:
                    preds.append(pred)
                    sampling_preds.append(pred)
                else:
                    if stage == "falsify":
                        sampling_preds.append("Insufficient")
                    elif stage == "cnt_evidence":
                        sampling_preds.append("Inter")
                    
            if len(preds) == 0:
                fail += 1
                
            id2examples[id]["preds_per_stage"][stage] = sampling_preds
            id2examples[id]["responses_per_stage"][stage] = res
            if stage == "falsify":
                if len(preds) == 0:
                    major_pred = "Insufficient"
                else:
                    major_pred = Counter(preds).most_common(1)[0][0]
                
                id2examples[id]["pred_per_stage"][stage] = major_pred
                if major_pred == "Insufficient":
                    id2examples[id]["pred"] = "Insufficient"
                    
            if stage == "cnt_evidence":
                if len(preds) == 0:
                    major_pred = "Inter"
                else:
                    major_pred = Counter(preds).most_common(1)[0][0]                    
                
                id2examples[id]["pred_per_stage"][stage] = major_pred
                id2examples[id]["pred"] = major_pred
                
        print("#fail:", fail)
        
        if args.ablation_stage != "none":
            predictions, references = [], []
            for example in id2examples.values():
                predictions.append(example["pred_per_stage"][stage])
                references.append(example["gold"])
                
            if stage == "falsify":
                scores = es_falsify(predictions, references)
            elif stage == "cnt_evidence":
                scores = es_cnt_evidence(predictions, references)
                
    if args.ablation_stage == "none":
        predictions, references = [], []
        for example in id2examples.values():
            predictions.append(example["pred"])
            references.append(example["gold"])
        
        scores = es_evaluate(predictions, references)             

    return id2examples, scores
    
def extract_output(stage, response):
    response = response.lower().replace("**", "").replace("```", "")
    
    if stage == "falsify":
        labels = ["Insufficient", "Contradiction"]
        for label in labels:
            if response.strip().split("\n")[0] == label.lower():
                return label
            
        pattern = r'answer:\s+(contradiction|insufficient)'
        match = re.search(pattern, response)

        if match:
            for label in labels:
                if label.lower() in match.group(0):
                    return label
        else:
            return None

    if stage == "cnt_evidence":
        pattern = r'answer:\s+(\d+(?:\.\d+)?)'
        match = re.search(pattern, response)

        if match:
            num = match.group(1)
            if num == "0":
                return "Insufficient"
            elif num == "1":
                return "Single"
            else:
                return "Inter"
        else:
            return None
        
def main():
    global args
    args = parse_args()
    print(args)
    set_random_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    exp_dir = f"{args.output_dir}/es_output/{args.model_nickname}-fine-{args.demo}shot"
    if args.num_trial == 1:
        exp_dir += "-greedy"
    else:
        exp_dir += f"-{args.num_trial}_trial"
        
    if args.is_reasoning:
        exp_dir += f"-reasoning"
        
    if args.ablation_stage != "none":
        exp_dir += f"-{args.ablation_stage}"
    
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
        
        model = GPT(args.model_nickname, "fine", args)   
            
    now = time.localtime()
    print("current time:", time.strftime("%Y-%m-%d %H:%M:%S", now))
    
    id2examples, scores = generate(model, id2examples, do_greedy=args.num_trial == 1)
    
    os.makedirs(exp_dir, exist_ok=True)
    with open(f"{exp_dir}/result.json", "w") as fout:
        json.dump({"score": scores, "config": vars(args), "result": id2examples}, fout, indent=3)
        
if __name__ == '__main__':
    main()