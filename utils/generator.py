import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch, tqdm, json, os, time, jsonlines
from torch.utils.data import DataLoader
import numpy as np

from openai import OpenAI
client = OpenAI(api_key="######YOUR OPEN AI KEY######")

class GPT:
    def __init__(self, model_nickname, strategy, args):
        self.model_nickname = model_nickname
        
        self.args = args
        self.seed = args.seed
        self.top_k = 20
        self.top_p = 0.8
        self.temperature = 0.7
        self.is_greedy = args.num_trial == 1
        
        self.batch_name = f"{model_nickname}_{strategy}"
        if args.is_oneshot:
            self.batch_name += "-oneshot"
        elif args.is_fewshot:
            self.batch_name += "-fewshot"
        else:
            self.batch_name += "-zeroshot"
            
    def model_generate(self, id_list, input_prompts):
        # Run
        os.makedirs(f"{self.args.output_dir}/gpt_batches/", exist_ok=True)
        id2outputNames = dict()
        for t in range(self.args.num_trial):
            input_batch_name = f"{self.args.output_dir}/gpt_batches/{self.batch_name}-trial{t+1}-gptBatchInput.jsonl"
            output_batch_name = f"{self.args.output_dir}/gpt_batches/{self.batch_name}-trial{t+1}-gptBatchOutput.jsonl"
            
            if not os.path.isfile(output_batch_name):
                batch_inputs = self.get_batch_inputs(id_list, input_prompts)
                self.save_input_batch_file(input_batch_name, batch_inputs)
                
                batch_id = self.run_batch(input_batch_name)
                id2outputNames[batch_id] = output_batch_name
            else:
                print(f"{output_batch_name} already exists.")
           
        # Save     
        for batch_id, output_batch_name in id2outputNames.items():
            print("Waiting...", output_batch_name)
            _ = self.save_result(batch_id, output_batch_name)
                
        cost = 0
        total_responses = []    
        for t in range(self.args.num_trial):
            output_batch_name = f"{self.args.output_dir}/gpt_batches/{self.batch_name}-trial{t+1}-gptBatchOutput.jsonl"
            responses = []
            with jsonlines.open(output_batch_name) as f:
                for line in f.iter():
                    id = line["custom_id"]
                    r = line["response"]["body"]["choices"][0]["message"]["content"]
                    cost += self.calc_price(line)
                        
                    responses.append(r)
            total_responses.append(responses)
            
        responses_per_example = []
        for idx in range(len(total_responses[0])):
            res = []
            for i in range(self.args.num_trial):
                res.append(total_responses[i][idx])
            responses_per_example.append(res)
        responses = responses_per_example
            
        print(cost)
        return responses      
            
    def get_batch_inputs(self, id_list, input_prompts):
        batch_inputs = []
        for id, input_prompt in zip(id_list, input_prompts):
            if self.is_greedy:
                line = {"custom_id": id, 
                        "method": "POST", 
                        "url": "/v1/chat/completions", 
                        "body": {"model": self.model_nickname, 
                                "messages": [{"role": "user", "content": input_prompt}],
                                "temperature": 0,
                                "top_p": 1}
                        }
            else:
                line = {"custom_id": id, 
                        "method": "POST", 
                        "url": "/v1/chat/completions", 
                        "body": {"model": self.model_nickname, 
                                "messages": [{"role": "user", "content": input_prompt}],
                                "temperature": self.temperature,
                                "top_p": self.top_p}
                        }                
            batch_inputs.append(line)
            
        print("#examples:", len(batch_inputs))
        return batch_inputs  
    
    def save_input_batch_file(self, input_batch_name, batch_inputs):
        with open(input_batch_name, "w", encoding="UTF-8") as fout:
            for line in batch_inputs:
                json.dump(line, fout, ensure_ascii=False)
                fout.write("\n")
                
    def run_batch(self, input_batch_name):
        print("### Run Batch")
        batch_input_file = client.files.create(
            file=open(input_batch_name, "rb"),
            purpose="batch"
        )

        batch_input_file_id = batch_input_file.id
        batch = client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
                "description": input_batch_name
            }
        )
        return batch.id
    
    def save_result(self, batch_id, output_batch_name):
        while True:
            batch_current = client.batches.retrieve(batch_id)
            print(batch_current.status)
            if batch_current.status == "completed":
                output_file_id = batch_current.output_file_id
                break
            elif batch_current.status == "failed":
                assert False, "FAIL"
            time.sleep(30)
    
        print("### Save Result")
        file_response = client.files.content(output_file_id)
        with open(output_batch_name, "w") as fout:
            for line in file_response.text.split("\n"):
                if len(line) == 0:
                    continue
                fout.write(line+"\n")
        return file_response
    
    def calc_price(self, line):
        model_name = line["response"]["body"]["model"]
        input_length = line["response"]["body"]["usage"]["prompt_tokens"]
        output_length = line["response"]["body"]["usage"]["completion_tokens"]
        
        if model_name == "gpt-4o-2024-08-06":
            return 0.00000125 * input_length + 0.000005 * output_length
        elif model_name == "gpt-4o-mini-2024-07-18":
            return 0.000000075 * input_length + 0.0000003 * output_length

class Generator:
    def __init__(self, model_name, device, args):
        self.model_name = model_name
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'left'

        self.seed = args.seed
        self.top_k = 20
        self.top_p = 0.8
        self.temperature = 0.7
        self.num_trial = args.num_trial
        self.batch_size = args.batch_size
        
        self.is_reasoning = args.is_reasoning
        
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
        self.model = torch.nn.DataParallel(self.model)
    
    def model_generate(self, input_prompts, do_greedy):
        max_tokens = 1000
        if self.is_reasoning:
            max_tokens = 30000

        chat_prompts = []
        for input_prompt in input_prompts:
            chat_prompt = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": input_prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.is_reasoning # Switches between thinking and non-thinking modes. Default is True.
            )
            chat_prompts.append(chat_prompt)
        test_loader = DataLoader(chat_prompts, batch_size=self.batch_size, shuffle=False)
        
        if do_greedy:
            print("DO GREEDY")
            do_sample=False
            self.top_p=None
            self.top_k=None
            self.temperature=None
        else:
            print(f"Top-k: {self.top_k} | Top-p: {self.top_p} | Temperature: {self.temperature} | Num of Trial: {self.num_trial}")
            do_sample=True
        
        total_responses = []
        for t in range(self.num_trial):
            print(f"Trial-{t+1}")
            responses = []
            for batch in tqdm.tqdm(test_loader, total=len(test_loader)):
                inputs = self.tokenizer(batch, padding=True, return_tensors='pt').to("cuda")
                input_length = inputs["input_ids"].shape[-1]
                
                res_ = self.model.module.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.eos_token_id,
                    do_sample=do_sample,
                    top_k=self.top_k,
                    top_p=self.top_p,
                    temperature=self.temperature,
                    use_cache=True
                )
                res = np.array(self.tokenizer.batch_decode(res_[:,input_length:], skip_special_tokens=True))
                responses += res.tolist()
            total_responses.append(responses)
                
        responses_per_example = []
        for idx in range(len(total_responses[0])):
            res = []
            for i in range(self.num_trial):
                res.append(total_responses[i][idx])
            responses_per_example.append(res)
        responses = responses_per_example
                
        return responses