import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, load_from_disk
from vllm import LLM, SamplingParams
import hashlib

from pool import POOL

torch.set_default_device("cuda")

model_name = "/proj/mtklmadm/models/Llama3-8b-Instruct"
dataset_save_path = "prompt-pool-eval-llama-8b"
dataset_intermediates_save_path = dataset_save_path + "_intermediate"


dataset = load_dataset("voidful/prompt-pool-eval")
print(dataset)
llm = LLM(model=model_name, trust_remote_code = True)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code = True)
sp = SamplingParams(temperature = 0, top_p = 0.01, max_tokens = 2048)

def chatify(x):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": x},
    ]
    message_with_chat_template = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize = False
    )
    return message_with_chat_template

def test_vllm_generation():
    chatify("Hi how are you today")
    res = llm.generate(inputs, sp)
    print(res)
    
test_vllm_generation()

def get_pool_responses(item, max_tokens=2048):
    
    # Create a hash for saving intermediate JSON files as a mapping, allowing the code to resume from the last saved state
    ha = hashlib.sha256(item["question"].encode()).hexdigest()
    instance_intermediate_save_fpath = os.path.join(dataset_intermediates_save_path, f"{ha}.json")
    
    if os.path.exists(instance_intermediate_save_fpath):
        with open(instance_intermediate_save_fpath, "r") as f:
            result_dict = json.load(f)
        item['llama_8b'] = json.dumps(result_dict, ensure_ascii=False)
        return item
    
    with torch.no_grad():
        ret = llm.generate([chatify(f"{i}\n{item['question']}") for i in POOL], sp)
    
    result_dict = {i: r.outputs[0].text for i,r in zip(POOL, ret)}
    
    with open(instance_intermediate_save_fpath, "w") as f:
        json.dump(result_dict, f, ensure_ascii=False, indent=4)
        
    item['llama_8b'] = json.dumps(result_dict, ensure_ascii=False)
    return item


dataset = dataset.map(get_pool_responses, num_proc=1)
dataset.save_to_disk(dataset_save_path)
