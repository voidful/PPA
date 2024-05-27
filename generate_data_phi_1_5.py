import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from pool import POOL

torch.set_default_device("cuda")

dataset = load_dataset("voidful/prompt-pool-eval")

model = AutoModelForCausalLM.from_pretrained("voidful/phi-1_5_chat_128k")
tokenizer = AutoTokenizer.from_pretrained("voidful/phi-1_5_chat_128k")



def get_pool_responses(item, max_tokens=2048):
    result_dict = {}
    for i in POOL:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"{i}\n{item['question']}"},
        ]
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(model.device)
        begin_gen_pos = inputs.shape[1]
        outputs = model.generate(inputs, max_new_tokens=max_tokens, do_sample=False)
        text = tokenizer.batch_decode(outputs[:, begin_gen_pos:-1])[0]
        result_dict[i] = text
    item['phi_1_5'] = json.dumps(result_dict, ensure_ascii=False)
    return item


dataset = dataset.map(get_pool_responses)
dataset.save_to_disk("prompt-pool-eval-phi-1_5")
