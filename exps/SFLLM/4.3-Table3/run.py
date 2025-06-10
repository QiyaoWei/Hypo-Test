import random
import sys
seed = int(sys.argv[1])
random.seed(seed)

import json
import numpy as np

from dbpa.utils.setup_llm import get_responses, get_embeddings
from dbpa.data.generate_data import get_prompt
from dbpa.model.core import calculate_cosine_similarities, jensen_shannon_divergence_and_pvalue

# Main execution
prompt = get_prompt("John")

responses = {"base": get_responses(prompt)}
for model in ["HuggingFaceTB/SmolLM-135M", "Gustavosta/MagicPrompt-Stable-Diffusion", "microsoft/Phi-3-mini-4k-instruct", "openai-community/gpt2", "mistralai/Mistral-7B-Instruct-v0.2", "meta-llama/Meta-Llama-3.1-8B-Instruct", "google/gemma-2-9b-it"]:
    responses[model] = get_responses(prompt, model)
with open(f"raw_gpt-35-1106-vdsT-AE_alignment_{seed}.json", 'w') as f:
    json.dump(responses, f)
response_embeddings = {key: get_embeddings(responses[key]) for key in responses.keys()}

response_similarities = {}
for key, value in response_embeddings.items():
    if key == "base":
        response_similarities[key] = calculate_cosine_similarities(value)
    else:
        response_similarities[key] = calculate_cosine_similarities(value, response_embeddings["base"]) 

response_stats = dict()
for key, value in response_similarities.items():
    if key == "base":
        continue
    jsd, p_value, jsd_std = jensen_shannon_divergence_and_pvalue(response_similarities["base"], value)
    response_stats[key] = {
        'jsd': jsd,
        'jsd_std': jsd_std,
        'p_value': p_value
    }

with open(f"gpt-35-1106-vdsT-AE_alignment_{seed}.json", 'w') as f:
    json.dump(response_stats, f)
    
    
# plotting

with open('gpt-35-1106-vdsT-AE_alignment_9.json') as f:
    data1 = json.load(f)
with open('gpt-35-1106-vdsT-AE_alignment_68.json') as f:
    data2 = json.load(f)
with open('gpt-35-1106-vdsT-AE_alignment_145.json') as f:
    data3 = json.load(f)
with open('gpt-35-1106-vdsT-AE_alignment_5998.json') as f:
    data4 = json.load(f)
with open('gpt-35-1106-vdsT-AE_alignment_66215.json') as f:
    data5 = json.load(f)

for key in data1.keys():
    effect_size = list()
    p_value = list()
    for data in [data1, data2, data3, data4, data5]:
        effect_size.append(data[key]['effect_size'])
        p_value.append(data[key]['p_value'])
    print(key)
    print(f"{np.mean(effect_size)} +- {np.std(effect_size)}")
    print(f"{np.mean(p_value)} +- {np.std(p_value)}")