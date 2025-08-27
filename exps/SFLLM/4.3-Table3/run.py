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
for model in ["openai-community/gpt2"]: #["HuggingFaceTB/SmolLM-135M", "Gustavosta/MagicPrompt-Stable-Diffusion", "microsoft/Phi-3-mini-4k-instruct", "openai-community/gpt2", "mistralai/Mistral-7B-Instruct-v0.2", "meta-llama/Meta-Llama-3.1-8B-Instruct", "google/gemma-2-9b-it"]:
    responses[model] = get_responses(prompt, model)
with open(f"raw_gpt-35-1106-vdsT-AE_alignment_{seed}.json", 'w') as f:
    json.dump(responses, f)
response_embeddings = {key: get_embeddings(responses[key]) for key in responses.keys()}

response_similarities = {}
response_stats = dict()
for key, value in response_embeddings.items():
    if key == "base":
        response_similarities[key] = calculate_cosine_similarities(value)
    else:
        response_similarities[key] = calculate_cosine_similarities(value, response_embeddings["base"])
        # jsd, p_value, jsd_std = jensen_shannon_divergence_and_pvalue(response_similarities["base"], value)
        jsd, p_value = jensen_shannon_divergence_and_pvalue(response_embeddings["base"], value)
        response_stats[key] = {
            'jsd': jsd,
            'p_value': p_value
        }

with open(f"gpt-35-1106-vdsT-AE_alignment_{seed}.json", 'w') as f:
    json.dump(response_stats, f)
    
    
# plotting - only run if all necessary files exist
import os

# Check if all required files exist
all_files_exist = True
for seed in [9, 68, 145, 5998, 66215]:
    if not os.path.exists(f'gpt-35-1106-vdsT-AE_alignment_{seed}.json'):
        all_files_exist = False
        break

if all_files_exist:
    print("\n=== Generating alignment statistics across all seeds ===")
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
            effect_size.append(data[key]['jsd'])
            p_value.append(data[key]['p_value'])
        print(f"\nModel: {key}")
        print(f"Effect size (JSD): {np.mean(effect_size):.4f} +- {np.std(effect_size):.4f}")
        print(f"P-value: {np.mean(p_value):.4f} +- {np.std(p_value):.4f}")
else:
    print("\nNote: Not all result files are present. Run all seeds to generate complete statistics.")
    print("Use: bash run.sh")
