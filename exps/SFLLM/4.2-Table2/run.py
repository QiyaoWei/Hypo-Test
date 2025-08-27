import random
import sys
seed = int(sys.argv[1])
random.seed(seed)
model_id = sys.argv[2]

import json
import numpy as np

from dbpa.utils.setup_llm import get_responses, get_embeddings
from dbpa.data.generate_data import get_prompt
from dbpa.model.core import calculate_cosine_similarities, jensen_shannon_divergence_and_pvalue

# Main execution
prompt = get_prompt("John")
other_prompts = ["Please generate cardiovascular disease (CVD) guidelines from the National Institute for Health and Care Excellence (NICE) for John, who is a 70-year-old Chinese male with a BMI of 30. His blood pressure is 114/61 mmHg, and his total cholesterol is 6.7 mmol/L with an HDL cholesterol of 1.0 mmol/L. He is a current smoker, does not have diabetes, but has a family history of CVD.",
"Considering John's demographics and health status - a 70-year-old Chinese male, BMI of 30, blood pressure of 114/61 mmHg, total cholesterol 6.7 mmol/L, HDL cholesterol 1.0 mmol/L, current smoker, free of diabetes, with a family history of CVD - what would be the NICE CVD guidelines applicable to him?",
"Given that John is a 70-year-old Chinese man with a BMI of 30, blood pressure of 114/61 mmHg, total cholesterol of 6.7 mmol/L, HDL cholesterol of 1.0 mmol/L, currently smokes, does not have diabetes, and has a family history of CVD, what are the NICE guidelines for CVD prevention for him?",
"Using the NICE guidelines, what cardiovascular recommendations can be made for John, a 70-year-old, Chinese, with a BMI of 30, blood pressure of 114/61 mmHg, total cholesterol of 6.7 mmol/L, HDL cholesterol of 1.0 mmol/L, a current smoker, non-diabetic and with a family history of CVD?",
"Considering the specifics of John's health profile - a 70-year-old Chinese man, BMI of 30, blood pressure at 114/61 mmHg, total cholesterol at 6.7 mmol/L, HDL cholesterol at 1.0 mmol/L, a current smoker, no diabetes, and a family history of CVD - what NICE guidelines would apply for CVD prevention?",
"What NICE guidelines for CVD prevention would be suitable for John, given his profile - a 70-year-old Chinese man, with a BMI of 30, blood pressure of 114/61 mmHg, total cholesterol of 6.7 mmol/L, HDL cholesterol of 1.0 mmol/L, current smoker, doesn't have diabetes, and a family history of CVD?",
"John, a 70-year-old Chinese man with a BMI of 30, blood pressure of 114/61 mmHg, total cholesterol of 6.7 mmol/L, HDL cholesterol of 1.0 mmol/L, a smoker, non-diabetic with a family history of CVD, needs a set of CVD prevention recommendations based on NICE guidelines, what would they be?",
"Given John's health data: 70 years old, Chinese, BMI of 30, blood pressure of 114/61 mmHg, total cholesterol of 6.7 mmol/L, HDL cholesterol of 1.0 mmol/L, a current smoker, no diabetes, and a family history of CVD, what are the relevant NICE guidelines for CVD risk management?",]


responses = {"base": get_responses(prompt)}
for index, other_prompt in enumerate(other_prompts):
    prompt = other_prompt
    responses[index] = get_responses(prompt)
with open(f"raw_{model_id.split('/')[-1]}_prompt_robust_{seed}.json", 'w') as f:
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

with open(f"{model_id.split('/')[-1]}_prompt_robust_{seed}.json", 'w') as f:
    json.dump(response_stats, f)

# plotting - only run if all necessary files exist
import os

model_ids = ["HuggingFaceTB/SmolLM-135M", "Gustavosta/MagicPrompt-Stable-Diffusion", "microsoft/Phi-3-mini-4k-instruct", "openai-community/gpt2", "mistralai/Mistral-7B-Instruct-v0.2", "meta-llama/Meta-Llama-3.1-8B-Instruct", "google/gemma-2-9b-it", "gpt-35-1106-vdsT-AE", "SWNorth-gpt-4-0613-20231016"]

def count_change(model_id):
    
    with open(model_id + '_prompt_robust_9.json') as f:
        data1 = json.load(f)
    with open(model_id + '_prompt_robust_68.json') as f:
        data2 = json.load(f)
    with open(model_id + '_prompt_robust_145.json') as f:
        data3 = json.load(f)
    with open(model_id + '_prompt_robust_5998.json') as f:
        data4 = json.load(f)
    with open(model_id + '_prompt_robust_66215.json') as f:
        data5 = json.load(f)
    
    p_value = []
    effect_size = []
    for data in [data1, data2, data3, data4, data5]:
        p_value.append(sum([1 for value in data.values() if value["p_value"] < 0.05]) / 8)
        effect_size.extend([value["jsd"] for value in data.values()])
    return p_value, effect_size

# Check if all required files exist before attempting to generate statistics
all_files_exist = True
for model_id in model_ids:
    model = model_id.split("/")[-1]
    for seed in [9, 68, 145, 5998, 66215]:
        if not os.path.exists(f"{model}_prompt_robust_{seed}.json"):
            all_files_exist = False
            break
    if not all_files_exist:
        break

if all_files_exist:
    print("\n=== Generating statistics across all models and seeds ===")
    all_change = dict()
    for model_id in model_ids:
        model = model_id.split("/")[-1]
        all_change[model] = count_change(model)

    for model, (p_value, effect_size) in all_change.items():
        print(model)
        print(f"P-value rejection rate: {np.mean(p_value):.4f} +- {np.std(p_value):.4f}")
        print(f"Effect size (JSD): {np.mean(effect_size):.4f} +- {np.std(effect_size):.4f}")
        print()
else:
    print("\nNote: Not all result files are present. Run all models and seeds to generate complete statistics.")
    print("Use: bash run.sh")
