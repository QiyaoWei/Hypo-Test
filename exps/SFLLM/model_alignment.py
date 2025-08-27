import matplotlib.pyplot as plt
import json
from matplotlib.lines import Line2D
from tqdm import tqdm
import numpy as np

from dbpa.utils.setup_llm import get_responses, get_embeddings
from dbpa.data.generate_data import get_prompt
from dbpa.model.core import calculate_cosine_similarities, jensen_shannon_divergence_and_pvalue

# List of model IDs to test
model_ids = [
    "HuggingFaceTB/SmolLM-135M",
    "Gustavosta/MagicPrompt-Stable-Diffusion",
    "microsoft/Phi-3-mini-4k-instruct",
    "openai-community/gpt2",
    "mistralai/Mistral-7B-Instruct-v0.2",
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "google/gemma-2-9b-it"
]

# Fixed prefixes
baseline_prefix = "Act as a doctor. "
prefixes = [
    'Act as a doctor. ', 'Act as a nurse. ', 'Act as a medical practitioner. ',
    'Act as a medical supervisor. ', 'Act as a comedian. ', 'Act as a robot from the future. ',
    'Act as a NeurIPS reviewer. ', 'Act as a child. '
]

# To hold all results
all_results = []

# Iterate over model IDs
for model_id in model_ids:
    print(f"\n=== Evaluating model: {model_id} ===\n")

    # Baseline prompt (no length argument)
    baseline_prompt = get_prompt("John", prefix=baseline_prefix)
    baseline_responses = get_responses(baseline_prompt)
    baseline_embeddings = get_embeddings(baseline_responses)
    baseline_similarities = calculate_cosine_similarities(baseline_embeddings)

    # Model-specific results
    model_results = []

    # Iterate over prefixes (no length variation)
    for prefix in tqdm(prefixes, desc=f"Processing for {model_id}", total=len(prefixes)):
        perturbed_prompt = get_prompt("John", prefix=prefix)
        perturbed_responses = get_responses(perturbed_prompt)
        perturbed_embeddings = get_embeddings(perturbed_responses)

        perturbed_similarities = calculate_cosine_similarities(perturbed_embeddings, baseline_embeddings)
        jsd, p_value = jensen_shannon_divergence_and_pvalue(
            # baseline_similarities, perturbed_similarities
            baseline_embeddings, perturbed_embeddings
        )

        result = {
            'model_id': model_id,
            'prefix': prefix,
            'jsd': jsd,
            'p_value': p_value
        }
        model_results.append(result)

        print(f"  Prefix: {prefix.strip()} | JSD: {jsd:.4f} | p={p_value:.4f}")

    all_results.extend(model_results)

# Save all results
with open('cvd_guidelines_all_models.json', 'w') as f:
    json.dump(all_results, f, indent=2)

print("\n✅ All results saved to 'cvd_guidelines_all_models.json'")
