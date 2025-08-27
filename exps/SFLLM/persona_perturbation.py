import matplotlib.pyplot as plt
import json
from matplotlib.lines import Line2D
from tqdm import tqdm
import numpy as np

from dbpa.utils.setup_llm import get_responses, get_embeddings
from dbpa.data.generate_data import get_prompt
from dbpa.model.core import calculate_cosine_similarities, jensen_shannon_divergence_and_pvalue

# Main execution
baseline_prefix = "Act as a doctor. "
prefixes = [
    'Act as a doctor. ', 'Act as a nurse. ', 'Act as a medical practitioner. ',
    'Act as a medical supervisor. ', 'Act as a comedian. ', 'Act as a robot from the future. ',
    'Act as a NeurIPS reviewer. ', 'Act as a child. '
]

# Vary lengths from 100 to 900 across prefixes
min_len, max_len = 100, 900
lengths = [int(x) for x in np.linspace(min_len, max_len, len(prefixes))]

# Get baseline (length of 100)
baseline_prompt = get_prompt("John", prefix=baseline_prefix, length=100)
baseline_responses = get_responses(baseline_prompt)
baseline_embeddings = get_embeddings(baseline_responses)
baseline_similarities = calculate_cosine_similarities(baseline_embeddings)

results = []

# Generate perturbed responses and calculate JSD, p-value, and JSD standard deviation
for prefix, length in tqdm(zip(prefixes, lengths), desc="Processing prefixes", total=len(prefixes)):
    perturbed_prompt = get_prompt("John", prefix=prefix, length=length)
    perturbed_responses = get_responses(perturbed_prompt)
    perturbed_embeddings = get_embeddings(perturbed_responses)

    # Uncomment to save responses
    # with open(f"{prefix.strip().replace(' ', '_')}.json", 'w') as f:
    #     json.dump(perturbed_responses, f)

    perturbed_similarities = calculate_cosine_similarities(perturbed_embeddings, baseline_embeddings)

    jsd, p_value = jensen_shannon_divergence_and_pvalue(
        # baseline_similarities, perturbed_similarities)
        baseline_embeddings, perturbed_embeddings)

    results.append({
        'prefix': prefix,
        'length': length,
        'jsd': jsd,
        'p_value': p_value
    })

# Print results
print("Results:")
for result in results:
    print(f"Prefix: {result['prefix']} (Length: {result['length']})")
    print(f"JSD: {result['jsd']:.6f}")
    print(f"p-value: {result['p_value']:.6f}")
    print()

# Save
with open('cvd_guidelines_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("Results saved to 'cvd_guidelines_results.json'")
