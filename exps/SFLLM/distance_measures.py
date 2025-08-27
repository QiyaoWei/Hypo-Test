import matplotlib.pyplot as plt
import json
from matplotlib.lines import Line2D
from tqdm import tqdm

from dbpa.utils.setup_llm import get_responses, get_embeddings
from dbpa.data.generate_data import get_prompt
from dbpa.model.core import calculate_cosine_similarities, jensen_shannon_divergence_and_pvalue, compute_energy_distance_fn

# Main execution
baseline_prefix = "Act as a doctor. "
prefixes = ['Act as a doctor. ', 'Act as a nurse. ', 'Act as a medical practitioner. ', 'Act as a medical supervisor. ',
            'Act as a comedian. ', 'Act as a robot from the future. ', 'Act as a NeurIPS reviewer. ', 'Act as a child. ']

# Generate baseline responses
baseline_prompt = get_prompt("John", prefix=baseline_prefix)
baseline_responses = get_responses(baseline_prompt)
baseline_embeddings = get_embeddings(baseline_responses)
baseline_similarities = calculate_cosine_similarities(baseline_embeddings)

results = []

# Generate perturbed responses and calculate JSD, p-value, and JSD standard deviation
for prefix in tqdm(prefixes, desc="Processing prefixes"):
    perturbed_prompt = get_prompt("John", prefix=prefix)
    perturbed_responses = get_responses(perturbed_prompt)
    perturbed_embeddings = get_embeddings(perturbed_responses)

    # with open(f"{prefix}.json", 'w') as f:
    #     json.dump(perturbed_responses, f)
    
    perturbed_similarities = calculate_cosine_similarities(perturbed_embeddings, baseline_embeddings)
    
    # jsd, p_value, jsd_std = jensen_shannon_divergence_and_pvalue(baseline_embeddings, perturbed_embeddings)
    energy, p_value = compute_energy_distance_fn(baseline_embeddings, perturbed_embeddings, distance='cosine')
    
    results.append({
        'prefix': prefix,
        'energy': energy,
        # 'jsd': jsd,
        # 'jsd_std': jsd_std,
        'p_value': p_value
    })

# Print results
print("Results:")
for result in results:
    print(f"Prefix: {result['prefix']}")
    print(f"Energy Distance: {result['energy']:.6f}")
    print(f"p-value: {result['p_value']:.6f}")
    print()

# Save
with open('cvd_guidelines_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("Results saved to 'cvd_guidelines_results.json'")
