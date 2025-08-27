import json
from tqdm import tqdm

from dbpa.utils.setup_llm import get_responses, get_embeddings
from dbpa.model.core import calculate_cosine_similarities, jensen_shannon_divergence_and_pvalue

# List of model IDs to test
model_ids = [
    "openai-community/gpt2"
]

# List of (baseline, perturbed) prompt pairs
prompt_pairs = [
    ( #negation
        "Describe the effect that the occurrence of a specific historic event has on the history in the 1800s",
        "Describe the effect that the non-occurrence of a specific historic event has on the history in the 1800s"
    ),
    ( #time period
        "Describe the advancement of mathematics in the 1800s",
        "Describe the advancement of mathematics in the 2000s"
    ),
    ( #topic change
        "Describe the advancement of mathematics in the 1800s",
        "Describe the advancement of biology in the 1800s"
    )
]

# To hold all results
all_results = []

# Iterate over model IDs
for model_id in model_ids:
    print(f"\n=== Evaluating model: {model_id} ===\n")

    model_results = []

    # Pre-run and cache embeddings for all baselines (used as a reference set)
    print("Generating baseline responses for control group...")
    baseline_embedding_cache = {}
    baseline_similarities_cache = {}
    for i, (baseline_prompt, _) in enumerate(prompt_pairs):
        baseline_responses = get_responses(baseline_prompt)
        baseline_embeddings = get_embeddings(baseline_responses)
        baseline_embedding_cache[i] = baseline_embeddings
        baseline_similarities = calculate_cosine_similarities(baseline_embeddings)
        baseline_similarities_cache[i] = baseline_similarities
    print("Baseline responses generated and cached.")

    # For each pair, run the test again using cached baseline and fresh runs
    for i, (baseline_prompt, perturbed_prompt) in tqdm(
        enumerate(prompt_pairs), total=len(prompt_pairs),
        desc=f"Evaluating prompt pairs for {model_id}"
    ):
        print(f"\n→ Processing Pair {i + 1}")

        # Control group (repeat run of the baseline)
        control_responses = get_responses(baseline_prompt)
        control_embeddings = get_embeddings(control_responses)
        control_similarities = calculate_cosine_similarities(control_embeddings, baseline_embedding_cache[i])
        control_jsd, control_p_value = jensen_shannon_divergence_and_pvalue(
            # baseline_similarities_cache[i], control_similarities
            baseline_embedding_cache[i], control_embeddings
        )

        # Target group (perturbed)
        perturbed_responses = get_responses(perturbed_prompt)
        perturbed_embeddings = get_embeddings(perturbed_responses)
        perturbed_similarities = calculate_cosine_similarities(perturbed_embeddings, baseline_embedding_cache[i])
        perturbed_jsd, perturbed_p_value = jensen_shannon_divergence_and_pvalue(
            # baseline_similarities_cache[i], perturbed_similarities
            baseline_embedding_cache[i], control_embeddings
        )

        result = {
            'model_id': model_id,
            'pair_index': i,
            'baseline_prompt': baseline_prompt,
            'perturbed_prompt': perturbed_prompt,
            'control_jsd': control_jsd,
            'control_p_value': control_p_value,
            'perturbed_jsd': perturbed_jsd,
            'perturbed_p_value': perturbed_p_value
        }
        model_results.append(result)

        # print(f"  JSD: {jsd:.4f} ± {jsd_std:.4f} | p={p_value:.4f}")

    all_results.extend(model_results)

# Save all results
with open('prompt_perturbation_results_control_vs_target.json', 'w') as f:
    json.dump(all_results, f, indent=2)

print("\n✅ All results saved to 'prompt_perturbation_results_control_vs_target.json'")
