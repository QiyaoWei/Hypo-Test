import matplotlib.pyplot as plt
import json
from matplotlib.lines import Line2D
from tqdm import tqdm
import numpy as np

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

from dbpa.utils.setup_llm import get_responses, get_embeddings
from dbpa.data.generate_data import get_prompt
from dbpa.model.core import calculate_cosine_similarities, jensen_shannon_divergence_and_pvalue

# Initialize ROUGE scorer once
rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
smooth_fn = SmoothingFunction().method1

# Main execution
baseline_prefix = "Act as a doctor. "
prefixes = [
    'Act as a doctor. ', 'Act as a nurse. ', 'Act as a medical practitioner. ',
    'Act as a medical supervisor. ', 'Act as a comedian. ', 'Act as a robot from the future. ',
    'Act as a NeurIPS reviewer. ', 'Act as a child. '
]

embedding_models = ["ada", "azure", "kalm", "jasper", "stella"]

results = []

for embedding_model_id in embedding_models:
    print(f"\n=== Processing embedding model: {embedding_model_id} ===")
    
    # Generate baseline prompt & responses
    baseline_prompt = get_prompt("John", prefix=baseline_prefix)  # no length param now
    baseline_responses = get_responses(baseline_prompt)
    baseline_embeddings = get_embeddings(baseline_responses)
    baseline_similarities = calculate_cosine_similarities(baseline_embeddings)

    for prefix in tqdm(prefixes, desc=f"Processing prefixes with {embedding_model_id}"):
        perturbed_prompt = get_prompt("John", prefix=prefix)
        perturbed_responses = get_responses(perturbed_prompt)
        perturbed_embeddings = get_embeddings(perturbed_responses)

        # Embedding-based similarity & JSD
        perturbed_similarities = calculate_cosine_similarities(perturbed_embeddings, baseline_embeddings)
        jsd, p_value = jensen_shannon_divergence_and_pvalue(
            # baseline_similarities, perturbed_similarities
            baseline_embeddings, perturbed_embeddings
        )

        # --- BLEU & ROUGE calculation ---

        # BLEU: average sentence-level BLEU between corresponding responses
        bleu_scores = []
        for ref, hyp in zip(baseline_responses, perturbed_responses):
            # Tokenize by splitting (you can swap with better tokenizer)
            ref_tokens = ref.split()
            hyp_tokens = hyp.split()
            score = sentence_bleu(
                [ref_tokens], hyp_tokens, smoothing_function=smooth_fn, weights=(0.25, 0.25, 0.25, 0.25)
            )
            bleu_scores.append(score)
        avg_bleu = np.mean(bleu_scores)

        # ROUGE: average ROUGE scores across pairs
        rouge1_scores, rouge2_scores, rougeL_scores = [], [], []
        for ref, hyp in zip(baseline_responses, perturbed_responses):
            scores = rouge.score(ref, hyp)
            rouge1_scores.append(scores['rouge1'].fmeasure)
            rouge2_scores.append(scores['rouge2'].fmeasure)
            rougeL_scores.append(scores['rougeL'].fmeasure)

        avg_rouge1 = np.mean(rouge1_scores)
        avg_rouge2 = np.mean(rouge2_scores)
        avg_rougeL = np.mean(rougeL_scores)

        results.append({
            'embedding_model': embedding_model_id,
            'prefix': prefix,
            'jsd': jsd,
            'p_value': p_value,
            'bleu': avg_bleu,
            'rouge1': avg_rouge1,
            'rouge2': avg_rouge2,
            'rougeL': avg_rougeL
        })

# Print results with new metrics
print("\nResults:")
for result in results:
    print(f"Embedding model: {result['embedding_model']}, Prefix: {result['prefix']}")
    print(f"JSD: {result['jsd']:.6f}, p-value: {result['p_value']:.6f}")
    print(f"BLEU: {result['bleu']:.4f}, ROUGE-1: {result['rouge1']:.4f}, ROUGE-2: {result['rouge2']:.4f}, ROUGE-L: {result['rougeL']:.4f}")
    print()

# Save results
with open('cvd_guidelines_results_all_embeddings.json', 'w') as f:
    json.dump(results, f, indent=2)

print("All results saved to 'cvd_guidelines_results_all_embeddings.json'")
