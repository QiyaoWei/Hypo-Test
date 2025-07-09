import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import jensenshannon

def calculate_cosine_similarities(embeddings1, embeddings2=None):
    """
    Calculates pairwise cosine similarities. This helper function remains unchanged.
    If only one set of embeddings is provided, it calculates intra-set similarities.
    If two sets are provided, it calculates inter-set similarities.
    """
    if embeddings2 is None:
        # P0: Similarities within a single group [cite: 143]
        similarities = cosine_similarity(embeddings1)
        # Return only the upper triangle of the matrix to avoid duplicates
        return similarities[np.triu_indices_from(similarities, k=1)]
    else:
        # P1: Similarities between two different groups [cite: 143]
        return cosine_similarity(embeddings1, embeddings2).flatten()

def jensen_shannon_divergence_and_pvalue(embeddings1, embeddings2, num_permutations=1000, bins=30):
    """
    This is the corrected, replaceable function.
    It now correctly performs the permutation test on the high-dimensional embeddings
    as described in the paper's methodology[cite: 148, 151].
    
    NOTE: The function name is the same, but the inputs have changed from 1D similarity
    arrays to the 2D embedding arrays to enable the correct procedure.
    
    Args:
        embeddings1 (np.array): Embeddings from the original prompt (k, dim).
        embeddings2 (np.array): Embeddings from the perturbed prompt (k, dim).
        num_permutations (int): Number of permutations (B in Algorithm 1).
        bins (int): Number of bins for creating empirical distributions from similarities.
        
    Returns:
        tuple: (observed_jsd, p_value)
    """
    
    # Internal helper to get JSD from two sets of similarity scores
    def _calculate_jsd(scores1, scores2):
        min_val = min(scores1.min(), scores2.min())
        max_val = max(scores1.max(), scores2.max())
        bin_edges = np.linspace(min_val, max_val, bins + 1)
        hist1, _ = np.histogram(scores1, bins=bin_edges, density=True)
        hist2, _ = np.histogram(scores2, bins=bin_edges, density=True)
        return jensenshannon(hist1, hist2)

    # Step 1: Compute the OBSERVED test statistic (T_obs) from the initial embeddings.
    # This corresponds to steps II and III in the paper's procedure overview[cite: 135].
    p0_observed = calculate_cosine_similarities(embeddings1)
    p1_observed = calculate_cosine_similarities(embeddings1, embeddings2)
    observed_jsd = _calculate_jsd(p0_observed, p1_observed)

    # Step 2: Perform the permutation test as per Algorithm 1[cite: 164].
    # Pool the original high-dimensional embeddings ('Z' vector)[cite: 138].
    # These are assumed to be exchangeable under the null hypothesis[cite: 153].
    combined_embeddings = np.vstack((embeddings1, embeddings2))
    k = len(embeddings1)
    count = 0
    
    for _ in range(num_permutations):
        # Randomly permute the pooled embeddings by shuffling their indices
        permuted_indices = np.random.permutation(len(combined_embeddings))
        perm_group1 = combined_embeddings[permuted_indices[:k]]
        perm_group2 = combined_embeddings[permuted_indices[k:]]
        
        # Calculate the test statistic for this permutation
        p0_perm = calculate_cosine_similarities(perm_group1)
        p1_perm = calculate_cosine_similarities(perm_group1, perm_group2)
        perm_jsd = _calculate_jsd(p0_perm, p1_perm)
        
        if perm_jsd >= observed_jsd:
            count += 1
            
    # Step 3: Calculate the final p-value[cite: 164].
    # The (1+count)/(1+B) formula prevents p-values of zero[cite: 150].
    p_value = (count + 1) / (num_permutations + 1)
    
    return observed_jsd, p_value

### How to Use (Before vs. After)

if __name__ == '__main__':
    # Generate dummy high-dimensional embeddings
    k = 50
    dim = 128
    np.random.seed(42)
    embeddings_A = np.random.rand(k, dim)
    embeddings_B = np.random.rand(k, dim) + 0.3 # Shifted embeddings

    # You pass the embeddings directly.
    print("Running correct implementation:")
    effect_size, p_value = jensen_shannon_divergence_and_pvalue(embeddings_A, embeddings_B, num_permutations=999)

    print(f"Observed Effect Size (JSD): {effect_size:.4f}")
    print(f"P-value: {p_value:.4f}")