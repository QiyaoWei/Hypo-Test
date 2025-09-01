import argparse
import json
import sys
import numpy as np
from pathlib import Path

# Add the src directory to the path so we can import from dbpa
sys.path.append(str(Path(__file__).parent.parent))

from dbpa.utils.setup_llm import get_responses, get_embeddings
from dbpa.model.core import calculate_cosine_similarities, jensen_shannon_divergence_and_pvalue, compute_energy_distance_fn

def quantify_perturbations(text_orig, change, method='energy', distance='cosine', num_permutations=500, llm_model='public', embedding_model='public'):
    """
    Quantify perturbations between original text and modified text using statistical measures.
    
    Args:
        text_orig (str): Original text
        change (dict): Dictionary mapping original phrases to replacement phrases
        method (str): 'energy' for energy distance or 'jsd' for Jensen-Shannon divergence
        distance (str): Distance metric for energy distance ('cosine', 'l1', 'l2')
        num_permutations (int): Number of permutations for statistical testing
        llm_model (str): Model ID for LLM ('public' uses GPT-2, no API needed)
        embedding_model (str): Model ID for embeddings ('public' uses all-MiniLM-L6-v2, no API needed)
    
    Returns:
        tuple: (statistic_value, p_value)
    """
    # Apply changes to create modified text
    text_modified = text_orig
    for original_phrase, replacement_phrase in change.items():
        text_modified = text_modified.replace(original_phrase, replacement_phrase)
    
    # Generate LLM responses for both texts
    baseline_responses = get_responses(text_orig, model_id=llm_model)
    perturbed_responses = get_responses(text_modified, model_id=llm_model)
    
    # Get embeddings for the responses
    baseline_embeddings = get_embeddings(baseline_responses, model_id=embedding_model)
    perturbed_embeddings = get_embeddings(perturbed_responses, model_id=embedding_model)
    
    if method == 'energy':
        statistic, p_value = compute_energy_distance_fn(
            baseline_embeddings, perturbed_embeddings, distance=distance
        )
    elif method == 'jsd':
        # Calculate cosine similarities
        baseline_similarities = calculate_cosine_similarities(baseline_embeddings)
        perturbed_similarities = calculate_cosine_similarities(perturbed_embeddings, baseline_embeddings)
        
        statistic, p_value, _ = jensen_shannon_divergence_and_pvalue(
            # baseline_similarities, perturbed_similarities, num_bootstraps=num_permutations
            baseline_embeddings, perturbed_embeddings, num_bootstraps=num_permutations
        )
    else:
        raise ValueError(f"Invalid method: {method}. Use 'energy' or 'jsd'.")
    
    return statistic, p_value

def main():
    parser = argparse.ArgumentParser(
        description="Quantify text perturbations using statistical measures",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python interface.py --text "My age is 45 and I am male. What is my life expectancy?" --change "age is 45" "age is 55"
  
  # Multiple changes
  python interface.py --text "I am 30 years old and live in NYC" --change "30 years old" "40 years old" "NYC" "LA"
  
  # Using JSON file for complex changes
  python interface.py --text "Hello world" --change-file changes.json
  
  # Different statistical method
  python interface.py --text "Test text" --change "Test" "Modified" --method jsd
  
  # Custom parameters
  python interface.py --text "Sample text" --change "Sample" "Example" --distance l2 --permutations 1000
        """
    )
    
    parser.add_argument(
        '--text', 
        required=True, 
        help='Original text to analyze'
    )
    
    parser.add_argument(
        '--change', 
        nargs='+', 
        help='Pairs of original and replacement phrases (e.g., "old phrase" "new phrase")'
    )
    
    parser.add_argument(
        '--change-file', 
        help='JSON file containing changes as key-value pairs'
    )
    
    parser.add_argument(
        '--method', 
        choices=['energy', 'jsd'], 
        default='energy',
        help='Statistical method to use (default: energy)'
    )
    
    parser.add_argument(
        '--distance', 
        choices=['cosine', 'l1', 'l2'], 
        default='cosine',
        help='Distance metric for energy method (default: cosine)'
    )
    
    parser.add_argument(
        '--permutations', 
        type=int, 
        default=500,
        help='Number of permutations for statistical testing (default: 500)'
    )
    
    parser.add_argument(
        '--output-format', 
        choices=['plain', 'json'], 
        default='plain',
        help='Output format (default: plain)'
    )
    
    parser.add_argument(
        '--verbose', 
        action='store_true',
        help='Show detailed output'
    )
    
    parser.add_argument(
        '--llm-model',
        default='public',
        help='LLM model to use (default: public, uses GPT-2 which requires no API)'
    )
    
    parser.add_argument(
        '--embedding-model',
        default='public',
        help='Embedding model to use (default: public, uses all-MiniLM-L6-v2 which requires no API)'
    )

    args = parser.parse_args()
    
    # Parse changes
    changes = {}
    
    if args.change_file:
        try:
            with open(args.change_file, 'r') as f:
                changes = json.load(f)
        except FileNotFoundError:
            print(f"Error: Change file '{args.change_file}' not found")
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in change file: {e}")
            sys.exit(1)
    
    if args.change:
        if len(args.change) % 2 != 0:
            print("Error: Changes must be provided in pairs (original, replacement)")
            sys.exit(1)
        
        # Convert pairs to dictionary
        for i in range(0, len(args.change), 2):
            changes[args.change[i]] = args.change[i + 1]
    
    if not changes:
        print("Error: No changes specified. Use --change or --change-file")
        sys.exit(1)
    
    if args.verbose:
        print(f"Original text: {args.text}")
        print(f"Changes: {changes}")
        print(f"Method: {args.method}")
        if args.method == 'energy':
            print(f"Distance metric: {args.distance}")
        print(f"Permutations: {args.permutations}")
        print(f"LLM model: {args.llm_model}")
        print(f"Embedding model: {args.embedding_model}")
        print("-" * 50)
        print("Generating LLM responses (this may take a moment on first run to download models)...")
    
    # Compute perturbation metrics
    statistic, p_value = quantify_perturbations(
        text_orig=args.text,
        change=changes,
        method=args.method,
        distance=args.distance,
        num_permutations=args.permutations,
        llm_model=args.llm_model,
        embedding_model=args.embedding_model
    )
    
    # Output results
    if args.output_format == 'json':
        result = {
            'original_text': args.text,
            'changes': changes,
            'method': args.method,
            'statistic': float(statistic),
            'p_value': float(p_value)
        }
        if args.method == 'energy':
            result['distance_metric'] = args.distance
        
        print(json.dumps(result, indent=2))
    else:
        print(f"Statistic: {statistic:.6f}")
        print(f"P-value: {p_value:.6f}")
        
        if args.verbose:
            print(f"\nInterpretation:")
            if p_value < 0.05:
                print("- The perturbation is statistically significant (p < 0.05)")
            else:
                print("- The perturbation is not statistically significant (p >= 0.05)")
            
            print(f"- Higher statistic values indicate larger perturbations")


def create_example_change_file():
    """Create an example JSON change file"""
    example_changes = {
        "age is 45": "age is 55",
        "male": "female",
        "life expectancy": "retirement age"
    }
    
    with open('example_changes.json', 'w') as f:
        json.dump(example_changes, f, indent=2)
    
    print("Created example_changes.json")

if __name__ == "__main__":
    # Check if user wants to create example file
    if len(sys.argv) > 1 and sys.argv[1] == '--create-example':
        create_example_change_file()
    else:
        main()
