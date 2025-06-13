import argparse
import json
import sys
from pathlib import Path

# Add the src directory to the path so we can import from dbpa
sys.path.append(str(Path(__file__).parent.parent))

from dbpa.model.core import quantify_perturbations

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
        print("-" * 50)
    
    try:
        # Compute perturbation metrics
        statistic, p_value = quantify_perturbations(
            text_orig=args.text,
            change=changes,
            method=args.method,
            distance=args.distance,
            num_permutations=args.permutations
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
    
    except Exception as e:
        print(f"Error during computation: {e}")
        sys.exit(1)

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