#!/usr/bin/env python
"""
Test script to demonstrate the interface with public models.
No API keys required!
"""

import subprocess
import sys

def run_example(description, command):
    """Run an example command and display results."""
    print(f"\n{'='*60}")
    print(f"Example: {description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(command)}")
    print("-" * 40)
    
    result = subprocess.run(command, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("Warnings/Info:", result.stderr[:500])  # Truncate long warnings
    
    return result.returncode == 0

def main():
    """Run a series of examples with public models."""
    
    examples = [
        (
            "Basic age perturbation test",
            ["python", "src/dbpa/interface.py",
             "--text", "I am 25 years old. What are my health priorities?",
             "--change", "25 years old", "65 years old"]
        ),
        
        (
            "Gender perturbation with JSON output",
            ["python", "src/dbpa/interface.py",
             "--text", "As a male patient, what screenings do I need?",
             "--change", "male", "female",
             "--output-format", "json"]
        ),
        
        (
            "Multiple changes with verbose output",
            ["python", "src/dbpa/interface.py",
             "--text", "I am a 30-year-old living in Boston",
             "--change", "30-year-old", "50-year-old", "Boston", "Miami",
             "--verbose"]
        ),
        
        (
            "Using Jensen-Shannon divergence method",
            ["python", "src/dbpa/interface.py",
             "--text", "The treatment is effective",
             "--change", "effective", "ineffective",
             "--method", "jsd",
             "--permutations", "100"]  # Fewer permutations for speed
        ),
    ]
    
    print("Testing Hypothesis Testing Interface with Public Models")
    print("No API keys required - using GPT-2 and all-MiniLM-L6-v2")
    print("Note: First run will download models (~600MB total)")
    
    success_count = 0
    for description, command in examples:
        if run_example(description, command):
            success_count += 1
            print("✓ Example completed successfully")
        else:
            print("✗ Example failed")
    
    print(f"\n{'='*60}")
    print(f"Results: {success_count}/{len(examples)} examples successful")
    print(f"{'='*60}")
    
    if success_count == len(examples):
        print("\nAll examples ran successfully!")
        print("You can now use the interface with any text and perturbations.")
        print("See PUBLIC_MODELS_EXAMPLES.md for more examples and options.")
    
    return 0 if success_count == len(examples) else 1

if __name__ == "__main__":
    sys.exit(main())