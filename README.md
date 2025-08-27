# Statistical Hypothesis Testing for Auditing Robustness in Language Models

This is the official implementation of Statistical Hypothesis Testing for Auditing Robustness in Language Models ([arxiv](https://arxiv.org/abs/2506.07947)). Our paper proposes a new method for detecting and quantifying perturbations in large language model (LLM) responses.

# Overview

This repository includes the necessary code, prompts, and collected responses to reproduce the experiments and results presented in the NeurIPS workshop SFLLM paper.

# Installation

`pip install -e .`

You should also make sure you have the packages in `requirements.txt`.

# Prerequisites

To use this repository, you will need an API to access the LLM. The simplest way to set this up is to create a Python file named `src/dbpa/utils/llm_config.py` with the following structure:

```
def get_llm_config():
    llm_config = dict(
        api_key = api_key,
        api_version = api_version,
        api_endpoint = api_endpoint,
        model_deployment_id = model_deployment_id
    )
    return llm_config

def get_embedding_config():
    ada_config = dict(
        api_key = api_key,
        api_version = api_version,
        api_endpoint = api_endpoint,
        embedding_model_deployment_id = embedding_model_deployment_id
    )
    return ada_config
```

Replace the placeholders `api_key, api_version, api_endpoint, model_deployment_id, embedding_model_deployment_id` with your actual configuration values.

# Repository Structure

The core component of the repository is the `src` folder. Under `src`, there are three subcategories:

1. **`data`** contains the necessary code to generate data for the experiments. For the purpose of this paper, we generate synthetic sentences, and focus on perturbing the immediate neighbors to each word in the sentence.

2. **`model`** contains the core code to calculate distance between the original response and the perturbed response. For this paper, we provide two methods to approximate the distance---JSD and energy distance.

3. **`utils`** contains the code to setup and query LLM + embedding models.

Additionally, **`exp`** contains all the sampled LLM responses for the experiments in the paper. All the raw LLM responses are stored under the `responses` folder, and the processed responses are stored under the `scores` folder, which calculates the distance between the original response and the perturbed response.

# Quick Start - Command Line Interface

For quick text perturbation analysis, you can use our command-line interface:

## Basic Usage

```bash
# Analyze a simple text perturbation
python src/dbpa/interface.py --text "My age is 45 and I am male. What is my life expectancy?" --change "age is 45" "age is 55"

# Multiple changes at once
python src/dbpa/interface.py --text "I am 30 years old and live in NYC" --change "30 years old" "40 years old" "NYC" "LA"

# Get detailed output with interpretation
python src/dbpa/interface.py --text "Hello world" --change "Hello" "Hi" --verbose
```

## Advanced Usage

```bash
# Use Jensen-Shannon divergence instead of energy distance
python src/dbpa/interface.py --text "Test text" --change "Test" "Modified" --method jsd

# Customize distance metric and number of permutations
python src/dbpa/interface.py --text "Sample text" --change "Sample" "Example" --distance l2 --permutations 1000

# Output results in JSON format
python src/dbpa/interface.py --text "Hello world" --change "Hello" "Hi" --output-format json
```

## Using JSON Configuration Files

For complex perturbations, create a JSON file with your changes:

```bash
# Create an example configuration file
python src/dbpa/interface.py --create-example

# Use the configuration file
python src/dbpa/interface.py --text "My age is 45 and I am male. What is my life expectancy?" --change-file example_changes.json
```

Example JSON format:
```json
{
  "age is 45": "age is 55",
  "male": "female", 
  "life expectancy": "retirement age"
}
```

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--text` | Original text to analyze | Required |
| `--change` | Pairs of original and replacement phrases | - |
| `--change-file` | JSON file containing changes | - |
| `--method` | Statistical method (`energy` or `jsd`) | `energy` |
| `--distance` | Distance metric (`cosine`, `l1`, `l2`) | `cosine` |
| `--permutations` | Number of permutations for testing | `500` |
| `--output-format` | Output format (`plain` or `json`) | `plain` |
| `--verbose` | Show detailed output and interpretation | `false` |

# Running the Full Experiments

## Prerequisites

1. **Setup API Configuration**: Ensure you can run `src/utils/openai_config.py` and `src/utils/setup_llm.py` before proceeding.

2. **Install Dependencies**: Make sure all requirements are installed via `pip install -e .`

## Experiment Execution

### Full Experiment Pipeline
If you wish to run the entire experiment including LLM generation:

```bash
cd exp/SFLLM
python experiment_file.py
```

This will:
- Generate new LLM responses
- Calculate perturbation distances
- Generate plots and analysis

Note that experiments are correspondingly named to match the paper, specifically Table 2

# Programmatic Usage

You can also use the core functionality directly in your Python code:

```python
from dbpa.model.core import quantify_perturbations

# Basic usage
text_orig = 'My age is 45 and I am male. What is my life expectancy?'
change = {'age is 45': 'age is 55'}

statistic, p_value = quantify_perturbations(text_orig, change)
print(f"Statistic: {statistic:.6f}, P-value: {p_value:.6f}")

# Advanced usage with custom parameters
statistic, p_value = quantify_perturbations(
    text_orig, 
    change, 
    method='jsd',  # or 'energy'
    distance='l2',  # for energy method
    num_permutations=1000
)
```

# Contributing

We welcome contributions! Please ensure your code follows the existing style and includes appropriate tests.

# Citation

```bibtex
@article{rauba2025statistical,
  title={Statistical Hypothesis Testing for Auditing Robustness in Language Models},
  author={Rauba, Paulius and Wei, Qiyao and van der Schaar, Mihaela},
  journal={arXiv preprint arXiv:2506.07947},
  year={2025}
}
```

# Support

For questions or issues, please open a GitHub issue or contact the authors.
