# Quantifying perturbation impacts for large language models

This is the official implementation of DBPA ([Quantifying perturbation impacts for large language models](https://arxiv.org/pdf/2412.00868)). Our paper proposes a new method for detecting and quantifying perturbations in large language model (LLM) responses.

# Overview

This repository includes the necessary code, prompts, and collected responses to reproduce the experiments and results presented in the NeurIPS workshop SFLLM paper.

# Installation

`pip install -e .`

Also check `requirements.txt`.

# Prerequisites

To use this repository, you will need an API to access the LLM. The simplest way to set this up is to create a Python file named `src/utils/openai_config.py` with the following structure:

```
def get_openai_config():
    openai_config = {
        api_key = api_key,
        api_version = api_version,
        api_endpoint = api_endpoint,
        model_deployment_id = model_deployment_id
    }
    return openai_config

def get_embedding_config():
    ada_config = {
        api_key = api_key,
        api_version = api_version,
        api_endpoint = api_endpoint,
        embedding_model_deployment_id = embedding_model_deployment_id
    }
    return ada_config
```

Replace the placeholders `api_key, api_version, api_endpoint, model_deployment_id, embedding_model_deployment_id` with your actual configuration values.

# Repository Structure

The core component of the repository is the `src` folder. Under `src`, there are three subcategories:

1. `data` contains the necessary code to generate data for the experiments. For the purpose of this paper, we generate synthetic sentences, and focus on perturbing the immediate neighbors to each word in the sentence.

2. `model` contains the core code to calculate distance between the original response and the perturbed response. For this paper, we provide two methods to approximate the distance---JSD and energy distance.

3. `utils` contains the code to setup and query LLM + embedding models.

Finally, `exp` contains all the sampled LLM responses for the experiments in the paper. All the raw LLM responses will fall under the folder `responses`, and the processed responses will fall under the folder `scores`, which calculates the distance between the original response and the perturbed response. In order to generate the responses from scratch, you should do `run.py`. Alternatively, you can only use the plotting bit of `run.py` and generate the plots in the paper directly from the calculated scores.

# Running the experiments

1. Go through the prerequisities, and setup your API config. Crucially, you should make sure you can run `src/utils/openai_config.py` and `src/utils/setup_llm.py` before moving on to the next step.

2. If you wish to run the entire experiment---including LLM generation---you should go to the corresponding experiment folder, and do `run.py`.

3. (To save some time) you can use the plotting bit in `run.py` and directly generate the plots in the paper from the calculated scores.

# Cite

```
@article{rauba2024quantifying,
  title={Quantifying perturbation impacts for large language models},
  author={Rauba, Paulius and Wei, Qiyao and van der Schaar, Mihaela},
  journal={arXiv preprint arXiv:2412.00868},
  year={2024}
}
```
