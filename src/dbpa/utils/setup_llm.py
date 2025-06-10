# This file contains a tutorial on creating an LLM with API keys
# Specifically, your goal is to define two functions, which you will then use to run the experiments
# def get_responses(prompt)
# def get_embeddings(text)
# We demonstrate using the AzureOpenAI API
# We also append an example of these two functions using open-source LLMs

from dbpa.utils.openai_config import get_llm_config, get_embedding_config # You need to create this function
from openai import AzureOpenAI
import numpy as np
import transformers
import torch
from sentence_transformers import SentenceTransformer

# Load configuration
llm_config = get_llm_config()
embedding_config = get_embedding_config()

# Initialize the AzureOpenAI client
llm_client = AzureOpenAI(
    api_key=llm_config["api_key"],
    api_version=llm_config["api_version"],
    azure_endpoint=llm_config["api_endpoint"],
)

embedding_client = AzureOpenAI(
    api_key=embedding_config["api_key"],
    api_version=embedding_config["api_version"],
    azure_endpoint=embedding_config["api_endpoint"],
)

def get_embeddings(texts, model_id="azure"):
    """
    Get embeddings for a list of texts using various providers.

    Args:
        texts (List[str]): Input text list.
        model_id (str): One of ['azure', 'ada', 'kalm', 'jasper', 'stella'].

    Returns:
        List[np.ndarray]: Embeddings.
    """
    if model_id == "ada":
        return get_azure_embeddings(texts)

    elif model_id == "kalm":
        model = SentenceTransformer("HIT-TMG/KaLM-embedding-multilingual-mini-v1")
        return model.encode(texts, convert_to_numpy=True)

    elif model_id == "jasper":
        model = SentenceTransformer("NovaSearch/jasper_en_vision_language_v1")
        return model.encode(texts, convert_to_numpy=True)

    elif model_id == "stella":
        model = SentenceTransformer("NovaSearch/stella_en_1.5B_v5")
        return model.encode(texts, convert_to_numpy=True)

    else:
        raise ValueError(f"Unsupported embedding model: {model_id}")
    
def get_azure_embeddings(texts):
    """
    Get embeddings for the input texts using Azure OpenAI API.
    Plural, because generally the input is a Monte-Carlo sample approximate of the LLM output distribution, i.e. list of strings.
    Args:
        texts (List[float]): The input texts to embed.
    
    Returns:
        List[float]: The embedding vector.
    """

    result = []
    for text in texts:
        if text != None and len(text) > 0:
            response = embedding_client.embeddings.create(
                input=text,
                model=embedding_config["embedding_model_deployment_id"],
            )
            result.append(response.data[0].embedding)
    return np.array(result)

def get_responses(prompt, model_id=None):
    """
    Get responses to the input prompt using Azure OpenAI API.
    Args:
        prompt (str): The input prompt.
    
    Returns:
        List[str]: The response to the prompt.
    """
    if model_id is None:
        response = llm_client.chat.completions.create(
            model=llm_config["model_deployment_id"],
            max_tokens=256,
            temperature=1,
            n=20,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return [choice.message.content for choice in response.choices]
    else: #if you wish to run the open-source models in the experiments
        generator = transformers.pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto"
        )
        outputs = generator(prompt, max_length=256, truncation=True, num_return_sequences=20, do_sample=True)
        return [output["generated_text"] for output in outputs]

# Here is what these two functions might look like if using an open-source model
# import tensorflow_hub as hub
# import transformers
# def get_embeddings(texts):
#     model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
#     return model(texts)

# def get_responses(prompt):
#     generator = transformers.pipeline(
#         "text-generation",
#         model="openai-community/gpt2",
#         model_kwargs={"torch_dtype": torch.bfloat16},
#         device_map="auto"
#     )
#     outputs = generator(prompt, max_length=256, truncation=True, num_return_sequences=20, do_sample=True)
#     if len(outputs) == 1:
#         return [output["generated_text"] for output in outputs[0]]
#     else:
#         return [output["generated_text"] for output in outputs]
