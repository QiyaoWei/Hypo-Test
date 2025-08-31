# This file contains implementations for LLM and embedding functions
# Using public models that don't require API keys
# def get_responses(prompt)
# def get_embeddings(text)

import numpy as np
import transformers
import torch
from sentence_transformers import SentenceTransformer
import os

# Try to import Azure OpenAI for backwards compatibility
try:
    from openai import AzureOpenAI
    # Try to load configuration if it exists
    try:
        from dbpa.utils.llm_config import get_llm_config, get_embedding_config
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
        AZURE_AVAILABLE = True
    except ImportError:
        AZURE_AVAILABLE = False
except ImportError:
    AZURE_AVAILABLE = False

def get_embeddings(texts, model_id="public"):
    """
    Get embeddings for a list of texts using various providers.

    Args:
        texts (List[str]): Input text list.
        model_id (str): One of ['public', 'ada', 'kalm', 'jasper', 'stella'].
                       'public' uses all-MiniLM-L6-v2 (default, no API needed)

    Returns:
        List[np.ndarray]: Embeddings.
    """
    if model_id == "public":
        # Use all-MiniLM-L6-v2 as default public model
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        return model.encode(texts, convert_to_numpy=True)
    
    elif model_id == "ada":
        if not AZURE_AVAILABLE:
            print("Azure OpenAI not configured. Falling back to public model.")
            return get_embeddings(texts, model_id="public")
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

def get_responses(prompt, model_id="public"):
    """
    Get responses to the input prompt using LLM.
    Args:
        prompt (str): The input prompt.
        model_id (str): Model to use. 'public' uses GPT-2 (default), 
                       'azure' uses Azure OpenAI if configured,
                       or any Hugging Face model ID.
    
    Returns:
        List[str]: The response to the prompt.
    """
    if model_id == "azure":
        if not AZURE_AVAILABLE:
            print("Azure OpenAI not configured. Falling back to public model.")
            model_id = "public"
        else:
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
    
    # Use public models
    if model_id == "public":
        model_id = "openai-community/gpt2"  # Default public model
    
    generator = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32},
        device_map="auto" if torch.cuda.is_available() else None
    )
    
    # Generate multiple responses
    outputs = generator(
        prompt, 
        max_new_tokens=256, 
        truncation=True, 
        num_return_sequences=min(20, generator.model.config.vocab_size if hasattr(generator.model.config, 'vocab_size') else 20),
        do_sample=True,
        temperature=1.0,
        pad_token_id=generator.tokenizer.eos_token_id
    )
    
    # Extract just the generated part (remove the prompt)
    results = []
    for output in outputs:
        generated_text = output["generated_text"]
        # Remove the prompt from the beginning if it's there
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()
        results.append(generated_text)
    
    return results

# Default models used:
# - Embeddings: sentence-transformers/all-MiniLM-L6-v2 (public, no API needed)
# - LLM: openai-community/gpt2 (public, no API needed)
# 
# You can also use other models by specifying model_id parameter:
# - For embeddings: 'kalm', 'jasper', 'stella', or any sentence-transformers model
# - For LLM: any Hugging Face text generation model ID
