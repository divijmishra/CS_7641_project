import json
import os
import shutil

import torch
import bitsandbytes as bnb


def split_and_save_data(raw_data_path: str):
    """
    Takes in the path to the raw data as a .json file, 
    separates them based on whether "split" is train, val, test,
    and saves each split as a separate json file in a new directory in raw_data_path

    Args:
        raw_data_path (str): Path to the raw data.

    Returns:
        None
    """

    base_dir = os.path.dirname(os.path.dirname(raw_data_path))
    save_dir_path = os.path.join(base_dir, 'splits')
    if os.path.exists(save_dir_path):
        shutil.rmtree(save_dir_path)
    os.makedirs(save_dir_path)

    with open(raw_data_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            split = data.get('split')
            split_file_path = os.path.join(save_dir_path, f"{split}.jsonl")
            with open(split_file_path, 'a', encoding='utf-8') as split_file:
                split_file.write(json.dumps(data, ensure_ascii=False) + '\n')
    
    
def load_data_as_lists(data_path: str) -> tuple:
    """
    Given the path to a data split, loads the data,
    and returns the list of article texts and list of labels.
    Labels: "human" -> 0, "machine" -> 1.

    Args:
        data_path (str): Path to the data split.

    Returns:
        texts: List[str] of article texts.
        labels: List[int] of article labels.
    """
    texts = []
    labels = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            texts.append(data['article'])
            label = data.get('label')
            if label == 'human':
                labels.append(0)
            elif label == 'machine':
                labels.append(1)
            else:
                print("Neither human nor machine")
                labels.append(-1)
    return texts, labels


def truncate_article(text, word_limit):
    """
    LLaMA-2 supports upto 4096 tokens. A basic heuristic is 0.75 tokens per word - let's go with 1.5 tokens per word just to be safe. Our news articles are 1000s of words, with max being 13k. This will truncate the article to the last sentence such that the truncated article has <= word_limit words.
    
    Args: 
        text (str): The article text
        word_limit (int): 
    """
    sentences = text.split('.')
    
    total_words = 0
    truncated_sentences = []
    
    # iterate through sentences
    for sentence in sentences:
        words = sentence.split(' ') 
    
        total_words += len(words)
        if total_words > word_limit:
            break
        
        truncated_sentences.append(sentence)
        
    truncated_text = '.'.join(truncated_sentences) + '.'
    return truncated_text


def create_prompt(row):
    """
    Creates a text prompt from the given data point.

    Args:
        row: Sample from the dataset - entry from the json files.
    """
    
    # preprocess the article
    """
    LLaMA-2 supports upto 4096 tokens. A basic heuristic is 0.75 tokens per word - let's go with 1.5 tokens per word just to be safe. 
    
    Our news articles are 1000s of words, with max being 13k. Let's truncate to 3000 words, just to be safe.
    """
    word_limit = 3000
    text = truncate_article(row['article'], word_limit)
    
    # prompt
    prompt = f"""### Instructions:
Your task is to classify an excerpt from a news article as being human-generated or machine-generated. If it was machine-generated, respond with 'machine', else respond with 'human'.

### Input:
Classify the following news article excerpt as being human-generated or machine-generated:

{text}

### Response:
{row['label']}
### End
    """
    
    # store prompt in a new key
    row['text'] = prompt
    
    return row


def get_max_length(model):
    """
    Extracts maximum token length from the model configuration.

    model: HF model
    """

    max_length = None
    # find maximum sequence length in the model configuration and save it in “max_length” if found
    for length_setting in ["n_positions", "max_position_embeddings", "seq_length"]:
        max_length = getattr(model.config, length_setting, None)
        if max_length:
            print(f"Found max length: {max_length}")
            break
        
    # Set “max_length” to 1024 (default value) if maximum sequence length is not found in the model configuration
    if not max_length:
        max_length = 1024
        print(f"Using default max length: {max_length}")
    return max_length


def tokenize_text(row, tokenizer, max_length):
    """
    Tokenizes a batch of data.

    Args:
        batch
        tokenizer
        max_length
    """
    return tokenizer(row["text"], max_length=max_length, padding=True, truncation=True)


def print_trainable_parameters(model):
    """
    (Cool utility from a tutorial I found)
    Prints the number of trainable parameters in the model.

    model: PEFT model
    """

    trainable_params = 0
    all_param = 0

    for _, param in model.named_parameters():
        num_params = param.numel()
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel
        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params

    print(
        f"All Parameters: {all_param:,d} || Trainable Parameters: {trainable_params:,d} || Trainable Parameters %: {100 * trainable_params / all_param}"
    )
