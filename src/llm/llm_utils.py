import json
import os
import shutil
from tqdm import tqdm

import numpy as np
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix
)
import torch
from transformers import (
    pipeline, 
)

import bitsandbytes as bnb


############
def split_and_save_data(raw_data_path: str, generator, p):
    """
    Takes in the path to the raw data as a .json file, 
    separates them based on whether "split" is train, val, test,
    and saves each split as a separate json file in a new directory in raw_data_path

    Args:
        raw_data_path (str): Path to the raw data.

    Returns:
        None
    """
    raw_data_file_path = raw_data_path + f"generator={generator}~dataset=p{p}.jsonl"
    base_dir = os.path.dirname(os.path.dirname(raw_data_path))
    save_dir_path = os.path.join(base_dir, 'splits')
    save_dir_path = os.path.join(save_dir_path, f"generator={generator}~dataset=p{p}")
    if os.path.exists(save_dir_path):
        shutil.rmtree(save_dir_path)
    os.makedirs(save_dir_path)

    with open(raw_data_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            split = data.get('split')
            split_file_path = os.path.join(save_dir_path, f"{split}.jsonl")
            with open(split_file_path, 'a', encoding='utf-8') as split_file:
                split_file.write(json.dumps(data, ensure_ascii=False) + '\n')
    

############
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


############
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


############
def create_prompt(row):
    """
    Creates a text prompt from the given data point.
    If split="train", the prompt contains the response.
    If split="val"/"test", the prompt ends before the response.

    Args:
        row: Sample from the dataset - entry from the json files.
    """
    
    # preprocess the article
    """
    LLaMA-2 supports upto 4096 tokens. A basic heuristic is 0.75 tokens per word - let's go with 1.5 tokens per word just to be safe. 
    
    Our news articles are 1000s of words, with max being 13k. Let's truncate to 3000 words, just to be safe.
    """
    word_limit = 800
    text = truncate_article(row['article'], word_limit)

    if row['split'] == "train":
        label = f"{row['label']}"
    else:
        label = ""

    prompt = f"""
    Classify the news excerpt enclosed between <input></input> tags as being written by a human or machine. Return the answer as the corresponding label "human" or "machine". You may refer to the example below:
    
    Example start.
    <input>Sample excerpt from a news article.</input> 
    Output: machine
    Example end.
    
    Please classify the following news excerpt:
    
    <input>{text}</input> 
    Output: {label}"""

    # store prompt in a new key
    row['text'] = prompt
    
    return row


############
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


############
def tokenize_text(row, tokenizer, max_length):
    """
    Tokenizes a batch of data.

    Args:
        batch
        tokenizer
        max_length
    """
    return tokenizer(row["text"], max_length=max_length, padding=True, truncation=True)


############
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


############
def evaluate_predictions(y_true, y_pred):
    """
    Given a list of true and predicted labels ('human'/'machine'), generates various classification metrics

    Args:
        y_true 
        y_pred
    """
    
    labels = ['human', 'machine']
    mapping = {'human': 0, 'machine': 1}
    
    # convert to 0/1s
    def map_func(label):
        return mapping[label]
    
    y_true = np.vectorize(map_func)(y_true)
    y_pred = np.vectorize(map_func)(y_pred)
    
    # calculate accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {accuracy}")
    
    # generate classification report
    clf_report = classification_report(y_true, y_pred)
    print('\nClassification Report:')
    print(clf_report)
    
    # generate confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    print('\nConfusion Matrix:')
    print(conf_matrix)
    

############
def predict(dataset, model, tokenizer, device, finetuned):
    """
    Given a dataset split, evaluate the model on the split.
    
    Args:
        dataset: HF dataset split
        model
        tokenizer
    """
    y_pred = []
    
    for i in tqdm(range(len(dataset))):
        prompt = dataset[i]["text"]
        # print(prompt)
        
        input = tokenizer(prompt, return_tensors="pt").to(device)
        # print(input.input_ids[0, -5:])
        output = model.generate(
            **input,
            max_length = input.input_ids.shape[1] + 5,
            do_sample=False
        )
        # print(f"Input length: {input.input_ids.shape}")
        # print(f"Output length: {output[0].shape}")
        # print(output[0][-5:])
        generated_text = tokenizer.decode(output[0])
        # print(generated_text)
        answer = generated_text.split("Output:")[-1]
        if "human" in answer:
            # print("human found")
            pred="human"
        elif "machine" in answer:
            # print("machine found")
            pred="machine"
        else:
            # print("No valid prediction.")
            pred="human"  # by default, model should assume it's a human
        y_pred.append(pred)
        
        if finetuned:
            dataset[i]["finetuned_prediction"] = pred
        else:
            dataset[i]["pretrained_prediction"] = pred
        
        del input
        del output
        
    return y_pred