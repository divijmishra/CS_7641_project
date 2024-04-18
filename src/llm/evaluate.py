############ IMPORTS
import os
os.environ['HF_HOME'] = './models/hf_cache'

from functools import partial
import torch
from datasets import (
    load_dataset,
    disable_caching,
)
disable_caching()
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    pipeline,
    logging,
    set_seed,
)
from trl import setup_chat_format
import bitsandbytes as bnb
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel,
    AutoPeftModelForCausalLM,
)
from llm.llm_utils import (
    create_prompt,
    get_max_length,
    tokenize_text,
    print_trainable_parameters,
    predict,
    evaluate_predictions,
)

print("Imports completed.")

# set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# paths
data_path = "data/"

finetuned = True
if finetuned == False:
    MODEL_NAME = "meta-llama/Llama-2-7b-hf"
else:
    train_sample, num_epochs = 2500, 4
    MODEL_NAME = f"./results/train_{train_sample}_epochs_{num_epochs}"
    # MODEL_NAME = f"./results/final"
print(f"MODEL_NAME: {MODEL_NAME}")
MAX_TOKENS = 1024  
# max for llama-2-7b is 4096, trying this if it speeds things up

seed = 42


############ LOADING THE QUANTIZED MODEL

# quantization configuration
if device == "cuda":
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_type="nf4",  
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
else:
    bnb_config = None

# load the model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    # device_map={"": 0},  # should try to load everything on the GPU
)
# get_max_length(model)  # 4096 for LLaMA-2 7B
model.to(device)
model.config.use_cache = False
model.generation_config.temperature=None
model.generation_config.top_p=None
print("Model loaded.")

# load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
print("Tokenizer loaded.")


############ LOADING DATASET

# loading raw dataset
generator, p = "base", "1.00"
partial_path = data_path + f"splits/generator={generator}~dataset=p{p}/"
data_files = {
    # "train": partial_path + "train.jsonl",  # 10k
    # "val" : partial_path + "val.jsonl",  # 3k
    "test": partial_path + "test.jsonl",  # 12k, won't need test data
}
dataset = load_dataset("json", data_files=data_files)
print("Dataset loaded.")

# preprocessing the dataset
for split in dataset.keys():
    # create prompts for each data point
    dataset[split] = dataset[split].map(create_prompt)
    
    # # remove unnecessary columns
    # dataset[split] = dataset[split].remove_columns([
    #     'article', 'domain', 'title', 'date', 'authors', 'ind30k',
    #     'url', 'orig_split', 'split', 'random_score', 'top_p',
    # ])
    
    # shuffle
    dataset = dataset.shuffle(seed=seed)
    
    # sample a tiny bit of val for eval
    if split in ["val", "test"]:
        dataset[split] = dataset[split].select(range(100))
    
print("Dataset preprocessed.")


############ EVALUATE THE MODEL

# generate predictions
y_true = [row["label"] for row in dataset["test"]]
y_pred = predict(dataset["test"], model, tokenizer, device, finetuned=False)
print("Generated predictions.")

# evaluate predictions
evaluate_predictions(y_true, y_pred)


############ STORE RESULTS

# store predicted labels 