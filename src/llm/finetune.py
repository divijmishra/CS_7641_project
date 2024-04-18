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
)
print("Imports completed.")

# set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# paths
data_path = "data/"

MODEL_NAME = "meta-llama/Llama-2-7b-hf"
MAX_TOKENS = 1024  
# max for llama-2-7b is 4096, trying this if it speeds things up

seed = 42

############
train_sample, num_epochs = 2000, 5
print(f"Train sample: {train_sample}, num_epochs: {num_epochs}")
############

# for the output model
output_dir = "results/"
output_merged_dir = f"results/train_{train_sample}_epochs_{num_epochs}"



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
print("Model loaded.")

# load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)
tokenizer.pad_token = tokenizer.eos_token
print("Tokenizer loaded.")


############ LOADING DATASET

# loading raw dataset
generator, p = "mega", "0.94"
partial_path = data_path + f"splits/generator={generator}~dataset=p{p}/"
data_files = {
    "train": partial_path + "train.jsonl",  # 10k
    # "val" : partial_path + "val.jsonl",  # 3k
    # "test": partial_path + "test.jsonl",  # 12k, won't need test data
}
dataset = load_dataset("json", data_files=data_files)
print("Dataset loaded.")

# preprocessing the dataset
_preproc_func = partial(
        tokenize_text,
        tokenizer=tokenizer,
        max_length=MAX_TOKENS    
    )
for split in dataset.keys():
    # create prompts for each data point
    dataset[split] = dataset[split].map(create_prompt)
    
    # remove unnecessary columns
    dataset[split] = dataset[split].remove_columns([
        'article', 'domain', 'title', 'date', 'authors', 'ind30k',
        'url', 'orig_split', 'split', 'random_score', 'top_p',
    ])
    
    # tokenize
    dataset[split] = dataset[split].map(_preproc_func)
    
    # remove more columns now
    dataset[split] = dataset[split].remove_columns(['text', 'label'])
    
    # shuffle
    dataset = dataset.shuffle(seed=seed)
    
    # sampling only 1k training data due to time constraints
    dataset[split] = dataset[split].select(range(train_sample))
    
print("Dataset preprocessed.")


############ PREPARE MODEL WITH LORA

# enable gradient checkpointing to reduce memory usage
model.gradient_checkpointing_enable()

# prepare the model for training
model = prepare_model_for_kbit_training(model)

# create PEFT config
target_modules = [
    'gate_proj', 'down_proj', 'up_proj', 'q_proj', 'v_proj', 'k_proj', 'o_proj',
]
print(target_modules)
peft_config = LoraConfig(
    r=16,
    lora_alpha=64,
    target_modules=target_modules,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, peft_config)

# print information about the %age of trainable parameters
print_trainable_parameters(model)
# All Parameters: 6,778,392,576 || Trainable Parameters: 39,976,960 || Trainable Parameters %: 0.589770503135875


############ TRAINING THE MODEL

trainer = Trainer(
    model=model,
    train_dataset=dataset["train"],  # it automatically infers that the relevant tokens are in dataset["train"]["input_ids"]
    # eval_dataset=dataset["val"],
    args=TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        warmup_steps=2,
        num_train_epochs=num_epochs,
        # learning_rate=2e-4,
        weight_decay=0.001,
        fp16=True,
        logging_steps=25,
        optim="paged_adamw_32bit",
        max_grad_norm=0.3,
        # lr_scheduler_type="cosine",
        
    ),
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

# start training
print("Training...")
train_result = trainer.train(),
print("Training finished.")
# metrics = train_result.metrics
# trainer.log_metrics("train", metrics)
# trainer.save_metrics("train", metrics)
# trainer.save_state()
# print(metrics)

# save model
os.makedirs(output_dir, exist_ok=True)
trainer.model.save_pretrained(output_dir)

# free memory (doesn't happen automatically?)
del model
del trainer
torch.cuda.empty_cache()


############ MERGE AND PUSH FINE-TUNED MODEL

# merge LoRA adapters into the model
model = AutoPeftModelForCausalLM.from_pretrained(
    output_dir, device_map="auto", torch_dtype=torch.bfloat16
)
model = model.merge_and_unload()

# save fine-tuned model
output_merged_dir = output_merged_dir
os.makedirs(output_merged_dir, exist_ok=True)
model.save_pretrained(output_merged_dir, safe_serialization=True)

# save tokenizer for easy inference
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.save_pretrained(output_merged_dir)

# fine-tuned model name on Hugging Face Hub
new_model = "divijmishra/grover-base-p1.00-llama-2-7b"

# # push fine-tuned model and tokenizer to Hugging Face Hub
# model.push_to_hub(new_model, use_auth_token=True)
# tokenizer.push_to_hub(new_model, use_auth_token=True)

