import os
os.environ['HF_HOME'] = './models/hf_cache'

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load tokenizer and model
model_name = "meta-llama/Llama-2-7b-hf"
# cache_dir = "~/scratch/CS-7641-Project/models/pretrained/"
tokenizer = AutoTokenizer.from_pretrained(model_name)
print("Loaded tokenizer.")
model = AutoModelForSequenceClassification.from_pretrained(model_name)
print("Loaded model.")

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define input text and labels
input_texts = ["Your input texts here", "Another input text here"]
labels = [1, 0]  # Example binary labels, adjust as needed

# Tokenize input texts
encoded_inputs = tokenizer(input_texts, padding=True, truncation=True, return_tensors="pt")
print("Inputs encoded.")
print(encoded_inputs)

# Move input tensors to device
input_ids = encoded_inputs["input_ids"].to(device)
attention_mask = encoded_inputs["attention_mask"].to(device)

# Perform inference
with torch.no_grad():
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
print("Model inference done.")

# Predict class probabilities
probabilities = torch.softmax(logits, dim=1)

# Get predicted labels
predicted_labels = torch.argmax(probabilities, dim=1)

# Print predicted labels
print("Predicted Labels:", predicted_labels.tolist())
