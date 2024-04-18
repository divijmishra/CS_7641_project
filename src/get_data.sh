#!/bin/bash

# Retrieve data as per https://github.com/rowanz/grover/tree/master/generation_examples
# We choose to 

mkdir data/
mkdir data/raw/
mkdir models/
mkdir models/llm_fine-tuning
mkdir models/text_topic_modeling
mkdir models/title_topic_clustering

wget https://storage.googleapis.com/grover-models/generation_examples/generator=base~dataset=p1.00.jsonl \
    -O data/raw/generator=base~dataset=p1.00.jsonl

wget https://storage.googleapis.com/grover-models/generation_examples/generator=base~dataset=p1.00.jsonl \
    -O data/raw/generator=base~dataset=p0.94.jsonl

wget https://storage.googleapis.com/grover-models/generation_examples/generator=base~dataset=p1.00.jsonl \
    -O data/raw/generator=mega~dataset=p0.94.jsonl
    