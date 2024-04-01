#!/bin/bash

# Retrieve data as per https://github.com/rowanz/grover/tree/master/generation_examples
# We choose to 

wget https://storage.googleapis.com/grover-models/generation_examples/generator=base~dataset=p1.00.jsonl \
    -O data/raw/generator=base~dataset=p1.00.jsonl
    