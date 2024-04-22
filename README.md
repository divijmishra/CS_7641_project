# CS 7641 Project
Repository for CS 7641 project group 46, Spring 2024.

To finetune LLaMA-2:
1. Clone this repo. 
2. Run /src/server_setup_files/setup_conda_env.sh to setup the Mamba env with required packages. 
3. Run /src/get_data.sh to download data, followed by /src/llm/save_data_split.py to save splits separately.
4. Run /src/llm/finetune.py to finetune LLaMA-2. 
5. Run /src/llm/evaluate.py to evaluate the finetuned model on a test split.

To run topic-modeling, you can ................

Directory structure: 

Server helper files:
* /src/server_setup_files/: Contains bash files to simplify environment setup on PACE. \
* /src/server_setup_files/setup_miniconda.sh: Downloads MiniForge and installs mamba in the scratch directory on PACE. \
* /src/server_setup_files/setup_conda_env.sh: Creates a mamba env with packages for LLM fine-tuning. \
* /src/server_setup_files/init_pace.sh: Requests a cluster on PACE acc to GPU requirements. \
* /src/server_setup_files/init_conda.sh: Init conda after conda installation (not used).

Downloading data:
* /src/get_data.sh: Downloads GROVER data.

Topic clustering files:
* /notebooks/title_topic_clustering/title_topic_clustering.ipynb: Contains code for the midpoint checkpoint (topic clustering on article title word embeddings)
* /notebooks/title_topic_clustering/.....................

LLM fine-tuning files:
* /src/llm/save_data_split.py: Reads the GROVER data and saves each split separately. \
* /src/llm/llm_utils.py: Contains various utility functions used in finetune.py and evaluate.py. \
* /src/llm/finetune.py: Finetunes LLaMA-2 on the GROVER dataset and saves the resultant model. \
* /src/llm/evaluate.py: Evaluates pretrained/finetuned LLaMA-2 on the GROVER test split. \
* /src/get_data.sh: Retrieves data according to https://github.com/rowanz/grover/tree/master/generation_examples. \
* /src/server_setup_files/: Contains bash files to simplify environment setup on PACE. \
* /src/server_setup_files/setup_miniconda.sh: Downloads MiniForge and installs mamba in the scratch directory on PACE. \
* /src/server_setup_files/setup_conda_env.sh/: Creates a mamba env with packages for LLM fine-tuning. \