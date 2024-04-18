import os
import json

from llm.llm_utils import (
    split_and_save_data,
    load_data_as_lists
)

# save data splits
generator, p = "base", "1.00"
data_path = "data/"
raw_data_path = data_path + "raw/"
split_and_save_data(raw_data_path, generator, p)