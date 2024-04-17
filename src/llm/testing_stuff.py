from llm_utils import split_and_save_data, load_data_as_lists

raw_data_path = '~/scratch/CS-7641-Project/data/raw/generator=base~dataset=p1.00.jsonl'
#split_and_save_data(raw_data_path)

raw_data_path2 = '~/scratch/CS-7641-Project/data/raw/split/val.jsonl'
print(load_data_as_lists(raw_data_path2)[1][0:100])