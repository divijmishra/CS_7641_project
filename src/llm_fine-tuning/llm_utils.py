



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

    save_dir_path = os.path.join(raw_data_path, 'split')
    if os.path.exists(save_dir_path):
        shutil.rmtree(save_dir_path)
    os.makedirs(save_dir_path)

    with open(raw_data_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            split = data.get('split')
            split_file_path = os.path.join(save_data_path, f"{split}.jsonl")
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
                labels.append(-1)
    return texts, labels