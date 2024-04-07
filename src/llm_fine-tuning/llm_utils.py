



def split_and_save_data(raw_data_path: str, save_data_path: str):
    """
    Takes in the path to the raw data as a .json file, 
    separates them based on whether "split" is train, val, test,
    and saves each split as a separate json file in the specified path.

    Args:
        raw_data_path (str): Path to the raw data.
        save_data_path (str): Directory where the splits will be saved.

    Returns:
        None
    """
    
    
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