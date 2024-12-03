"""
Utilities for data loading, etc.
"""

import os
import json

def load_json_data(data_path, split) -> list:
    """
    read data from dataset file
    Args:
        args:

    Returns:
    """

    assert split in ["train", "test"], "Data split can only be train or test!"

    file_path = os.path.join(data_path, "test.json")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"can not find dataset file : {file_path}")
    json_data = json.load(open(file_path, 'r'))
    return json_data