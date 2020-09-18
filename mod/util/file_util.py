import json


def read_json_file(filepath):
    with open(filepath, "r") as f:
        json_dict = json.load(f)
    return json_dict


def join_dictionary_keys(base_dict):
    joined_dict = {}
    for k, d in base_dict.items():
        print(f"Loading '{k}' settings...")
        joined_dict.update(d)
    return joined_dict
