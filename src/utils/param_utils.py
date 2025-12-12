import os
import json

def save_param(path, key, value):

    '''
    Save a parameter to a JSON file.

    If the file exists, it loads the current parameters, updates or adds
    the given key-value pair, and saves the updated dictionary back to the file.
    If the file does not exist, it creates a new JSON file containing the key-value pair.

    Parameters:
        path (str): Path to the JSON file where parameters are stored.
        key (str): The parameter name to save or update.
        value: The value to associate with the key.
    '''

    if os.path.exists(path):
        with open(path, "r") as f:
            params = json.load(f)
    else:
        params = {}

    params[key] = value

    with open(path, "w") as f:
        json.dump(params, f, indent=4)


def load_params(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}