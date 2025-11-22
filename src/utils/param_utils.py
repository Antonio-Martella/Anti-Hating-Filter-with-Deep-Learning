import os
import json

def save_param(path, key, value):

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