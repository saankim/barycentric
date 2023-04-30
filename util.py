import pandas as pd
from datetime import datetime
import pickle
import os
from IPython.core.display import Markdown, Latex, display
from datetime import datetime

dumpdir = "data/dump/"


def Dump(data, name, path=""):
    if not path:
        os.makedirs(os.path.join(dumpdir, name), exist_ok=True)
        path = os.path.join(
            dumpdir, name, f"{datetime.now().strftime('%Y.%m.%d_%H:%M:%S')}.pkl"
        )
    with open(path, "wb") as f:
        pickle.dump(data, f)


def Load(name):
    if not name:
        raise ValueError("name is empty")

    def get_latest_file(directory):
        files = os.listdir(directory)
        files = [file for file in files if file.endswith(".pkl")]
        if not files:
            raise FileNotFoundError(
                f"No pickle files found in the directory: {directory}"
            )

        latest_file = max(
            files, key=lambda x: datetime.strptime(x, "%Y.%m.%d_%H:%M:%S.pkl")
        )
        return os.path.join(directory, latest_file)

    path = get_latest_file(os.path.join(dumpdir, name))
    return Load_path(path)


def Load_path(path):
    with open(path, "rb") as f:
        return pickle.load(f)
