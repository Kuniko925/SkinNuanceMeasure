import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent # Path to folder opened file

def main():

    db_name = 'milk10k/dermoscopic'
    model_name = 'MobileNetV3'

    filepath = PROJECT_ROOT / f"skin{db_name}/results/{model_name}_valid_results.csv"
    usecols = ["filename", "filepath", "label", "pred", "skin tone", "distance"]
    df = pd.read_csv(filepath, usecols=usecols)
    df["db"] = db_name
    df["model"] = model_name
    df["label"] = df["label"].astype(str)
    df["pred"] = df["pred"].astype(str)


if __name__== "__main__":

    main()