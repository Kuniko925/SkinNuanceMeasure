import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from pathlib import Path
from PIL import Image
PROJECT_ROOT = Path(__file__).parent

def main():
    df = pd.read_csv(PROJECT_ROOT / "skin/UTKFace/metadata.csv")
    plt.hist(df["age"], bins=10)
    plt.xlabel("Age")
    plt.ylabel("Count")
    plt.title("Age Distribution")
    plt.savefig(str(PROJECT_ROOT / "skin/UTKFace/age.png"))
    plt.close()

    df["age_group"] = None

    df.loc[df["age"].between(0, 12), "age_group"] = "0-12"
    df.loc[df["age"].between(13, 19), "age_group"] = "13-19"
    df.loc[df["age"].between(20, 59), "age_group"] = "20-59"
    df.loc[df["age"] >= 60, "age_group"] = "60+"

    df["label"] = -1
    df.loc[df["age_group"] == "0-12", "label"] = 0
    df.loc[df["age_group"] == "13-19", "label"] = 1
    df.loc[df["age_group"] == "20-59", "label"] = 2
    df.loc[df["age_group"] == "60+", "label"] = 3

    df["maskpath"] = df["filepath"].replace("/data/", "/mask/")

    train_df, temp_df = train_test_split(
        df,
        test_size=0.4,
        random_state=42,
        stratify=df["label"]
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=42,
        stratify=temp_df["label"]
    )

    train_df.to_csv(PROJECT_ROOT / "skin/UTKFace/df_train.csv")
    val_df.to_csv(PROJECT_ROOT / "skin/UTKFace/df_valid.csv")
    test_df.to_csv(PROJECT_ROOT / "skin/UTKFace/df_test.csv")


    src_dir = Path("/home/kuniko/dataset/Core/UTKFace/original/")
    dst_dir = Path("/home/kuniko/notebooks/SkinNuanceMeasure/skin/UTKFace/data")
    dst_dir.mkdir(parents=True, exist_ok=True)
    new_size = (224, 224)

    extensions = {".jpg"}

    for img_path in src_dir.iterdir():
        if img_path.suffix.lower() in extensions:
            try:
                with Image.open(img_path) as img:
                    img = img.convert("RGB")
                    resized = img.resize(new_size)

                    save_path = dst_dir / img_path.name
                    resized.save(save_path)

                    print(f"saved: {save_path}")
            except Exception as e:
                print(f"error: {img_path} -> {e}")

if __name__ == "__main__":
    main()