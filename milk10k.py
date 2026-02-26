import pandas as pd
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).parent # Path to folder opened files
SOURCE_ROOT = '../../dataset/SKIN/milk10k/'
DEST_ROOT = PROJECT_ROOT / 'skin/milk10k/'

def resize_and_save_images(df: pd.DataFrame, size=(224, 224)):

    for _, r in df.iterrows():
        with Image.open(r['source_paths']) as img:
            img = img.convert('RGB')
            img = img.resize(size, Image.BILINEAR)

            save_path = Path(r['filepath'])
            save_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(save_path)

def split_dataset(df: pd.DataFrame, root: Path):

    df['stratify_key'] = df['label'].astype(str) + "_" + df['skin tone'].astype(str)
    train_df, tmp = train_test_split(df, train_size=0.60, random_state=42, stratify=df['stratify_key'])
    val_df, test_df = train_test_split(tmp, train_size=0.5, random_state=42, stratify=tmp['stratify_key'])

    print(val_df["stratify_key"].value_counts().sort_index())
    print(test_df["stratify_key"].value_counts().sort_index())

    train_df['filepath'] = train_df.apply(lambda row: root / str(row['label']) / f"{row['isic_id']}.jpg", axis=1)
    val_df['filepath'] = val_df.apply(lambda row: root / str(row['label']) / f"{row['isic_id']}.jpg", axis=1)
    test_df['filepath'] = test_df.apply(lambda row: root / str(row['label']) / f"{row['isic_id']}.jpg", axis=1)

    resize_and_save_images(train_df, size=(224, 224))
    resize_and_save_images(val_df, size=(224, 224))
    resize_and_save_images(test_df, size=(224, 224))

    train_df.to_csv(root / 'df_train.csv', index=False)
    val_df.to_csv(root / 'df_valid.csv', index=False)
    test_df.to_csv(root / 'df_test.csv', index=False)

    print('Completed.')

LABEL_MAP = {
    'AKIEC': 0, 'BCC': 1, 'BEN_OTH': 10, 'BKL': 2, 'DF': 10, 'INF': 10, 'MAL_OTH': 10, 'MEL': 3, 'NV': 4, 'SCCKA': 5, 'VASC': 10
}

def map_skin_tone(x):
    if x in [0, 1, 2]:
        return 0  # dark
    elif x == 3:
        return 1  # mid
    else:
        return 2  # light

# Only one for experiments
def main():

    meta_path = Path('../../dataset/SKIN/milk10k/MILK10k_Training_Metadata.csv')
    gt_path = Path('../../dataset/SKIN/milk10k/MILK10k_Training_GroundTruth.csv')
    image_folder_path = Path('../../dataset/SKIN/milk10k/MILK10k_Training_Input')

    df = pd.read_csv(meta_path)
    gt_df = pd.read_csv(gt_path)

    cols = list(LABEL_MAP.keys())
    gt_df['label'] = gt_df[cols].idxmax(axis=1).map(LABEL_MAP)

    df = df.merge(
        gt_df[['lesion_id', 'label']],
        on='lesion_id',
        how='left'
    )

    df['source_paths'] = df.apply(
        lambda row: image_folder_path / str(row["lesion_id"]) / f'{row["isic_id"]}.jpg',
        axis=1
    )

    df["skin tone"] = df["skin_tone_class"].apply(map_skin_tone)

    keep_labels = list(range(6))
    derm_df = df[(df['image_type'] == 'dermoscopic') & (df['label'].isin(keep_labels))].copy()
    clinical_df = df[(df['image_type'] == 'clinical: close-up') & (df['label'].isin(keep_labels))].copy()
    print('Removed "Category is Other".')

    split_dataset(derm_df, root=(DEST_ROOT / 'dermoscopic'))
    split_dataset(clinical_df, root=(DEST_ROOT / 'clinical'))

if __name__ == '__main__':
    main()
