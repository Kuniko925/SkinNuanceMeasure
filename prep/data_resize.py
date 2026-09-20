from pathlib import Path
from PIL import Image

src_dir = Path("skin/BCN/org")
dst_dir = Path("skin/BCN/data")

dst_dir.mkdir(parents=True, exist_ok=True)

extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

n_success = 0
n_failed = 0

for img_path in src_dir.rglob("*"):
    if img_path.suffix.lower() not in extensions:
        continue

    relative_path = img_path.relative_to(src_dir)
    save_path = dst_dir / relative_path
    save_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            img = img.resize((224, 224), Image.Resampling.BILINEAR)

            if save_path.suffix.lower() in {".jpg", ".jpeg"}:
                img.save(save_path, quality=95)
            else:
                img.save(save_path)

        n_success += 1

    except Exception as e:
        print(f"Failed: {img_path}")
        print(e)
        n_failed += 1

print(f"Completed.")
print(f"Success: {n_success}")
print(f"Failed : {n_failed}")