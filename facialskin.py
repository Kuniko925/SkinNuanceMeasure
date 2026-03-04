import cv2
import dlib
import numpy as np
import pandas as pd
from imutils import face_utils
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent # Path to folder opened file

class FaceSkinMasker:
    def __init__(self, predictor_path="shape_predictor_68_face_landmarks.dat"):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)

    def detect_faces(self, bgr_img):
        gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
        return self.detector(gray, 1), gray

    def get_landmarks(self, gray, face):
        shape = self.predictor(gray, face)
        return face_utils.shape_to_np(shape)  # (68,2)

    @staticmethod
    def _parts_from_landmarks(lm):
        return {
            "jaw": lm[0:17],
            "right_eyebrow": lm[17:22],
            "left_eyebrow": lm[22:27],
            "nose": lm[27:36],
            "right_eye": lm[36:42],
            "left_eye": lm[42:48],
            "mouth": lm[48:68],
        }

    @staticmethod
    def _fill_poly(mask, pts, value=255):
        pts = np.array(pts, dtype=np.int32)
        cv2.fillPoly(mask, [pts], value)

    def build_mask(
        self,
        bgr_img,
        exclude=("right_eye", "left_eye", "mouth"),
        keep_face_hull=True,
        cut_above_eyes=True,
        exclude_mouth_inner=True,
    ):
        faces, gray = self.detect_faces(bgr_img)
        h, w = bgr_img.shape[:2]

        # multiple face
        final_keep = np.zeros((h, w), dtype=np.uint8)

        for face in faces:
            lm = self.get_landmarks(gray, face)
            parts = self._parts_from_landmarks(lm)

            # detect face
            keep = np.ones((h, w), dtype=np.uint8) * 255

            if keep_face_hull:
                hull_pts = np.concatenate(
                    (parts["right_eyebrow"], parts["left_eyebrow"], parts["jaw"])
                )
                hull = cv2.convexHull(hull_pts)
                keep_hull = np.zeros((h, w), dtype=np.uint8)
                cv2.fillConvexPoly(keep_hull, hull, 255)
                keep = cv2.bitwise_and(keep, keep_hull)

            # remove area above eyebrows
            if cut_above_eyes:
                eye_top = int(np.mean([parts["right_eye"][1][1], parts["left_eye"][1][1]]))
                keep[:eye_top, :] = 0

            # masking on non-skin area
            for name in exclude:
                if name in parts:
                    self._fill_poly(keep, parts[name], value=0)

            # making mouth area
            if "mouth" in parts:
                outer_mouth = parts["mouth"][:12]
                if "mouth" in exclude or exclude_mouth_inner:
                    self._fill_poly(keep, outer_mouth, value=0)

            final_keep = cv2.bitwise_or(final_keep, keep)

        return final_keep, faces

    def apply_mask(self, bgr_img, keep_mask):
        out = bgr_img.copy()
        out[keep_mask == 0] = 0
        return out

    def landmark_debug_image(self, bgr_img):
        faces, gray = self.detect_faces(bgr_img)
        out = bgr_img.copy()
        for face in faces:
            lm = self.get_landmarks(gray, face)
            for (x, y) in lm:
                cv2.circle(out, (int(x), int(y)), 1, (0, 0, 255), -1)
        return out

def is_frontal_face(filepath, predictor_path="shape_predictor_68_face_landmarks.dat"):
    img = cv2.imread(filepath)
    if img is None:
        return False
    masker = FaceSkinMasker(predictor_path)
    faces, _ = masker.detect_faces(img)
    return len(faces) > 0

def mask_image(filepath, predictor_path="shape_predictor_68_face_landmarks.dat"):
    img = cv2.imread(filepath)
    masker = FaceSkinMasker(predictor_path)
    keep, _ = masker.build_mask(img, exclude=("right_eye","left_eye","mouth"), cut_above_eyes=True)
    masked = masker.apply_mask(img, keep)
    return masked

def check_non_frontal_face_image(dataset, filepath_column_name):
    non_frontal_index = []

    for i, row in dataset.iterrows():
        if not is_frontal_face(row[filepath_column_name]):
            non_frontal_index.append(i)
            print(f"Non frontal image: {row[filepath_column_name]}")

    print(f"Number of non frontal image: {len(non_frontal_index)}")
    return non_frontal_index

def create_masked_image(dataset, save_directory, filepath_column_name, mask_column_name):
    blacks = []

    for i, d in dataset.iterrows():
        maskpath = d[mask_column_name]
        masked_image = mask_image(d[filepath_column_name])

        # Remove all black photos after mask
        is_all_black = np.all(masked_image == [0, 0, 0])
        if is_all_black:
            blacks.append(i)
            print(f"All black: {d[mask_column_name]}")
        else:
            cv2.imwrite(d[mask_column_name], masked_image)

    print(f"Masked image was created: {len(dataset)}")
    print(f"All black images: {len(blacks)}")
    return blacks

def main():

    filepaths = [PROJECT_ROOT / "skin/UTKFace/df_train.csv", PROJECT_ROOT / 'skin/UTKFace/df_valid.csv', PROJECT_ROOT / 'skin/UTKFace/df_test.csv']
    for filepath in filepaths:
        df = pd.read_csv(filepath)
        non_frontal_index = check_non_frontal_face_image(df, "filepath")
        df.drop(index=non_frontal_index, inplace=True)

        save_directory = PROJECT_ROOT / f"skin/UTKFace/mask/"
        all_blacks_index = create_masked_image(df, save_directory, "filepath", "maskpath")
        df.drop(index=all_blacks_index, inplace=True)



if __name__== "__main__":
    main()