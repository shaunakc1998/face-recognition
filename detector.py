import argparse
import pickle
import numpy as np
from collections import Counter
from pathlib import Path
from sklearn.svm import SVC
from sklearn.decomposition import PCA as RandomizedPCA
from sklearn.pipeline import make_pipeline

import face_recognition
from PIL import Image, ImageDraw

DEFAULT_ENCODINGS_PATH = Path("output/encodings.pkl")
BOUNDING_BOX_COLOR = "blue"
TEXT_COLOR = "white"

# Create directories if they don't already exist
Path("training").mkdir(exist_ok=True)
Path("output").mkdir(exist_ok=True)
Path("validation").mkdir(exist_ok=True)
Path("testing").mkdir(exist_ok=True)

parser = argparse.ArgumentParser(description="Recognize faces in an image")
parser.add_argument("--train", action="store_true", help="Train on input data")
parser.add_argument(
    "--validate", action="store_true", help="Validate trained model"
)
parser.add_argument(
    "--test", action="store_true", help="Test the model with an unknown image"
)
parser.add_argument(
    "-m",
    action="store",
    default="hog",
    choices=["hog", "cnn"],
    help="Which model to use for training: hog (CPU), cnn (GPU)",
)
args = parser.parse_args()


def encode_known_faces(
    model: str = "hog", encodings_location: Path = DEFAULT_ENCODINGS_PATH
) -> None:
    """
    Loads images in the training directory and builds a dictionary of their
    names and encodings.
    """
    names = []
    encodings = []

    for filepath in Path("training").glob("*/*"):
        name = filepath.parent.name
        image = face_recognition.load_image_file(filepath)

        face_locations = face_recognition.face_locations(image, model=model)
        face_encodings = face_recognition.face_encodings(image, face_locations)

        for encoding in face_encodings:
            names.append(name)
            encodings.append(encoding)

    name_encodings = {"names": names, "encodings": encodings}
    with encodings_location.open(mode="wb") as f:
        pickle.dump(name_encodings, f)


def recognize_faces(
    image_location: str,
    model: str = "hog",
    encodings_location: Path = DEFAULT_ENCODINGS_PATH,
) -> None:
    """
    Given an unknown image, get the locations and encodings of any faces and
    compares them against the known encodings to find potential matches.
    """
    with encodings_location.open(mode="rb") as f:
        loaded_encodings = pickle.load(f)

    input_image = face_recognition.load_image_file(image_location)

    input_face_locations = face_recognition.face_locations(
        input_image, model=model
    )
    input_face_encodings = face_recognition.face_encodings(
        input_image, input_face_locations
    )

    pillow_image = Image.fromarray(input_image)
    draw = ImageDraw.Draw(pillow_image)

    name = ""
    for bounding_box, unknown_encoding in zip(
        input_face_locations, input_face_encodings
    ):
        # name = _recognize_face(unknown_encoding, loaded_encodings)
        # name = knn_recognize_face(unknown_encoding, loaded_encodings)
        # name = svm_recognize_face(unknown_encoding, loaded_encodings)
        name = pca_svm_recognize_face(unknown_encoding, loaded_encodings)
        if not name:
            name = "Unknown"
        _display_face(draw, bounding_box, name)
        break
    del draw
    pillow_image.show()
    return name

def face_distance(face_encodings, face_to_compare):
    if len(face_encodings) == 0:
        return np.empty((0))
    return np.linalg.norm(face_encodings - face_to_compare, axis=1)

def knn_recognize_face(unknown_encoding, loaded_encodings):
    distances = face_distance(loaded_encodings["encodings"], unknown_encoding)
    index = np.argmin(distances)
    return loaded_encodings["names"][index]

def compare_faces(known_face_encodings, face_encoding_to_check, tolerance=0.4):
    return list(face_distance(known_face_encodings, face_encoding_to_check) <= tolerance)

def svm_recognize_face(unknown_encoding, loaded_encodings):
    svc = SVC(kernel = 'rbf', class_weight = 'balanced')
    svc.fit(loaded_encodings["encodings"], loaded_encodings["names"])
    name = svc.predict([unknown_encoding])
    return name[0]

def pca_svm_recognize_face(unknown_encoding, loaded_encodings):
    pca = RandomizedPCA(n_components=128, whiten=True, random_state=42)
    svc = SVC(kernel = 'rbf', class_weight = 'balanced')
    model = make_pipeline(pca, svc)
    model.fit(np.array(loaded_encodings["encodings"]), np.array(loaded_encodings["names"]))
    name = model.predict([unknown_encoding])
    return name[0]

def _recognize_face(unknown_encoding, loaded_encodings):
    """
    Given an unknown encoding and all known encodings, find the known
    encoding with the most matches.
    """
    boolean_matches = compare_faces(
        loaded_encodings["encodings"], unknown_encoding, 0.4
    )
    votes = Counter(
        name
        for match, name in zip(boolean_matches, loaded_encodings["names"])
        if match
    )
    if votes:
        return votes.most_common(1)[0][0]


def _display_face(draw, bounding_box, name):
    """
    Draws bounding boxes around faces, a caption area, and text captions.
    """
    top, right, bottom, left = bounding_box
    draw.rectangle(((left, top), (right, bottom)), outline=BOUNDING_BOX_COLOR)
    text_left, text_top, text_right, text_bottom = draw.multiline_textbbox(
        (left, bottom), name
    )
    draw.rectangle(
        ((text_left, text_top), (text_right, text_bottom)),
        fill=BOUNDING_BOX_COLOR,
        outline=BOUNDING_BOX_COLOR,
    )
    draw.multiline_text(
        (text_left, text_top),
        name,
        fill=TEXT_COLOR,
    )


def validate_test(model, imagePath):
    """
    Runs recognize_faces on a set of images with known faces to validate
    known encodings.
    """
    totalNum = 0
    correctNum = 0
    for filepath in Path(imagePath).glob("*/*"):
        if filepath.is_file():
            totalNum += 1
            name = recognize_faces(
                image_location=str(filepath.absolute()), model=model
            )
            if (name == filepath.parent.name):
                correctNum += 1
            else:
                print(name)
    print("Accuracy:", correctNum / totalNum)

if __name__ == "__main__":
    if args.train:
        encode_known_faces(model=args.m)
    if args.validate:
        validate_test(args.m, "validation")
    if args.test:
        validate_test(args.m, "testing")
