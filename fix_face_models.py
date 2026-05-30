"""Patch face_recognition_models to use os.path instead of pkg_resources."""
import os

path = os.path.expanduser(
    "~/AIRobot_v2.0/venv/lib/python3.12/site-packages/face_recognition_models/__init__.py"
)

code = '''import os

_models_dir = os.path.join(os.path.dirname(__file__), "models")


def pose_predictor_model_location():
    return os.path.join(_models_dir, "shape_predictor_68_face_landmarks.dat")


def pose_predictor_five_point_model_location():
    return os.path.join(_models_dir, "shape_predictor_5_face_landmarks.dat")


def face_recognition_model_location():
    return os.path.join(_models_dir, "dlib_face_recognition_resnet_model_v1.dat")


def cnn_face_detector_model_location():
    return os.path.join(_models_dir, "mmod_human_face_detector.dat")
'''

with open(path, "w") as f:
    f.write(code)

print("Patched face_recognition_models successfully")

# Verify
import face_recognition_models
print(f"pose_predictor: {face_recognition_models.pose_predictor_model_location()}")
print(f"exists: {os.path.exists(face_recognition_models.pose_predictor_model_location())}")
