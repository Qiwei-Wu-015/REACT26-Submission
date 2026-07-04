import numpy as np

from .FaceVerseModel import FaceVerseModel


def get_faceverse(**kargs):
    faceverse_dict = np.load(
        "external/FaceVerse/data/faceverse_simple_v2.npy", allow_pickle=True
    ).item()
    faceverse_model = FaceVerseModel(faceverse_dict, **kargs)
    return faceverse_model, faceverse_dict
