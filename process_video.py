#!/usr/bin/env python
"""
process_video.py:
This python module contains useful function for Deep Learning video processing using Tensorflow.
Tensorflow uses CPU and GPU (for accelerated computing). However, a GPU is recommended, preferably Nvidia with CUDA.

"""

# Deep Learning frameworks
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras

import cv2
import pandas as pd
import numpy as np
from os.path import join as pathjoin
import argparse

#### Globals
# Only use for argparser if arg(s) is(are) not provided
__DATASET_PATH = pathjoin("/", "dataset", "action-recognition-dataset")
__IMG_SIZE = 224
__MAX_LENGTH = 20
__NUM_FEATURES = 2048

############


def _crop_center_square(frame: np.array):
    """
    Crop part of the frame to reduce the size if the frame is not square, otherwise return the frame unchanged
    :param frame: A numpy array representing the features of this frame
    :return: A new numpy array after cutting to match the global variable size
    """
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y:start_y + min_dim, start_x:start_x + min_dim]


def _load_video(video_name: str) -> np.array:
    """
    Opens the video located at the 'args.dataset_path' variable and resizes it to 224x224 pixels.
    In addition, you can limit the maximum frames by setting the 'max_frames' variable.
    :param video_name: A string value representing the path of the video
    : max_frames: A int value representing the limit of maximum number of frames in the video. Only effective if
                        the num of frames in the original video is greater than 'max_frames'. Default is 0,
                        which does not limit the frames in the video.
    :return: A 2D numpy array containing a series of numpy arrays representing the frames.
    """
    cap = cv2.VideoCapture(pathjoin(args.dataset_path, video_name))
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = _crop_center_square(frame)
            frame = cv2.resize(frame, (args.image_size, args.image_size))
            frame = frame[:, :, [2, 1, 0]]
            frames.append(frame)
            if len(frames) == args.max_length:
                break
    finally:
        cap.release()  # closes the file descriptor for the video
    return np.array(frames) / 255.0  # divide between 255 since the pixels have a value between 0 and 255


def build_feature_extractor():
    """
    Instantiate an InceptionV3 model using keras trained on the ImageNet dataset to extract features from frames
    """
    feature_extractor = keras.applications.InceptionV3(weights="imagenet",
                                                       include_top=False, pooling="avg",
                                                       input_shape=(args.image_size, args.image_size, 3))
    preprocess_input = keras.applications.inception_v3.preprocess_input
    inputs = keras.Input((args.image_size, args.image_size, 3))
    preprocessed = preprocess_input(inputs)
    outputs = feature_extractor(preprocessed)
    return keras.Model(inputs, outputs, name="feature_extractor")


def prepare_all_videos(df, feature_extractor):
    num_samples = len(df)
    video_paths = df["video_name"].values.tolist()
    tags = df["class"].values.tolist()
    labels = df["class"].values
    encoded_labels = keras.layers.experimental.preprocessing.StringLookup(
        num_oov_indices=0, vocabulary=np.unique(labels))

    # `frame_masks` and `frame_features` are what we will feed to our sequence model.
    # `frame_masks` will contain a bunch of booleans denoting if a timestep is
    # masked with padding or not.
    frame_masks = np.zeros(shape=(num_samples, args.max_length), dtype="bool")
    frame_features = np.zeros(shape=(num_samples, args.max_length, args.num_features),
                              dtype="float32")

    # For each video.
    for idx, (tag, video_path) in enumerate(zip(tags, video_paths)):
        # Gather all its frames and add a batch dimension.
        frames = _load_video(pathjoin(args.dataset_path, tag, video_path))
        frames = frames[None, ...]
        # Initialize placeholders to store the masks and features of the current video.
        temp_frame_mask = np.zeros(shape=(1, args.max_length,), dtype="bool")
        temp_frame_featutes = np.zeros(shape=(1, args.max_length, args.num_features),
                                       dtype="float32")
        # Extract features from the frames of the current video.
        for i, batch in enumerate(frames):
            video_length = batch.shape[0]
            length = min(args.max_length, video_length)
            for j in range(length):
                temp_frame_featutes[i, j, :] = feature_extractor.predict(batch[None, j, :])
            temp_frame_mask[i, :length] = 1  # 1 = not masked, 0 = masked

        frame_features[idx,] = temp_frame_featutes.squeeze()
        frame_masks[idx,] = temp_frame_mask.squeeze()

    return (frame_features, frame_masks), labels


def main():
    print(_load_video("MilitaryParade/v_MilitaryParade_g25_c07.avi"))
    dataset = pd.read_csv(pathjoin(args.dataset_path, "dataset_info.csv"))
    feature_extractor = build_feature_extractor()
    prepare_all_videos(dataset, feature_extractor)
    return


if __name__ == "__main__":
    global args
    parser = argparse.ArgumentParser(
        prog="process_video",
        description="Process videos for action detection by extracting using a pre-trained feature extractor model",
        epilog="Created by Kevin B, Iot Security Laboratory, UTSA"
    )
    parser.add_argument('-d', '--dataset_path', metavar='DATASET_PATH', type=str, default=__DATASET_PATH)
    parser.add_argument('--image_size', metavar='IMAGE_SIZE', type=int, default=__IMG_SIZE)
    parser.add_argument('--num_features', metavar='FEATURES', type=int, default=__NUM_FEATURES)
    parser.add_argument('--max_length', metavar='MAX_LENGTH', type=int, default=__MAX_LENGTH)
    args = parser.parse_args()
    main()
