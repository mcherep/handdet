# Copyright (c) 2019
# Manuel Cherep <manuel.cherep@epfl.ch>

"""
This module is intended for processing the hand db dataset
from CMU to save the data in the format needed to train.
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import pandas as pd
import shutil
import random
import json

DATA_PATH = '../data/handdb'  # Path for the final dataset
CLASS = 1  # There's only one class (i.e. hand)
TRAIN_PER = .7
VAL_PER = .1
TEST_PER = .2


def main():
    create_labels()
    clean()
    split_train_test()


def create_labels():
    """ This function saves the labels for all images in a .csv """

    labels = []
    data_dirs = [os.path.join(DATA_PATH, 'manual_train'),
                 os.path.join(DATA_PATH, 'manual_test')]

    # Iterate over all dirs in the original dataset
    for data_dir in data_dirs:
        # Select only the json files
        json_files = [f for f in os.listdir(data_dir) if f.endswith('.json')]

        for json_f in json_files:
            with open(os.path.join(data_dir, json_f)) as json_file:
                data = json.load(json_file)

            # Use only valid points (i.e. 3rd argument is 1)
            hand_pts = np.array(data['hand_pts'])
            hand_pts = hand_pts[hand_pts[:, 2] == 1]

            [xmin, ymin, _] = np.min(hand_pts, axis=0)
            [xmax, ymax, _] = np.max(hand_pts, axis=0)

            width = xmax-xmin
            height = ymax-ymin

            # Resize boxes because points are inside the hand
            # and the box needs to be surrounding
            xmin = xmin-width*0.1
            ymin = ymin-height*0.1
            xmax = xmax+width*0.1
            ymax = ymax+height*0.1

            # Remove the l/r and .json from the filename
            filename = json_f[:-7] + ".jpg"
            # Save as bounding box
            labels.append({"frame": filename,
                           "xmin": int(np.floor(xmin)),
                           "xmax": int(np.ceil(xmax)),
                           "ymin": int(np.floor(ymin)),
                           "ymax": int(np.ceil(ymax)),
                           "class_id": CLASS})

    # Pandas dataframe to save in .csv
    labels_df = pd.DataFrame(labels)
    labels_df.to_csv(os.path.join(DATA_PATH, 'labels.csv'), index=False)


def clean():
    """ This function relocates images and removes useless files """

    data_dirs = [os.path.join(DATA_PATH, 'manual_train'),
                 os.path.join(DATA_PATH, 'manual_test')]

    # Move all images to DATA_PATH
    for data_dir in data_dirs:
        for f in os.listdir(data_dir):
            if ".jpg" in f:
                # Move only .jpg
                os.rename(os.path.join(data_dir, f),
                          os.path.join(DATA_PATH, f[:-6] + ".jpg"))

    # Remove unnecessary files (i.e. all except .jpg or .csv)
    useless_files = list(
        filter(lambda f: (".jpg" not in f) and (".csv" not in f), os.listdir(DATA_PATH)))
    # List is only here to avoid lazy evaluation
    list(map(lambda f: remove_file_or_directory(
        os.path.join(DATA_PATH, f)), useless_files))


def split_train_test():
    # Create directories for the data
    os.makedirs(os.path.join(DATA_PATH, 'train'))
    os.makedirs(os.path.join(DATA_PATH, 'validation'))
    os.makedirs(os.path.join(DATA_PATH, 'test'))

    # Read .csv with labels
    labels_df = pd.read_csv(os.path.join(DATA_PATH, 'labels.csv'))

    images = [f for f in os.listdir(DATA_PATH) if '.jpg' in f]
    # Shuffle dataframe rows before splitting
    random.seed(1)
    random.shuffle(images)
    # Divide in train and validation. The rest is for testing
    train_idxs = int(TRAIN_PER*len(images))
    val_idxs = int((TRAIN_PER+VAL_PER)*len(images))
    train_imgs = images[:train_idxs]
    val_imgs = images[train_idxs:val_idxs]
    test_imgs = images[val_idxs:]

    train_df = labels_df[labels_df['frame'].isin(train_imgs)]
    val_df = labels_df[labels_df['frame'].isin(val_imgs)]
    test_df = labels_df[labels_df['frame'].isin(test_imgs)]

    # Save labels in different directories
    train_df.to_csv(os.path.join(DATA_PATH, 'train/labels.csv'), index=False)
    val_df.to_csv(os.path.join(
        DATA_PATH, 'validation/labels.csv'), index=False)
    test_df.to_csv(os.path.join(DATA_PATH, 'test/labels.csv'), index=False)
    # Remove old labels.csv
    os.remove(os.path.join(DATA_PATH, 'labels.csv'))

    # Move all images to corresponding directory
    for f in os.listdir(DATA_PATH):
        if '.jpg' in f:
            if f in train_imgs:
                os.rename(os.path.join(DATA_PATH, f),
                          os.path.join(DATA_PATH, 'train', f))
            elif f in val_imgs:
                os.rename(os.path.join(DATA_PATH, f),
                          os.path.join(DATA_PATH, 'validation', f))
            elif f in test_imgs:
                os.rename(os.path.join(DATA_PATH, f),
                          os.path.join(DATA_PATH, 'test', f))


def plot_bbox(img_filename):
    """ This function plots an image with the corresponding boxes """

    # Read the image
    img = cv2.imread(os.path.join(DATA_PATH, 'test', img_filename))
    img = img[:, :, ::-1]  # RGB -> BGR

    # Read .csv with labels
    labels_df = pd.read_csv(os.path.join(DATA_PATH, 'test', 'labels.csv'))
    boxes_df = labels_df[labels_df.frame == img_filename]

    plt.figure(figsize=(20, 12))
    current_axis = plt.gca()

    for idx, row in boxes_df.iterrows():
        xmin = row.xmin
        xmax = row.xmax
        ymin = row.ymin
        ymax = row.ymax

        # Add rectangle to the plot
        current_axis.add_patch(plt.Rectangle(
            (xmin, ymin), xmax-xmin, ymax-ymin, color='g', fill=False, linewidth=2))

    plt.imshow(img)
    return plt.show()


def remove_file_or_directory(f):
    if os.path.isfile(f):
        os.remove(f)
    if os.path.exists(f):
        shutil.rmtree(f)


if __name__ == "__main__":
    main()
