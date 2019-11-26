# Copyright (c) 2019
# Manuel Cherep <manuel.cherep@epfl.ch>

"""
This module is intended for processing the omohands dataset
to save the data in the format needed to train.
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import pandas as pd
import shutil
import random
import xml.etree.ElementTree as ET

DATA_PATH = '../data/omohands'  # Path for the final dataset
CLASS = 1  # There's only one class (i.e. hand)
TRAIN_PER = .8
VAL_PER = .2
TEST_PER = .0


def main():
    create_labels()
    clean()
    split_train_test()


def create_labels():
    """ This function saves the labels for all images in a .csv """

    labels = []

    # Select only the xml files
    xml_files = [f for f in os.listdir(DATA_PATH) if f.endswith('.xml')]

    for xml_f in xml_files:
        tree = ET.parse(os.path.join(DATA_PATH, xml_f))
        root = tree.getroot()

        for boxes in root.iter("object"):
            for box in boxes.findall("bndbox"):
                ymin = int(box.find("ymin").text)
                xmin = int(box.find("xmin").text)
                ymax = int(box.find("ymax").text)
                xmax = int(box.find("xmax").text)

                # Remove the l/r and .json from the filename
                filename = xml_f[:-4] + ".jpg"
                # Save as bounding box
                labels.append({"frame": filename,
                               "xmin": xmin,
                               "xmax": xmax,
                               "ymin": ymin,
                               "ymax": ymax,
                               "class_id": CLASS})

    # Pandas dataframe to save in .csv
    labels_df = pd.DataFrame(labels)
    labels_df.to_csv(os.path.join(DATA_PATH, 'labels.csv'), index=False)


def clean():
    """ This function relocates images and removes useless files """
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
