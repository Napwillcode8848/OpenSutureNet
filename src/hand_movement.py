"""This file provides function that will be used for
trancking hand movements.
1. extract frames
2. segment gloves
3. point to track
- centroid
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np

# data
lower_blue = np.array([100, 80,  60])   # HSV
upper_blue = np.array([130, 220, 120])

# deal with only 1 video first
# -- on the main loop we can run it for many videos
def segment_gloves(video_path: str, glove_mask_dir: str) -> None:
    """Save segmented json file to a given directory path"""
    vidcap = cv2.VideoCapture(video_path)
    success,image = vidcap.read()
    count = 0
    while success:
        success,image = vidcap.read()
        # convert from BGR to HSV
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # select only the color defined
        mask   = cv2.inRange(image_hsv, lower_blue, upper_blue)
        result = cv2.bitwise_and(image_rgb, image_rgb, mask=mask)
        # save mask to a folder
        cv2.imwrite(f"../result/test/frame_{count}.png", mask)
        # extract centroid and contour from here
