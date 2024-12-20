import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import argparse # custom arguments
import csv # data collection
import torch

# Classifiers
from models.locomotion_classifier.locomotion_classifier import LocomotionClassifier
from models.rotation_classifier.rotation_classifier import RotationClassifier
from models.speed_classifier.speed_classifier import SpeedClassifier

import mediapipe as mp
import cv2 as cv
import numpy as np

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--width", help='Enter width as an integer', type=int, default=960)
    parser.add_argument("--height", help='Enter height as an integer', type=int, default=540)

    args = parser.parse_args()
    return args

def extract_landmarks():
    return ""

def main():
    # Grab arguments for parsing
    args = get_args()
    
    # Prepare the camera
    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, args.height)

    # Setup the mediapipe for hand detection
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode = False, # needed for processing a continuous stream of images
        max_num_hands = 2, # one hand for movement; another hand for rotation
                           # or it can be two hands for movement and rotation 
    )

    locomotion_classifier = LocomotionClassifier()
    rotation_classifier = RotationClassifier()
    speed_classifier = SpeedClassifier()

    # Collect the classification labels
    with open('models/locomotion_classifier/locomotion_labels.csv', encoding='utf-8-sig') as f:
        locomotion_labels = csv.reader(f)
        locomotion_labels_list = [row[0] for row in locomotion_labels]

    with open('models/rotation_classifier/rotation_labels.csv', encoding='utf-8-sig') as f:
        rotation_labels = csv.reader(f)
        rotation_labels_list = [row[0] for row in rotation_labels]

    with open('models/speed_classifier/speed_labels.csv', encoding='utf-8-sig') as f:
        speed_labels = csv.reader(f)
        speed_labels_list = [row[0] for row in speed_labels]

    # Continuously run gesture recognition system
    while True:
        # Exit case
        key = cv.waitKey(10)
        if key == 27:  # ESC
            break

        # Camera capture of a frame
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv.flip(frame, 1) # Flip or mirror the display
        
        # Mark the image as not writeable for improved performance and process the RGB image to detect hands
        frame.flags.writeable = False
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = hands.process(frame)

        # After processing, draw landmarks on the frame, requiring it to be modifiable
        frame.flags.writeable = True
        frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)

        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # Extract the hand landmarks

                # Preprocess the landmarks for the Pytorch model

                # Run the pytorch model to classify the gesture

                # Display result in frame
                mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS)
        cv.imshow('Handtracker', frame)
        cv.waitKey(1)

if __name__ == '__main__':
    main()