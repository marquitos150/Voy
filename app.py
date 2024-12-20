import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import argparse # custom arguments
import csv # data collection

import mediapipe as mp
import cv2 as cv
import numpy as np

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--width", help='Enter width as an integer', type=int, default=960)
    parser.add_argument("--height", help='Enter height as an integer', type=int, default=540)

    args = parser.parse_args()
    return args

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

    # Collect the classification labels
    with open('./navigation_labels.csv', encoding='utf-8-sig') as f:
        navigation_labels = csv.reader(f)
        navigation_labels_list = [row[0] for row in navigation_labels]

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
                mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS)
        cv.imshow('Handtracker', frame)
        cv.waitKey(1)

if __name__ == '__main__':
    main()