import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import argparse # custom arguments
import csv # data collection
import uuid

# Classifiers
from models.locomotion_classifier.locomotion_classifier import LocomotionClassifier

import mediapipe as mp
import cv2 as cv
import numpy as np

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--video", help="Enter path to video file to process", type=str)
    parser.add_argument("--width", help='Enter width as an integer', type=int, default=960)
    parser.add_argument("--height", help='Enter height as an integer', type=int, default=540)

    args = parser.parse_args()
    return args

def main():
    # Grab arguments for parsing
    args = get_args()
    
    # Other important variables such as for debugging
    recording_on = False
    curr_class_id = None  # To store the label for the current gesture being recorded
    video_uuid = None
    frame_count = 0

    cap = None
    if args.video:
        # Capture the video and get the class id for the gesture
        cap = cv.VideoCapture(f"video_dataset\{args.video}")
        if not cap.isOpened():
            print(f"Error: Unable to open video file {args.video}")
            return
        
        # Capture gesture from args.video string ending at '_'
        gesture = args.video.split("_")[0]
        if gesture == "forward":
            curr_class_id = 0
        elif gesture == "backward":
            curr_class_id = 1
        elif gesture == "left":
            curr_class_id = 2
        elif gesture == "right":
            curr_class_id = 3
        elif gesture == "up":
            curr_class_id = 4
        elif gesture == "down":
            curr_class_id = 5
        else:
            print("Unknown gesture")
            return

        print(f"Processing video: {args.video}")
        video_uuid = str(uuid.uuid4)
    else:
        # Prepare the camera
        cap = cv.VideoCapture(0)
        if not cap.isOpened():
            print("Something went wrong with the camera")
            return
        
        cap.set(cv.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, args.height)

    # Setup the mediapipe for hand detection
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode = False, # needed for processing a continuous stream of images
        max_num_hands = 1,
    )

    locomotion_classifier = LocomotionClassifier()

    # Collect the classification labels
    with open('models/locomotion_classifier/locomotion_labels.csv', encoding='utf-8-sig') as f:
        locomotion_labels = csv.reader(f)
        locomotion_labels_list = [row[0] for row in locomotion_labels]

    # Continuously run demo for gesture recognition system
    while cap.isOpened():
        # Exit case
        key = cv.waitKey(10)
        if key == 27:  # ESC or external video finishes processing
            break
        
        # Toggle recording with 'r'
        if key == ord('r') and not args.video:
            recording_on = not recording_on
            if recording_on:
                curr_class_id = input("To begin recording, please enter the class ID # for the gesture (0: Forward, 1: Backward, 2: Left, 3: Right, 4: Up, 5: Down): ")
                print("Recording has begun. Press 'r' again to stop.")
                video_uuid = str(uuid.uuid4)
            else:
                print("Recording stopped")
                frame_count = 0

        # Capture frame
        ret, frame = cap.read()
        if not ret:
            break
        if not args.video:
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
                # Display result in frame
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

                if recording_on or args.video:
                    # Process video frame for collecting data
                    process_video_frame(cap, curr_class_id, video_uuid, frame_count, hand_landmarks)
                    frame_count += 1

        cv.imshow('Handtracker', frame)
        cv.waitKey(1)

    cap.release()
    cv.destroyAllWindows()
    print("Done")

def process_video_frame(cap, curr_class_id, video_uuid, frame_count, hand_landmarks):
    # Collect frame data
    frame_data = [
        curr_class_id,
        video_uuid,
        frame_count + 1,
        cap.get(cv.CAP_PROP_POS_MSEC)
    ]

    # append hand landmark coordinates to frame data list
    for landmark in hand_landmarks.landmark:
        frame_data.extend([landmark.x, landmark.y, landmark.z])

    # Write data to the CSV file
    with open('models/locomotion_classifier/locomotion_dataset.csv', mode='a', newline="") as f:
        writer = csv.writer(f)
        writer.writerow(frame_data)
        print(frame_data[2])

if __name__ == '__main__':
    main()