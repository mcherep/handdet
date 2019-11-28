# Copyright (c) 2019
# Manuel Cherep <manuel.cherep@epfl.ch>

"""
Logic to create your own dataset
"""

from imutils.video import FPS, WebcamVideoStream
import cv2
import os

VIDEO_SOURCE = 0  # Index for the webcam
DATA_DIR = '../data/omohands'


def main():
    # Video capture constructor multithreaded
    stream = WebcamVideoStream(VIDEO_SOURCE).start()

    print("Tracking...")
    recording = False
    tracking = True
    frame_id = 1000000
    while(tracking):
        # Capture frame
        frame = stream.read()
        frame = cv2.flip(frame, 1)

        if recording:
            if frame_id % 30 == 0:
                # Save image
                filename = os.path.join(DATA_DIR, str(frame_id) + '.jpg')
                cv2.imwrite(filename, frame)

            frame_id += 1
            if frame_id % 1000 == 0:

        cv2.imshow('frame', frame)

        # Waits for pressed key to stop execution
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            tracking = False
            cv2.destroyAllWindows()
            stream.stop()
        elif key == ord('r'):
            print("Recording...")
            recording = True


if __name__ == "__main__":
    main()
