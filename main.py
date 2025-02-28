# -----------------------------------------------------------------------------
# Main function for the driver's fatigue level estimation algorithm
# -----------------------------------------------------------------------------
# Author: Daniel Oliveira
# https://github.com/danielsousaoliveira
# -----------------------------------------------------------------------------

import cv2
from utils import *
from detection.face import *
from detection.pose import *
from state import *
import mediapipe as mp
import time

# -----------------------------------------------------------------------------
# Main Function
# -----------------------------------------------------------------------------

def main():

    """ Main function to monitor the driver's state and detect signs of drowsiness.

    This function records video using the provided camera, analyzes the frames to estimate the head pose and facial landmarks, 
    and then assesses the driver's condition using a variety of facial indicators such eye openness, lip position, and head pose. 
    An alert is sent if the driver exhibits indicators of sleepiness.

    """

    # Thresholds defined for driver state evaluation
    marThresh = 0.7
    marThresh2 = 0.15
    headThresh = 6
    earThresh = 0.28
    blinkThresh = 10
    gazeThresh = 5

    cap = cv2.VideoCapture(4)
    # cap = cv2.VideoCapture('/home/daniel/test-dataset/processed_data/training/001_glasses_sleepyCombination.avi')

    if (cap.isOpened()== False): 
        print("Error opening video stream or file")

    faceMesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    captureFps = cap.get(cv2.CAP_PROP_FPS)

    driverState = DriverState(marThresh, marThresh2, headThresh, earThresh, blinkThresh, gazeThresh)
    headPose = HeadPose(faceMesh)
    faceDetector = FaceDetector(faceMesh, captureFps, marThresh, marThresh2, headThresh, earThresh, blinkThresh)


    startTime = time.time()
    drowsinessCounter = 0

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break
        
        frame, results = headPose.process_image(frame)
        frame = headPose.estimate_pose(frame, results, True)
        roll, pitch, yaw = headPose.calculate_angles()

        frame, sleepEyes, mar, gaze, yawning, baseR, baseP, baseY, baseG = faceDetector.evaluate_face(frame, results, roll, pitch, yaw, True)

        frame, state = driverState.eval_state(frame, sleepEyes, mar, roll, pitch, yaw, gaze, yawning, baseR, baseP, baseG)

        # Update drowsiness counter if the driver is drowsy
        if state == "Drowsy":
            drowsinessCounter += 1

        drowsinessTime = drowsinessCounter / captureFps
        drowsy = drowsinessTime / 60

        # Reset the drowsiness counter after 1 minute (can be updated)
        if time.time() - startTime >= 60:
            drowsinessCounter = 0
            startTime = time.time()
    
        cv2.imshow('Driver State Monitoring', frame)

        # Alert if the driver is showing signs of drowsiness for more than the threshold
        if drowsy > 0.08:

            # This will be sent to the driver's MiBand so that he gets a vibrating alert
            print("USER IS SHOWING SIGNALS OF DROWSINESS. SENDING ALERT")

        if cv2.waitKey(10) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()









