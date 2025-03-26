# -----------------------------------------------------------------------------
# Main function for the driver's fatigue level estimation algorithm with streaming
# -----------------------------------------------------------------------------
# Author: Daniel Oliveira
# https://github.com/danielsousaoliveira
# Modified for streaming by Grok
# -----------------------------------------------------------------------------

import cv2
from utils import *
from detection.face import *
from detection.pose import *
from state import *
import mediapipe as mp
import time
from flask import Flask, Response
import threading
import pygame  # For cross-platform sound
camera_index = 0
# Flask app initialization
app = Flask(__name__)

# Global variable to store the latest frame
frame = None
frame_lock = threading.Lock()

# Initialize pygame for sound
pygame.mixer.init()

# Beep sound function
def beep(seconds=12):
    """Play an annoying beep sound for the specified duration."""
    frequency = 100  # Hz (annoying high-pitched beep)
    duration = seconds * 1000  # Convert to milliseconds
    sample_rate = 44100
    bits = 16

    # Generate a square wave beep
    n_samples = int(sample_rate * (duration / 1000))
    max_amplitude = 2 ** (bits - 1) - 1
    buffer = np.zeros(n_samples, dtype=np.int16)  # Use NumPy array with int16 type
    for i in range(n_samples):
        t = float(i) / sample_rate
        value = max_amplitude * 0.5 * (1 if t * frequency % 1 < 0.5 else -1)
        buffer[i] = int(value)
    
    # Convert to bytearray for Pygame
    sound = pygame.mixer.Sound(buffer)
    sound.play()
    #time.sleep(seconds)  # Wait for the sound to finish
    
# -----------------------------------------------------------------------------
# Streaming Function
# -----------------------------------------------------------------------------
def generate_frames():
    global frame
    while True:
        success, img = cap.read()
        if not success:
            print("Failed to read frame from camera")
            time.sleep(1)  # Wait before retrying
            continue

        # Convert to RGB for Mediapipe
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(img_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                head_posture.get_head_pose(img, face_landmarks)
                face_detection.detect_facial_features(img, face_landmarks)

        # Update the global frame
        with frame_lock:
            frame = img.copy()

        # Remove cv2.imshow to reduce load
        # cv2.imshow("Driver Monitor", img)
        # if cv2.waitKey(1) == 27:
        #     break

def video_feed():
    global frame
    while True:
        with frame_lock:
            if frame is None:
                time.sleep(0.01)
                continue
            ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if not ret:
                continue
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.05)  # Small delay to prevent overwhelming the client

@app.route('/video_feed')
def video_feed_route():
    return Response(video_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')
# -----------------------------------------------------------------------------
# Main Function
# -----------------------------------------------------------------------------
def main():
    """ Main function to monitor the driver's state and detect signs of drowsiness.

    This function records video using the provided camera, analyzes the frames to estimate the head pose and facial landmarks, 
    and then assesses the driver's condition using a variety of facial indicators such eye openness, lip position, and head pose. 
    An alert is sent if the driver exhibits indicators of sleepiness. The processed frames are streamed via Flask.

    """

    # Thresholds defined for driver state evaluation
    marThresh = 0.7
    marThresh2 = 0.15
    headThresh = 6
    earThresh = 0.28
    blinkThresh = 10
    gazeThresh = 5

    cap = cv2.VideoCapture(camera_index)  # Adjust camera index as needed
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)  # Lower resolution
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    # cap.set(cv2.CAP_PROP_FPS, 15)  # Lower frame rate

    # cap = cv2.VideoCapture('/home/daniel/test-dataset/processed_data/training/001_glasses_sleepyCombination.avi')

    if not cap.isOpened():
        print("Error opening video stream or file")
        return

    faceMesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    captureFps = cap.get(cv2.CAP_PROP_FPS)

    driverState = DriverState(marThresh, marThresh2, headThresh, earThresh, blinkThresh, gazeThresh)
    headPose = HeadPose(faceMesh)
    faceDetector = FaceDetector(faceMesh, captureFps, marThresh, marThresh2, headThresh, earThresh, blinkThresh)

    startTime = time.time()
    drowsinessCounter = 0
    
    # Starting beep
    beep(0.5)
    time.sleep(1)
    beep(0.5)
    time.sleep(1)
    beep(3)

    while cap.isOpened():
        ret, frame_local = cap.read()

        if not ret:
            break
        
        # Process the frame
        frame_local, results = headPose.process_image(frame_local)
        frame_local = headPose.estimate_pose(frame_local, results, True)
        roll, pitch, yaw = headPose.calculate_angles()

        frame_local, sleepEyes, mar, gaze, yawning, baseR, baseP, baseY, baseG = faceDetector.evaluate_face(frame_local, results, roll, pitch, yaw, True)

        frame_local, state = driverState.eval_state(frame_local, sleepEyes, mar, roll, pitch, yaw, gaze, yawning, baseR, baseP, baseG)

        # Update drowsiness counter if the driver is drowsy
        if state == "Drowsy":
            drowsinessCounter += 1

        drowsinessTime = drowsinessCounter / captureFps
        drowsy = drowsinessTime / 60

        # Reset the drowsiness counter after 1 minute
        if time.time() - startTime >= 60:
            drowsinessCounter = 0
            startTime = time.time()
    
        # Update the global frame for streaming
        with frame_lock:
            global frame
            frame = frame_local.copy()

        # Local display (optional, can comment out if not needed)
        cv2.imshow('Driver State Monitoring', frame_local)
        print(f"drowsy stats {drowsy} threashold is 0.08")	
        
        # Alert if the driver is showing signs of drowsiness for more than the threshold
        if drowsy > 0.08:
            print("USER IS SHOWING SIGNALS OF DROWSINESS. SENDING ALERT")
            beep(seconds=1)  # Play annoying beep for 12 seconds
            drowsinessCounter = 0

        if cv2.waitKey(10) & 0xFF == 27:  # Exit on 'Esc'
            break

    cap.release()
    cv2.destroyAllWindows()

# -----------------------------------------------------------------------------
# Run the Application
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Start the main video processing in a separate thread
    processing_thread = threading.Thread(target=main)
    processing_thread.daemon = True
    processing_thread.start()

    # Start the Flask server
    app.run(host='0.0.0.0', port=5000, threaded=True)
