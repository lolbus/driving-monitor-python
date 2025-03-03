'''import cv2
import requests
import numpy as np

url = "http://192.168.0.28:5000/video_feed"  # Replace with Raspberry Pi IP

while True:
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        bytes_data = bytes()
        for chunk in response.iter_content(chunk_size=1024):
            bytes_data += chunk
            a = bytes_data.find(b'\xff\xd8')  # JPEG start
            b = bytes_data.find(b'\xff\xd9')  # JPEG end
            if a != -1 and b != -1:
                jpg = bytes_data[a:b+2]
                bytes_data = bytes_data[b+2:]
                img = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                cv2.imshow('Video Feed', img)
                if cv2.waitKey(1) == 27:  # Exit on 'Esc'
                    break
    else:
        print("Failed to connect")
        break

cv2.destroyAllWindows()'''

import cv2

# Replace with your Raspberry Pi's IP address
url = "http://192.168.0.28:5000/video_feed"

# Open the stream
cap = cv2.VideoCapture(url)

if not cap.isOpened():
    print("Error: Could not open video stream")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to retrieve frame")
        break

    # Display the frame
    cv2.imshow('Video Feed', frame)

    # Exit on 'Esc' key
    if cv2.waitKey(1) == 27:
        break

# Clean up
cap.release()
cv2.destroyAllWindows()