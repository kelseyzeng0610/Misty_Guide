import time
import threading
import math
import requests
import base64
import numpy as np
import cv2
import websocket
import json
from gtts import gTTS
from mistyPy.Robot import Robot
from ultralytics import YOLO

# Misty Robot Setup
MISTY_IP = "10.5.9.252"  # Replace with your Misty's IP address
misty = Robot(MISTY_IP)

# Load YOLO model
model = YOLO("yolo_weights/yolo11n.pt")  


classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus",
              "train", "truck", "boat", "traffic light", "fire hydrant",
              "stop sign", "parking meter", "bench", "bird", "cat", "dog",
              "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
              "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
              "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket",
              "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
              "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
              "hot dog", "pizza", "donut", "cake", "chair", "sofa",
              "pottedplant", "bed", "diningtable", "toilet", "tvmonitor",
              "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book",
              "clock", "vase", "scissors", "teddy bear", "hair drier",
              "toothbrush"
              ]

# Global Flags
stop_requested = False
obstacle_detected = False  # Flag for obstacle avoidance
ws = None  # WebSocket connection


def wait_for_keypress():
    """Stop the robot when ENTER is pressed."""
    global stop_requested
    input("Press ENTER to stop the robot.\n")
    stop_requested = True


def fetch_misty_camera_frame():
    """Fetch the latest camera frame from Misty."""
    url = f"http://{MISTY_IP}/api/cameras/rgb"
    headers = {"Accept": "image/jpeg"}
    try:
        response = requests.get(url, headers=headers, stream=True, timeout=5)
        if response.status_code == 200:
            img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
            return cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    except Exception as e:
        print("Error fetching camera frame:", e)
    return None


def announce(text):
    """Generate and play audio using TTS."""
    try:
        print(f"Announcing: '{text}'")
        speech = gTTS(text=text, lang="en", slow=False)
        file_name = "announcement.mp3"
        speech.save(file_name)


        ENCODING = 'utf-8'
        encode_string = base64.b64encode(open(file_name, "rb").read())
        base64_string = encode_string.decode(ENCODING)

        save_audio_response = misty.SaveAudio(file_name, data=base64_string, overwriteExisting=True, immediatelyApply=True)
        print(save_audio_response)
        misty.PlayAudio(file_name, volume=10)
        time.sleep(2)  
    except Exception as e:
        print(f"Error during announcement: {e}")


def run_yolo_and_announce(frame):
    """Run YOLO detection and announce the detected object."""
    detected = False
    results = model(frame, stream=False)
    for r in results:
        for box in r.boxes:
            confidence = box.conf[0]
            if confidence > 0.5:
                obj_name = classNames[int(box.cls[0])]
                text = f"I see a {obj_name} in front of me."
                print(text)
                announce(text)
                detected = True
    if not detected:
        announce("Obstacle detected, but I cannot identify it.")
    return detected





def on_error(ws, error):
    print("WebSocket Error:", error)


def on_close(ws, close_status_code, close_msg):
    print("WebSocket connection closed")


def on_open(ws):
    """Subscribe to the TOF event on WebSocket."""
    print("WebSocket connection opened.")
    ws.send(json.dumps({
        "Operation": "subscribe",
        "Type": "TimeOfFlight",
        "DebounceMs": 100,
        "EventName": "TimeOfFlight",
        "Message": "TimeOfFlight"
    }))


def start_websocket():
    """Start the WebSocket connection."""
    global ws
    websocket_url = f"ws://{MISTY_IP}/pubsub"
    ws = websocket.WebSocketApp(websocket_url,
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close)
    ws.on_open = on_open
    ws.run_forever()



def smooth_stop():
    """Gradually slow down the robot."""
    for speed in range(10, 0, -5):  # Decrease speed in steps
        misty.Drive(speed, 0)
        time.sleep(0.2)
    misty.Stop()
    print("Robot smoothly stopped.")


def on_message(ws, message):
    """Handle incoming messages from Misty's WebSocket."""
    global obstacle_detected
    if stop_requested or obstacle_detected:
        return

    try:
        # Parse message
        data = json.loads(message) if isinstance(message, str) else message
        tof_message = data.get("message", {})
        sensor_position = tof_message.get("sensorPosition", "")
        distance = tof_message.get("distanceInMeters", None)

        if sensor_position == "Center" and distance is not None:
            print(f"Distance in meters (Center): {distance}")
            
            if 0.4 < distance < 0.6:  # Slow down if an obstacle is near
                print("Slowing down due to nearby obstacle.")
                misty.Drive(10, 0)  # Reduce speed
            
            elif distance <= 0.4:  # Stop if very close
                print("Obstacle detected! Preparing to stop.")
                obstacle_detected = True
                smooth_stop()

                # Announce before processing
                announce("Obstacle detected. Analyzing the situation.")

                # Run YOLO detection
                frame = fetch_misty_camera_frame()
                if frame is not None:
                    print("Processing YOLO for detected obstacle...")
                    run_yolo_and_announce(frame)

                # Avoidance maneuver
                announce("Avoiding the obstacle now.")
                print("Avoiding obstacle...")
                misty.DriveTime(0, 230, 3000)  # Turn right
                time.sleep(2.5)
                misty.DriveTime(20, 0, 1000)  # Move forward
                time.sleep(3)
                misty.DriveTime(0, -250, 3000)  # Turn left
                time.sleep(2.5)

                print("Resuming forward motion.")
                announce("Obstacle cleared. Resuming movement.")
                obstacle_detected = False  # Reset flag



                print("Debouncing...")
                # prevent immediate trigger
                time.sleep(2) 

    except json.JSONDecodeError:
        print("Invalid JSON received:", message)
    except Exception as e:
        print("Error processing WebSocket message:", e)


def keep_moving_forward():
    """Continuously move forward unless interrupted."""
    while not stop_requested:
        if not obstacle_detected:
            misty.Drive(20, 0)  # Move at full speed
            time.sleep(0.1)  # Shorter sleep for better control
        else:
            time.sleep(0.1)  # Allow obstacle handling
    smooth_stop()
    print("Robot stopped by user.")


def main():
    global stop_requested
    print("Starting robot... Press ENTER to stop.")
    threading.Thread(target=wait_for_keypress, daemon=True).start()

    # Start WebSocket for TOF sensor
    threading.Thread(target=start_websocket, daemon=True).start()

    # Start forward motion
    keep_moving_forward()


if __name__ == "__main__":
    main()


