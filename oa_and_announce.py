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
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
              "boat", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant"]

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
            if distance < 0.4:  # Obstacle detected
                print("Obstacle detected! Stopping.")
                obstacle_detected = True
                misty.Stop()
                time.sleep(1)

                # Announce before avoidance
                announce("Obstacle detected. Analyzing the situation.")

                # Run YOLO detection
                frame = fetch_misty_camera_frame()
                if frame is not None:
                    print("Processing YOLO for detected obstacle...")
                    run_yolo_and_announce(frame)

                # Avoidance maneuver
                announce("Avoiding the obstacle now.")
                print("Avoiding obstacle...")
                misty.DriveTime(0, 50, 2000)  # Turn right
                time.sleep(2.5)
                misty.DriveTime(20, 0, 2000)  # Move forward
                time.sleep(2.5)
                misty.DriveTime(0, -50, 2000)  # Turn left
                time.sleep(2.5)

                print("Resuming forward motion.")
                announce("Obstacle cleared. Resuming movement.")
                obstacle_detected = False  # Reset flag

    except json.JSONDecodeError:
        print("Invalid JSON received:", message)
    except Exception as e:
        print("Error processing WebSocket message:", e)


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


def keep_moving_forward():
    """Continuously move forward unless interrupted."""
    while not stop_requested:
        if not obstacle_detected:
            misty.DriveTime(20, 0, 4000)
            for _ in range(40):  # Check every 0.1s for 4s
                if stop_requested or obstacle_detected:
                    break
                time.sleep(0.3)
        else:
            time.sleep(0.3)  # Allow obstacle handling
    misty.Stop()
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
