import requests
import json
import time
import threading
import websocket
from mistyPy.Robot import Robot

MISTY_IP = "10.5.11.234"
ENCODER_CPR = 100
LINEAR_VELOCITY = 0.5
DRIVE_DURATION = 5

misty = Robot(MISTY_IP)
initial_encoders = None
final_encoders = None

def on_message(ws, message):
    global initial_encoders, final_encoders
    try:
        data = json.loads(message)
        if "message" in data:
            encoders = data["message"]
            if initial_encoders is None:
                initial_encoders = encoders
            final_encoders = encoders
    except Exception as e:
        print(f"Error parsing WebSocket message: {e}")

def on_error(ws, error):
    print(f"WebSocket error: {error}")

def on_close(ws, close_status_code, close_msg):
    print("WebSocket connection closed")

def subscribe_to_encoders():
    ws_url = f"ws://{MISTY_IP}/pubsub"
    ws = websocket.WebSocketApp(
        ws_url,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close
    )
    ws.on_open = lambda ws: ws.send(json.dumps({
        "Operation": "subscribe",
        "Type": "DriveEncoders",
        "DebounceMs": 100,
        "EventName": "DriveEncoderData"
    }))
    return ws

def drive_misty():
    misty.DriveTime(LINEAR_VELOCITY, 0, DRIVE_DURATION * 1000)

def calculate_wheel_circumference():
    global initial_encoders, final_encoders
    if initial_encoders and final_encoders:
        left_ticks = final_encoders['LeftEncoderTicks'] - initial_encoders['LeftEncoderTicks']
        right_ticks = final_encoders['RightEncoderTicks'] - initial_encoders['RightEncoderTicks']
        average_ticks = (left_ticks + right_ticks) / 2
        wheel_rotations = average_ticks / ENCODER_CPR
        distance_traveled = LINEAR_VELOCITY * DRIVE_DURATION
        wheel_circumference = distance_traveled / wheel_rotations
        print(f"Wheel Circumference: {wheel_circumference:.4f} meters")
    else:
        print("Encoder data not available")

if __name__ == "__main__":
    try:
        ws = subscribe_to_encoders()
        ws_thread = threading.Thread(target=ws.run_forever)
        ws_thread.start()
        time.sleep(2)  # Wait for WebSocket to initialize
        drive_misty()
        time.sleep(DRIVE_DURATION + 2)  # Wait for Misty to finish driving
        calculate_wheel_circumference()
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        ws.close()
        ws_thread.join()
