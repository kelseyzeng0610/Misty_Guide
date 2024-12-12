# import os
# import time
# from mistyPy.Robot import Robot
# from pynput import keyboard
# import base64
# import cv2
# import numpy as np
# import requests
# # Set up the robot's IP address
# ROBOT_IP = "10.5.11.234"
# misty = Robot(ROBOT_IP)

# # Directory to save pictures
# PICTURE_DIR = "ceiling_images"
# os.makedirs(PICTURE_DIR, exist_ok=True)

# def capture_image(filename):
#     """
#     Captures an image directly from Misty's RGB camera feed and saves it locally.

#     Parameters:
#     - filename (str): The local path where the image will be saved.
#     """
#     url = f"http://{ROBOT_IP}/api/cameras/rgb"
#     headers = {"Accept": "image/jpeg"}
    
#     try:
#         response = requests.get(url, headers=headers, stream=True, timeout=10)
#         if response.status_code == 200:
#             # Read the image data from the response
#             img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
#             frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
#             if frame is not None:
#                 # Save the image locally
#                 cv2.imwrite(filename, frame)
#                 print(f"Saved image: {filename}")
#             else:
#                 print("Failed to decode image.")
#                 return False
#         else:
#             print(f"Error fetching frame: {response.status_code} - {response.text}")
#             return False
#     except requests.exceptions.RequestException as e:
#         print(f"Error fetching image: {e}")
#         return False
    
#     return True

# # Function to move Misty forward for 1 meter
# def move_forward_1m():
#     # Misty's linear velocity is in mm/s
#     speed_mm_s = 250  # Adjust speed as needed
#     distance_mm = 650  # 1 meter = 1000 mm
#     duration_s = distance_mm / speed_mm_s
#     misty.DriveTime(speed_mm_s, 0, int(duration_s * 1000))
#     time.sleep(duration_s + 1)  # Wait for the movement to complete

# # Function to rotate Misty left by 90 degrees
# def rotate_left_90():
#     # Angular velocity is in degrees per second
#     angular_speed = 30  # degrees per second
#     angle = 90  # degrees
#     duration_s = angle / angular_speed
#     misty.DriveTime(0, angular_speed, int(duration_s * 1000))
#     time.sleep(duration_s + 1)

# # Function to rotate Misty right by 90 degrees
# def rotate_right_90():
#     # Angular velocity is in degrees per second
#     angular_speed = -30  # degrees per second (negative to rotate right)
#     angle = 90  # degrees
#     duration_s = abs(angle / angular_speed)
#     misty.DriveTime(0,angular_speed,int(duration_s * 1000))
#     time.sleep(duration_s + 1)

# # Function to handle the Tab key press
# def on_press(key):
#     if key == keyboard.Key.tab:
#         print("Tab key pressed. Starting sequence...")
#         move_forward_1m()
#         timestamp = int(time.time())
#         forward_img = os.path.join(PICTURE_DIR, f"forward_{timestamp}.jpg")
#         capture_image(forward_img)
#         rotate_left_90()
#         left_img = os.path.join(PICTURE_DIR, f"left_{timestamp}.jpg")
#         capture_image(left_img)
#         rotate_right_90()
#         print("Sequence completed.")
#     elif key == keyboard.Key.esc:
#         # Stop listener
#         print("Escape key pressed. Exiting.")
#         return False

# # Start listening for key presses
# with keyboard.Listener(on_press=on_press) as listener:
#     print("Listening for Tab key press. Press ESC to exit.")
#     listener.join()



import os
import time
from mistyPy.Robot import Robot
from pynput import keyboard
import base64
import cv2
import numpy as np
import requests

# Configuration Constants
ROBOT_IP = "10.5.11.234"  # Replace with your Misty's actual IP address
PICTURE_DIR = "ceiling_images"
ORIGIN = np.array([-15.4, -12.2, 0.0])  # Real-world origin at the door
RESOLUTION = 0.05  # meters per pixel
CEILING_GRID_SIZE = 6  # Grid cells per ceiling map cell
wheel_base = 0.11  # Wheel base (track width) in meters
LOOP_INTERVAL = 0.5  # Time interval in seconds for sensor updates

# Ensure the picture directory exists
os.makedirs(PICTURE_DIR, exist_ok=True)

# Initialize the robot
misty = Robot(ROBOT_IP)

def capture_image(filename):
    """
    Captures an image directly from Misty's RGB camera feed and saves it locally.

    Parameters:
    - filename (str): The local path where the image will be saved.
    
    Returns:
    - bool: True if the image was successfully captured and saved, False otherwise.
    """
    url = f"http://{ROBOT_IP}/api/cameras/rgb"
    headers = {"Accept": "image/jpeg"}
    
    try:
        response = requests.get(url, headers=headers, stream=True, timeout=10)
        if response.status_code == 200:
            # Read the image data from the response
            img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            if frame is not None:
                # Save the image locally
                cv2.imwrite(filename, frame)
                print(f"Saved image: {filename}")
                return True
            else:
                print("Failed to decode image.")
                return False
        else:
            print(f"Error fetching frame: {response.status_code} - {response.text}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"Error fetching image: {e}")
        return False

def move_forward_1m():
    """
    Moves Misty forward by 1 meter.
    """
    # Misty's linear velocity is in mm/s
    speed_mm_s = 250  # Adjust speed as needed
    distance_mm = 750  # 1 meter = 1000 mm
    duration_s = distance_mm / speed_mm_s
    misty.DriveTime(speed_mm_s, 0, int(duration_s * 1000))
    time.sleep(duration_s + 1)  # Wait for the movement to complete

def rotate_left_90():
    """
    Rotates Misty left by 90 degrees.
    """
    angular_speed_deg_s = 30  # degrees per second
    angle_deg = 100 # degrees
    duration_s = angle_deg / angular_speed_deg_s
    misty.DriveTime(0, angular_speed_deg_s, int(duration_s * 1000))
    time.sleep(duration_s + 1)

def rotate_right_90():
    """
    Rotates Misty right by 90 degrees.
    """
    angular_speed_deg_s = -30  # negative for right rotation
    angle_deg = 100  # degrees
    duration_s = angle_deg / abs(angular_speed_deg_s)
    misty.DriveTime(0, angular_speed_deg_s, int(duration_s * 1000))
    time.sleep(duration_s + 1)

def on_press(key):
    """
    Handles key press events. Starts the image capture sequence when Tab is pressed.
    Exits the program when ESC is pressed.
    
    Parameters:
    - key: The key that was pressed.
    """
    if key == keyboard.Key.tab:
        print("Tab key pressed. Starting image capture sequence...")
        
        # Step 1: Move forward to the new position
        move_forward_1m()
        timestamp = int(time.time())
        
        # Step 2: Capture image facing forward
        front_img = os.path.join(PICTURE_DIR, f"position_{timestamp}_0_front.jpg")
        capture_image(front_img)
        
        # Step 3: Rotate left 90 degrees and capture left image
        rotate_left_90()
        left_img = os.path.join(PICTURE_DIR, f"position_{timestamp}_1_left.jpg")
        capture_image(left_img)
        
        # Step 4: Rotate left 90 degrees and capture back image
        rotate_left_90()
        back_img = os.path.join(PICTURE_DIR, f"position_{timestamp}_2_back.jpg")
        capture_image(back_img)
        
        # Step 5: Rotate left 90 degrees and capture right image
        rotate_left_90()
        right_img = os.path.join(PICTURE_DIR, f"position_{timestamp}_3_right.jpg")
        capture_image(right_img)
        
        # Step 6: Rotate left 90 degrees to return to original orientation
        rotate_left_90()
        
        print("Image capture sequence completed.")
    
    elif key == keyboard.Key.esc:
        # Stop listener
        print("Escape key pressed. Exiting.")
        return False

def start_key_listener():
    """
    Starts the keyboard listener to detect key presses.
    """
    with keyboard.Listener(on_press=on_press) as listener:
        print("Listening for Tab key press to start image capture. Press ESC to exit.")
        listener.join()

if __name__ == "__main__":
    # Start the key listener in the main thread
    try:
        start_key_listener()
    except KeyboardInterrupt:
        print("Program terminated by user.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
