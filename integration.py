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
import yaml
from PIL import Image
import logging
import random
import re
import os
from queue import Queue
import asyncio
import io

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# -----------------------
# CONFIGURATIONS
# -----------------------
MISTY_IP = "10.5.9.252"  # Replace with your Misty's IP address
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

misty = Robot(MISTY_IP)

# Global Flags
stop_requested = False
obstacle_detected = False
obstacle_processing = False
ws = None  # WebSocket connection

data_queue = Queue()  
subscribed = False
last_obstacle_cleared_time = 0.0
COOLDOWN_PERIOD = 3.0 

# -----------------------
# PARTICLE FILTER 
# -----------------------
wheel_base = 0.11 
ORIGIN = np.array([-15.4, -12.2, 0.0])
RESOLUTION = 0.05
CEILING_GRID_SIZE = 7
WIDTH_MAX = 0
HEIGHT_MAX = 0
WIDTH_MIN = -15.4
HEIGHT_MIN = -12.2

occupied_threshold = 0.65
free_threshold = 0.196
HEIGHT = int(15.4 / 0.05)  # 308
WIDTH = int(15.4 / 0.05)   # 308
blocked_columns = [4, 5, 6]

def world_to_grid(world_x, world_y):
    try:
        grid_x = int((world_x - ORIGIN[0]) / RESOLUTION)
        grid_y = int((world_y - ORIGIN[1]) / RESOLUTION)
        if grid_x < 0 or grid_y < 0 or grid_x >= WIDTH or grid_y >= HEIGHT:
            return None, None
        return grid_x, grid_y
    except:
        return None, None

def grid_to_world(grid_x, grid_y):
    try:
        if grid_x < 0 or grid_y < 0 or grid_x >= WIDTH or grid_y >= HEIGHT:
            return None, None
        world_x = grid_x * RESOLUTION + ORIGIN[0]
        world_y = grid_y * RESOLUTION + ORIGIN[1]
        return world_x, world_y
    except:
        return None, None

def grid_to_ceiling(grid_x, grid_y):
    try:
        if grid_x is None or grid_y is None:
            return None, None
        grid_x = int(grid_x)
        grid_y = int(grid_y)
        if not (0 <= grid_x < WIDTH and 0 <= grid_y < HEIGHT):
            return None, None
        ceiling_x = grid_x // CEILING_GRID_SIZE
        ceiling_y = grid_y // CEILING_GRID_SIZE
        max_ceiling_x = WIDTH // CEILING_GRID_SIZE
        max_ceiling_y = HEIGHT // CEILING_GRID_SIZE
        if not (0 <= ceiling_x < max_ceiling_x and 0 <= ceiling_y < max_ceiling_y):
            return None, None
        return ceiling_x, ceiling_y
    except:
        return None, None

class ParticleFilter:
    def __init__(self, num_particles, grid_map, origin, resolution, ceiling_grid_size=6):
        self.num_particles = num_particles
        self.grid_map = grid_map  
        self.origin = origin
        self.resolution = resolution
        self.ceiling_grid_size = ceiling_grid_size
        self.particles = self.initialize_particles()
        self.particle_weights = np.ones(self.num_particles) / self.num_particles

    def initialize_particles(self):
        particles = np.random.uniform(
            low=[-15.4, -12.2], high=[0, 0], size=(self.num_particles, 2)
        )
        return particles

    def move_particles(self, dx, dy, yaw, sigma=0.1):
        for i, particle in enumerate(self.particles):
            noise = np.random.normal(0, sigma, size=2)
            particle[0] += dx + noise[0]
            particle[1] += dy + noise[1]
            particle[0] = np.clip(particle[0], WIDTH_MIN, WIDTH_MAX)
            particle[1] = np.clip(particle[1], HEIGHT_MIN, HEIGHT_MAX)

    def image_similarity(self, expected_image, actual):
        if expected_image is None or actual is None:
            return 0.0
        actual_hsv = cv2.cvtColor(actual, cv2.COLOR_BGR2HSV)
        expected_hsv = cv2.cvtColor(expected_image, cv2.COLOR_BGR2HSV)
        actual_hist = cv2.calcHist([actual_hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
        expected_hist = cv2.calcHist([expected_hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
        cv2.normalize(actual_hist, actual_hist)
        cv2.normalize(expected_hist, expected_hist)
        similarity = 1 - cv2.compareHist(actual_hist, expected_hist, cv2.HISTCMP_BHATTACHARYYA)
        return max(0.0, similarity)

    def update_weights(self, actual_image):
        if actual_image is None:
            logging.error("Received None actual_image")
            return
        for i, particle in enumerate(self.particles):
            try:
                if np.isnan(particle[0]) or np.isnan(particle[1]):
                    self.particle_weights[i] = 0.0
                    continue
                grid_x, grid_y = world_to_grid(particle[0], particle[1])
                if grid_x is None or grid_y is None:
                    self.particle_weights[i] = 0.0
                    continue
                ceiling_x, ceiling_y = grid_to_ceiling(grid_x, grid_y)
                if ceiling_x is None or ceiling_y is None:
                    self.particle_weights[i] = 0.0
                    continue
                ceiling_key = (ceiling_y, ceiling_x)
                if ceiling_key not in self.grid_map['ceiling_images']:
                    self.particle_weights[i] = 0.0
                    continue

                similarities = []
                ceiling_images = self.grid_map['ceiling_images'][ceiling_key]
                for orientation, image_path in ceiling_images.items():
                    if not os.path.exists(image_path):
                        continue
                    expected_image = cv2.imread(image_path)
                    if expected_image is not None:
                        similarity = self.image_similarity(expected_image, actual_image)
                        similarities.append(similarity)
                    else:
                        continue
                if similarities:
                    self.particle_weights[i] = max(similarities)
                else:
                    self.particle_weights[i] = 0.0
            except:
                self.particle_weights[i] = 0.0

        total_weight = np.sum(self.particle_weights)
        if total_weight > 0:
            self.particle_weights /= total_weight
        else:
            self.particle_weights = np.ones(self.num_particles) / self.num_particles

    def resample(self):
        indices = np.random.choice(
            range(self.num_particles),
            self.num_particles,
            p=self.particle_weights
        )
        self.particles = self.particles[indices]
        self.particle_weights.fill(1.0 / self.num_particles)

    def estimate_position(self):
        x_estimate = np.average(self.particles[:, 0], weights=self.particle_weights)
        y_estimate = np.average(self.particles[:, 1], weights=self.particle_weights)
        return x_estimate, y_estimate

def load_grid_map(pgm_path):
    if not os.path.exists(pgm_path):
        raise FileNotFoundError(f"Map file {pgm_path} not found.")
    img = Image.open(pgm_path).convert('L')
    img = np.array(img, dtype=np.uint8)/255.0
    height, width = img.shape
    grid_map = {
        "map_array": img,
        "height": height,
        "width": width,
        "ceiling_images": {}
    }
    ceiling_images_dir = 'ceiling_images'
    if not os.path.exists(ceiling_images_dir):
        raise FileNotFoundError(f"Ceiling images directory '{ceiling_images_dir}' not found.")

    pattern = re.compile(r'^(\d+)_(\d+)_(front|left|back|right)\.(jpg|png)$', re.IGNORECASE)
    unique_vals = np.unique(img)
    THRESHOLD = unique_vals[1] if len(unique_vals) > 1 else 0.5

    for filename in os.listdir(ceiling_images_dir):
        if filename.lower().endswith('.jpg') or filename.lower().endswith('.png'):
            match = pattern.match(filename)
            if match:
                y = int(match.group(1))
                x = int(match.group(2))
                orientation = match.group(3).lower()
                if x in [4, 5, 6]:
                    continue
                if (y, x) not in grid_map["ceiling_images"]:
                    grid_map["ceiling_images"][(y, x)] = {}
                ceiling_image_path = os.path.join(ceiling_images_dir, filename)
                grid_map["ceiling_images"][(y, x)][orientation] = ceiling_image_path
            else:
                continue

    def random_free_cell():
        attempts = 0
        max_attempts = 10000
        while attempts < max_attempts:
            x = random.randint(0, width - 1)
            y = random.randint(0, height - 1)
            pixel_value = grid_map["map_array"][y, x]
            if pixel_value.all() <= THRESHOLD and (x // 7 not in blocked_columns):
                world_x, world_y = grid_to_world(x, y)
                return world_x, world_y
            attempts += 1

    grid_map['random_free_cell'] = random_free_cell
    return grid_map

class RobotLocalizer:
    def __init__(self, map_image_path, misty_ip, num_particles=100):
        self.grid_map = load_grid_map(map_image_path)
        self.origin = ORIGIN
        self.resolution = RESOLUTION
        self.pf = ParticleFilter(
            num_particles=num_particles,
            grid_map=self.grid_map,
            origin=self.origin,
            resolution=self.resolution,
            ceiling_grid_size=CEILING_GRID_SIZE
        )
        self.misty_ip = misty_ip
        self.current_position = np.array([0.0, 0.0])
        self.lock = threading.Lock()
        self.dx = 0.0
        self.dy = 0.0
        self.yaw = 0.0  # orientation
        self.vx = 0.0
        self.omega = 0.0

    def fetch_camera_image(self):
        misty.EnableCameraService()
        camera_url = f"http://{self.misty_ip}/api/cameras/rgb"
        headers = {"Accept": "image/jpeg"}
        try:
            response = requests.get(camera_url, headers=headers, stream=True, timeout=5)
            if response.status_code == 200:
                img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
                frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                if frame is not None:
                    frame = cv2.resize(frame, (720, 1080))
                    return frame
        except requests.exceptions.RequestException:
            pass
        return None

    def extract_movement(self, event_type, data):
        # Simplified: Just get linear/rotational velocities
        interval = 0.1
        if event_type == "DriveEncoders":
            left_dist = data.get("message", {}).get("leftDistance", 0.0) / 1000.0
            right_dist = data.get("message", {}).get("rightDistance", 0.0) / 1000.0
            self.vx = (left_dist + right_dist) / 2.0
        elif event_type == "IMU":
            yaw_velocity_deg = data.get("message", {}).get("yawVelocity", 0.0)
            yaw_velocity = math.radians(yaw_velocity_deg)
            self.yaw += yaw_velocity * interval
            self.vx += data.get("message", {}).get("xAcceleration", 0.0) * interval

        dx = self.vx * math.cos(self.yaw) * interval
        dy = self.vx * math.sin(self.yaw) * interval
        return dx, dy

    def run(self):
        step = 0
        fusion_interval = 0.01
        particle_interval = 0.1
        plot_interval = 2
        last_fusion_time = time.time()
        last_particle_time = time.time()

        while not stop_requested:
            with self.lock:
                current_time = time.time()
                if current_time - last_fusion_time >= fusion_interval:
                    if not data_queue.empty():
                        data_ = data_queue.get()
                        event_type = data_.get("event_type")
                        dx, dy = self.extract_movement(event_type, data_)
                        self.dx += dx
                        self.dy += dy
                    last_fusion_time = current_time

                if current_time - last_particle_time >= particle_interval:
                    # Move particles
                    self.pf.move_particles(self.dx, self.dy, self.yaw)
                    self.dx = 0.0
                    self.dy = 0.0
                    actual_image = self.fetch_camera_image()
                    if actual_image is not None:
                        self.pf.update_weights(actual_image)
                        estimated_position = self.pf.estimate_position()
                        self.pf.resample()
                        self.current_position = np.array(estimated_position)
                        # Just print the estimated position instead of plotting
                        if step % plot_interval == 0:
                            print(f"Step {step}: Estimated Position: {estimated_position}")
                        step += 1
                    last_particle_time = current_time
            time.sleep(0.01)

# -----------------------
# YOLO & OBSTACLE AVOIDANCE 
# -----------------------
def wait_for_keypress():
    global stop_requested
    input("Press ENTER to stop the robot.\n")
    stop_requested = True

def fetch_misty_camera_frame():
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

def smooth_stop():
    for speed in range(10, 0, -5):
        misty.Drive(speed, 0)
        time.sleep(0.2)
    misty.Stop()
    print("Robot smoothly stopped.")

def handle_tof(data):
    global obstacle_detected, obstacle_processing, last_obstacle_cleared_time
    if stop_requested or obstacle_detected:
        return

    # Check cooldown
    current_time = time.time()
    if (current_time - last_obstacle_cleared_time) < COOLDOWN_PERIOD:
        # Still in cooldown period, ignore this TOF event
        return

    tof_message = data.get("message", {})
    sensor_position = tof_message.get("sensorPosition", "")
    distance = tof_message.get("distanceInMeters", None)

    if sensor_position == "Center" and distance is not None:
        print(f"Distance in meters (Center): {distance}")
        if 0.4 < distance < 0.6:  # Slow down if obstacle is near
            print("Slowing down due to nearby obstacle.")
            misty.Drive(10, 0)  # Reduce speed
        elif distance <= 0.4:  # Stop if very close
            print("Obstacle detected! Preparing to stop.")
            obstacle_detected = True
            smooth_stop()
            announce("Obstacle detected. Analyzing the situation.")

            if not obstacle_processing:
                obstacle_processing = True
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
                obstacle_detected = False
                print("Debouncing...")
                time.sleep(2)

                # Reset processing flag and record the time
                obstacle_processing = False
                last_obstacle_cleared_time = time.time()  # Record the timestamp

def keep_moving_forward():
    while not stop_requested:
        if not obstacle_detected:
            misty.Drive(20, 0)  
            time.sleep(0.1)
        else:
            time.sleep(0.1)
    smooth_stop()
    print("Robot stopped by user.")

# -----------------------
# WEBSOCKET CALLBACKS
# -----------------------
def on_message(ws, message):
    try:
        data = json.loads(message) 
    except json.JSONDecodeError:
     
        return

    eventName = data.get("eventName", "")
    if eventName == "TimeOfFlight":
        handle_tof(data)
    elif eventName == "IMUEvent":
        data_dict = {"event_type":"IMU","message":data}
        data_queue.put(data_dict)
    elif eventName == "EncoderEvent":
        data_dict = {"event_type":"DriveEncoders","message":data}
        data_queue.put(data_dict)
    else:
        # Unknown event, ignore or log
        pass

def on_error(ws, error):
    print("WebSocket Error:", error)

def on_close(ws, close_status_code, close_msg):
    print("WebSocket connection closed")

def on_open(ws):
    global subscribed
    print("WebSocket connection opened.")
    # Subscribe to TOF events
    ws.send(json.dumps({
        "Operation": "subscribe",
        "Type": "TimeOfFlight",
        "DebounceMs": 100,
        "EventName": "TimeOfFlight",
        "Message": "TimeOfFlight"
    }))
    # Subscribe to IMU events
    ws.send(json.dumps({
        "Operation": "subscribe",
        "Type": "IMU",
        "DebounceMs": 50,
        "EventName": "IMUEvent"
    }))
    # Subscribe to Encoder events
    ws.send(json.dumps({
        "Operation": "subscribe",
        "Type": "DriveEncoders",
        "DebounceMs": 50,
        "EventName": "EncoderEvent"
    }))
    subscribed = True
    print("Subscribed to TOF, IMUEvent and EncoderEvent")

def start_websocket():
    global ws
    websocket_url = f"ws://{MISTY_IP}/pubsub"
    ws = websocket.WebSocketApp(websocket_url,
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close)
    ws.on_open = on_open
    ws.run_forever()

# -----------------------
# MAIN
# -----------------------
def main():
    global stop_requested
    print("Starting robot... Press ENTER to stop.")
    threading.Thread(target=wait_for_keypress, daemon=True).start()

    # Start WebSocket for TOF, IMU, Encoders
    threading.Thread(target=start_websocket, daemon=True).start()

    # Initialize localizer
    localizer = RobotLocalizer(
        map_image_path="lab_474_rotated_cropped.pgm",
        misty_ip=MISTY_IP,
        num_particles=50
    )

    # Start localizer in its own thread
    threading.Thread(target=localizer.run, daemon=True).start()

    # Start forward motion loop
    keep_moving_forward()

if __name__ == "__main__":
    main()
