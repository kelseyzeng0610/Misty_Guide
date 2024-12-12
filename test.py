# import yaml
# import numpy as np
# from PIL import Image
# import math
# import time
# import websocket
# import logging
# import json
# import random
# import re
# import cv2
# import os
# import requests
# import matplotlib.pyplot as plt
# from collections import deque
# import threading
# import base64
# import io

# # Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# # Constants
# ORIGIN = np.array([-15.4, -12.2, 0.0])  
# RESOLUTION = 0.05  
# CEILING_GRID_SIZE = 7

# LOOP_INTERVAL = 0.5

# occupied_threshold = 0.65
# free_threshold = 0.196
# HEIGHT = int(15.4 / 0.05)  # 308
# WIDTH = int(15.4 / 0.05)   # 308
# blocked_columns = [4, 5, 6]

# wheel_base = 0.11  # Wheel base (track width) in meters

# # Helper Functions
# def world_to_grid(world_x, world_y):
#     grid_x = int((world_x - ORIGIN[0]) / RESOLUTION)
#     grid_y = int((world_y - ORIGIN[1]) / RESOLUTION)
    
#     # Check lower bounds
#     if grid_x < 0 or grid_y < 0:
#         return None, None
        
#     # Check upper bounds
#     if grid_x >= WIDTH or grid_y >= HEIGHT:
#         return None, None
        
#     return grid_x, grid_y

# def grid_to_world(grid_x, grid_y):
#     if grid_x < 0 or grid_y < 0 or grid_x >= WIDTH or grid_y >= HEIGHT:
#             return None, None
        
#     world_x = grid_x * RESOLUTION + ORIGIN[0]
#     world_y = grid_y * RESOLUTION + ORIGIN[1]
        
#     return world_x, world_y

# def grid_to_ceiling(grid_x, grid_y):
#     """
#     Convert grid coordinates to ceiling map coordinates.
    
#     Returns:
#     - (int, int): Ceiling map coordinates (ceiling_x, ceiling_y) or (None, None) if out of bounds.
#     """
#     if grid_x < 0 or grid_y < 0 or grid_x >= WIDTH or grid_y >= HEIGHT:
#         return None, None
    
#     ceiling_x = grid_x // CEILING_GRID_SIZE
#     ceiling_y = grid_y // CEILING_GRID_SIZE
#     # Calculate maximum ceiling indices based on map dimensions
#     max_ceiling_x = WIDTH // CEILING_GRID_SIZE
#     max_ceiling_y = HEIGHT // CEILING_GRID_SIZE
    
#     if ceiling_x >= max_ceiling_x or ceiling_y >= max_ceiling_y:
#         return None, None
    
#     return ceiling_x, ceiling_y

# # Particle Filter Class
# class ParticleFilter:
#     def __init__(self, num_particles, grid_map, origin, resolution, ceiling_grid_size=6):
#         self.num_particles = num_particles
#         self.grid_map = grid_map  # Dictionary with ceiling cell as key and image path as value
#         self.origin = origin
#         self.resolution = resolution
#         self.ceiling_grid_size = ceiling_grid_size
#         self.particles = self.initialize_particles()
#         self.particle_weights = np.ones(self.num_particles) / self.num_particles

#     def initialize_particles(self):
#         """
#         Initialize particles randomly in free cells.
#         """
#         particles = []
#         for _ in range(self.num_particles):
#             px, py = self.grid_map['random_free_cell']()
#             particles.append([px, py])
#         return np.array(particles, dtype=float)

#     def move_particles(self, dx, dy, sigma=0.1):
#         """
#         Move particles based on control input with added Gaussian noise.
#         """
#         noise = np.random.normal(0, sigma, self.particles.shape)
#         movement = np.array([dx, dy])
#         self.particles += movement + noise

#         # Optionally, implement boundary conditions here
#         # For example, reflect particles that move out of bounds
#         # Currently, particles can go out of bounds

#     def image_similarity(self, expected_image, actual):
#         """
#         Compute similarity between expected and actual images using histogram comparison.
#         """
#         if expected_image is None or actual is None:
#             return 0.0

#         # Convert images to HSV
#         actual_hsv = cv2.cvtColor(actual, cv2.COLOR_BGR2HSV)
#         expected_hsv = cv2.cvtColor(expected_image, cv2.COLOR_BGR2HSV)

#         # Calculate histograms
#         actual_hist = cv2.calcHist([actual_hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
#         expected_hist = cv2.calcHist([expected_hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])

#         # Normalize histograms
#         cv2.normalize(actual_hist, actual_hist)
#         cv2.normalize(expected_hist, expected_hist)

#         # Compute similarity using Bhattacharyya distance
#         similarity = 1 - cv2.compareHist(actual_hist, expected_hist, cv2.HISTCMP_BHATTACHARYYA)

#         return max(0.0, similarity)

#     def update_weights(self, actual_image):
#         """
#         Update particle weights based on image similarity.
#         """
#         for i, particle in enumerate(self.particles):
#             grid_x, grid_y = world_to_grid(particle[0], particle[1])
#             if grid_x is None or grid_y is None:
#                 self.particle_weights[i] = 0.0
#                 continue

#             ceiling_x, ceiling_y = grid_to_ceiling(grid_x, grid_y)
#             if ceiling_x is None or ceiling_y is None:
#                 self.particle_weights[i] = 0.0
#                 continue

#             ceiling_key = (ceiling_y, ceiling_x)  # Ensure (y, x)

#             if ceiling_key not in self.grid_map['ceiling_images']:
#                 self.particle_weights[i] = 0.0
#             else:
#                 similarity = []
#                 ceiling_images = self.grid_map['ceiling_images'][ceiling_key]   
#                 for orientation, image_path in ceiling_images.items():
#                     expected_image = cv2.imread(image_path)
#                     if expected_image is None:
#                         logging.warning(f"Expected image not found: {image_path}")
#                         continue
#                     sim = self.image_similarity(expected_image, actual_image)
#                     similarity.append(sim)
#                 if similarity:
#                     self.particle_weights[i] = max(similarity)
#                 else:
#                     self.particle_weights[i] = 0.0

#         # Normalize weights
#         total_weight = np.sum(self.particle_weights)
#         if total_weight == 0:
#             # Avoid division by zero; assign uniform weights
#             self.particle_weights = np.ones(self.num_particles) / self.num_particles
#         else:
#             # Add a small value to avoid zero weights
#             self.particle_weights += 1e-6
#             self.particle_weights /= np.sum(self.particle_weights)

#     def resample(self):
#         """
#         Resample particles based on their weights.
#         """
#         indices = np.random.choice(
#             range(self.num_particles),
#             self.num_particles,
#             p=self.particle_weights
#         )
#         self.particles = self.particles[indices]
#         self.particle_weights.fill(1.0 / self.num_particles)

#     def estimate_position(self):
#         """
#         Estimate the robot's position as the weighted average of particle positions.
#         """
#         x_estimate = np.average(self.particles[:, 0], weights=self.particle_weights)
#         y_estimate = np.average(self.particles[:, 1], weights=self.particle_weights)
#         return x_estimate, y_estimate

#     def draw_particles(self, estimated_position=None):
#         """
#         Visualize particles and estimated position.
#         """
#         plt.clf()
#         # Plot particles
#         plt.scatter(
#             self.particles[:, 0],
#             self.particles[:, 1],
#             s=10,
#             c=self.particle_weights,
#             cmap='viridis',
#             alpha=0.5,
#             label='Particles'
#         )
#         # Plot estimated position
#         if estimated_position is not None:
#             plt.scatter(
#                 estimated_position[0],
#                 estimated_position[1],
#                 c='blue',
#                 marker='+',
#                 s=100,
#                 label='Estimated Position'
#             )
#         plt.legend()
#         plt.xlabel("World X (meters)")
#         plt.ylabel("World Y (meters)")
#         plt.title("Particle Filter Localization")
#         plt.grid(True)
#         plt.pause(0.01)

# # WebSocket Client Class
# class MistyWebSocketClient:
#     def __init__(self, ip, on_message_callback):
#         self.ip = ip
#         self.ws_url = f"ws://{ip}/pubsub"
#         self.on_message_callback = on_message_callback
#         self.ws = None
#         self.subscribed = False
    
#     def on_message(self, ws, message):
#         """
#         Callback function when a message is received from the WebSocket.
#         """
#         try:
#             # Determine the type of the incoming message
#             if isinstance(message, str):
#                 if not message.strip():
#                     logging.warning("Received empty message. Skipping.")
#                     return
#                 event_data = json.loads(message)
#             elif isinstance(message, (bytes, bytearray)):
#                 message_str = message.decode('utf-8').strip()
#                 if not message_str:
#                     logging.warning("Received empty byte message. Skipping.")
#                     return
#                 event_data = json.loads(message_str)
#             elif isinstance(message, dict):
#                 event_data = message
#             else:
#                 logging.warning(f"Unknown message type: {type(message)}. Skipping.")
#                 return

#             if not isinstance(event_data, dict):
#                 logging.warning(f"Parsed event_data is not a dict: {event_data}. Skipping.")
#                 return

#             event_name = event_data.get("eventName")
#             event_message = event_data.get("message")

#             if event_name in ["IMUEvent", "DriveEncoders"]:
#                 # Check if the message is a registration status error
#                 if isinstance(event_message, str) and "Registration Status" in event_message:
#                     logging.error(f"Subscription error for {event_name}: {event_message}")
#                     return
#                 # Assuming event_message is a JSON string or dict
#                 if isinstance(event_message, str):
#                     try:
#                         event_message = json.loads(event_message)
#                     except json.JSONDecodeError:
#                         logging.error(f"Event message for {event_name} is not a valid JSON: {event_message}")
#                         return
#                 if isinstance(event_message, dict):
#                     self.on_message_callback(event_name, event_message)
#                 else:
#                     logging.error(f"Event message for {event_name} is not a dict: {event_message}")
#             else:
#                 logging.info(f"Ignoring irrelevant event: {event_name}")
#         except json.JSONDecodeError as e:
#             logging.error(f"JSON Decode Error: {e} | Message: {message}")
#         except Exception as e:
#             logging.error(f"Error processing message: {e} | Message: {message}")

#     def on_error(self, ws, error):
#         logging.error(f"WebSocket Error: {error}")
    

#     def on_close(self, ws, close_status_code, close_msg):
#         logging.info(f"WebSocket Closed: {close_status_code} - {close_msg}")
        

#     def on_open(self, ws):
#         """
#         Callback function when the WebSocket connection is opened.
#         Subscribes to IMU and DriveEncoders events.
#         """
#         if not self.subscribed:
#             imu_subscription = {
#                 "Operation": "subscribe",
#                 "Type": "IMU",
#                 "DebounceMs": 200,
#                 "EventName": "IMUEvent"
#             }
#             encoders_subscription = {
#                 "Operation": "subscribe",
#                 "Type": "DriveEncoders",
#                 "DebounceMs": 200,
#                 "EventName": "DriveEncoders"
#             }
#             ws.send(json.dumps(imu_subscription))
#             ws.send(json.dumps(encoders_subscription))
#             self.subscribed = True
#             logging.info("Subscribed to IMU and DriveEncoders events.")

#     def run_forever(self):
#         """
#         Starts the WebSocket connection and runs it forever.
#         """
#         self.ws = websocket.WebSocketApp(
#             self.ws_url,
#             on_message=self.on_message,
#             on_error=self.on_error,
#             on_close=self.on_close
#         )
#         self.ws.on_open = self.on_open

#         # Run WebSocket in a separate thread
#         self.thread = threading.Thread(target=self.ws.run_forever)
#         self.thread.daemon = True
#         self.thread.start()
#         logging.info("WebSocket thread started.")

# # Load Grid Map Function
# def load_grid_map(pgm_path):
#     """
#     Load the grid map and prepare the ceiling images library.
#     Assumes ceiling images are stored in 'ceiling_images/y_x_orientation.jpg/png'.
#     """
#     if not os.path.exists(pgm_path):
#         raise FileNotFoundError(f"Map file {pgm_path} not found.")

#     # Load the map image in grayscale
#     img = Image.open(pgm_path).convert('L')  # Convert to grayscale
#     img = np.array(img, dtype=np.uint8) / 255.0  # Normalize to [0,1]
    
#     # Log unique pixel values
#     unique_values = np.unique(img)
#     logging.info(f"Unique pixel values in map: {unique_values}")
    
#     # Plot histogram of pixel values for better understanding
#     plt.figure(figsize=(6,4))
#     plt.hist(img.ravel(), bins=256, range=(0,1), color='gray')
#     plt.title("Pixel Value Distribution in Map")
#     plt.xlabel("Pixel Value (Normalized)")
#     plt.ylabel("Frequency")
#     plt.show()
    
#     THRESHOLD =0.81  # Using the defined occupied_threshold
#     height, width = img.shape

#     grid_map = {
#         "map_array": img,
#         "height": height,
#         "width": width,
#         "ceiling_images": {}
#     }

#     ceiling_images_dir = 'ceiling_images'
#     if not os.path.exists(ceiling_images_dir):
#         raise FileNotFoundError(f"Ceiling images directory '{ceiling_images_dir}' not found.")

#     # Regular expression to match the renamed filename pattern
#     # Expected format: y_x_orientation.jpg/png
#     pattern = re.compile(r'^(\d+)_(\d+)_(front|left|back|right)\.(jpg|png)$', re.IGNORECASE)

#     for filename in os.listdir(ceiling_images_dir):
#         if filename.lower().endswith(('.jpg', '.png')):
#             match = pattern.match(filename)
#             if match:
#                 y = int(match.group(1))
#                 x = int(match.group(2))
#                 orientation = match.group(3).lower()
#                 extension = match.group(4).lower()

#                 # Skip blocked columns
#                 if x in blocked_columns:
#                     logging.info(f"Skipping blocked column for file: {filename}")
#                     continue

#                 # Initialize the dict for (y, x) if not already
#                 if (y, x) not in grid_map["ceiling_images"]:
#                     grid_map["ceiling_images"][(y, x)] = {}

#                 # Assign the image path to the corresponding orientation
#                 ceiling_image_path = os.path.join(ceiling_images_dir, filename)
#                 grid_map["ceiling_images"][(y, x)][orientation] = ceiling_image_path
#                 logging.info(f"Loaded image for grid ({y}, {x}) - {orientation}: {filename}")
#             else:
#                 logging.warning(f"Invalid ceiling image filename format: {filename}. Expected 'y_x_orientation.jpg/png'. Skipping.")
#                 continue

#     def random_free_cell():
#         attempts = 0
#         max_attempts = 10000  # To prevent infinite loops

#         while attempts < max_attempts:
#             x = random.randint(0, width - 1)
#             y = random.randint(0, height - 1)
#             pixel_value = grid_map["map_array"][y, x]

#             if pixel_value <= THRESHOLD:
#                 world_x, world_y = grid_to_world(x, y)
#                 if world_x is not None and world_y is not None:
#                     return world_x, world_y

#             attempts += 1

#         raise ValueError("Failed to find a free cell within the maximum number of attempts.")

#     grid_map['random_free_cell'] = random_free_cell

#     return grid_map

# # Robot Localizer Class
# class RobotLocalizer:
#     def __init__(self, map_image_path, misty_ip, num_particles=300):
#         self.grid_map = load_grid_map(map_image_path)
#         self.origin = ORIGIN
#         self.resolution = RESOLUTION
#         self.pf = ParticleFilter(
#             num_particles=num_particles,
#             grid_map=self.grid_map,
#             origin=self.origin,
#             resolution=self.resolution,
#             ceiling_grid_size=CEILING_GRID_SIZE
#         )
#         self.data_queue = deque()
#         self.misty_ip = misty_ip
        
#         self.current_position = np.array([0.0, 0.0])
#          # To ensure thread-safe operations
#         self.lock = threading.Lock()

    

#     def handle_sensor_data(self, event_type, message):
#         """
#         Callback to handle incoming sensor data and enqueue it for processing.
#         """
#         with self.lock:
#             if not isinstance(message, dict):
#                 logging.warning(f"Received non-dict message for event {event_type}: {message}")
#                 return
#             self.data_queue.append((event_type, message))

#     def process_imu_data(self, event_data):
#         """
#         Process IMU data if needed.
#         Currently not used, but can be integrated for better movement estimation.
#         """
#         # Placeholder for processing IMU data
#         pass

#     def fetch_camera_image(self):
#         """
#         Fetch the latest camera image from Misty via HTTP.
#         """
#         camera_url = f"http://{self.misty_ip}/api/cameras/rgb"
#         headers = {"Accept": "image/jpeg"}
#         try:
#             response = requests.get(camera_url, headers=headers, stream=True, timeout=5)
#             if response.status_code == 200:
#                 img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
#                 frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
#                 if frame is not None:
#                     # Resize the image if necessary
#                     frame = cv2.resize(frame, (720, 1080))  # (width, height)
#                     return frame
#                 else:
#                     logging.error("Failed to decode the image. Frame is None.")
#             else:
#                 logging.error(f"Error fetching frame: {response.status_code} - {response.text}")
#         except requests.exceptions.RequestException as e:
#             logging.error(f"Request Exception while fetching image: {e}")
#         return None

#     def extract_movement(self, event_type, data):
#         """
#         Convert encoder velocities to movement deltas (dx, dy).
#         """
#         if event_type == "DriveEncoders":
#             left_velocity_mm_s = data.get("leftVelocity", 0.0)
#             right_velocity_mm_s = data.get("rightVelocity", 0.0)
#             left_v = left_velocity_mm_s / 1000.0  # Convert mm/s to m/s 
#             right_v = right_velocity_mm_s / 1000.0  # Convert mm/s to m/s
#             v = (left_v + right_v) / 2.0  # Linear velocity
#             omega = (right_v - left_v) / wheel_base  # Angular velocity
#             dt = LOOP_INTERVAL  # Time interval in seconds
#             dx = v * math.cos(omega * dt) * dt
#             dy = v * math.sin(omega * dt) * dt  # Assuming robot moves in the direction of heading
#             return dx, dy
#         else:
#             return 0.0, 0.0
#     def start_websocket(self):
#         """
#         Start the WebSocket client in a separate thread.
#         """
#         with self.lock:
#             if not self.websocket_started:
#                 self.websocket_client.run_forever()
#                 self.websocket_started = True
#                 logging.info("WebSocket client started.")
#             else:
#                 logging.info("WebSocket client is already running.")

#     def run(self, step=0):
#         """
#         Main loop to process sensor data, update the particle filter, and visualize.
#         """
#         plt.figure(figsize=(8, 8))
#         plt.ion()  # Interactive mode on

#         while step < 30:
#             with self.lock:
#                 if self.data_queue:
#                     event_type, message = self.data_queue.popleft()
#                     if event_type == "DriveEncoders":
#                         dx, dy = self.extract_movement(event_type, message)
#                         self.pf.move_particles(dx, dy)
#                         # Fetch the latest camera image
#                         actual_image = self.fetch_camera_image()
#                         if actual_image is not None:
#                             self.pf.update_weights(actual_image)
#                             self.pf.resample()
#                             estimated_position = self.pf.estimate_position()
#                             self.current_position = np.array(estimated_position)
#                             self.pf.draw_particles(estimated_position)
#                             print(f"Step {step + 1}: Estimated Position = ({estimated_position[0]:.2f}, {estimated_position[1]:.2f})")
#                             step += 1
#                     elif event_type == "IMUEvent":
#                         # Currently, IMU data is not directly used in movement calculation
#                         # You can integrate IMU data to enhance movement estimates if needed
#                         self.process_imu_data(message)

#             plt.pause(0.01)
#             time.sleep(LOOP_INTERVAL)

# # Entry Point
# if __name__ == "__main__":
#     # Configuration
#     MISTY_IP = "10.5.11.234"  # Replace with your Misty's IP address
#     MAP_IMAGE_PATH = "rotated_lab_474.pgm"  # Replace with your map image path

#     # Test helper functions
#     grid_x_test, grid_y_test = world_to_grid(-17, -12.2)
#     logging.info(f"world_to_grid(-17, -12.2) = ({grid_x_test}, {grid_y_test})")
#     world_x_test, world_y_test = grid_to_world(0, 0)
#     logging.info(f"grid_to_world(0, 0) = ({world_x_test}, {world_y_test})")
#     ceiling_x_test, ceiling_y_test = grid_to_ceiling(8, 3)
#     logging.info(f"grid_to_ceiling(8, 3) = ({ceiling_x_test}, {ceiling_y_test})")

#     # Initialize Robot Localizer
#     try:
#         localizer = RobotLocalizer(
#             map_image_path=MAP_IMAGE_PATH,
#             misty_ip=MISTY_IP,
#             num_particles=300
#         )
#     except FileNotFoundError as e:
#         logging.error(e)
#         exit(1)

#     # Start WebSocket Client
#     localizer.websocket_client.run_forever()

#     # Run Localization
#     try:
#         localizer.run()
#     except KeyboardInterrupt:
#         logging.info("Stopping the program...")
#     except Exception as e:
#         logging.error(f"An error occurred: {e}")



    