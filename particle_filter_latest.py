import yaml
import numpy as np
from PIL import Image
import math
import time
import websocket
from mistyPy.Robot import Robot
import logging
import json
import random
import re
import cv2
import os
import requests
import matplotlib.pyplot as plt
from queue import Queue
import threading
import base64
import asyncio
import io
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter


logging.basicConfig(
    # Set to INFO to reduce verbosity. Change to DEBUG if detailed logs are needed.
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
wheel_base = 0.11  # Wheel base (track width) in meters
MISTY_IP = "10.5.11.234"
# MISTY_IP = "10.5.9.252"
misty = Robot(MISTY_IP)
# Constants



WIDTH_MAX = 0
HEIGHT_MAX = 0
WIDTH_MIN = -15.4
HEIGHT_MIN = -12.2

subscribed = False
ORIGIN = np.array([-15.4, -12.2, 0.0])
RESOLUTION = 0.05
CEILING_GRID_SIZE = 7
FRONT_TOF_SENSOR_ID = "toffc"
LOOP_INTERVAL = 5

occupied_threshold = 0.65
free_threshold = 0.196
HEIGHT = int(15.4 / 0.05)  # 308
WIDTH = int(15.4 / 0.05)   # 308

blocked_columns = [4, 5, 6]


data_queue = Queue()
wheel_base = 0.11  # Wheel base (track width) in meters

# Global Variables (Avoid using globals if possible)
# We'll encapsulate variables within classes

# Helper Functions


def world_to_grid(world_x, world_y):
    """Convert world coordinates to grid coordinates"""
    try:
        grid_x = int((world_x - ORIGIN[0]) / RESOLUTION)
        grid_y = int((world_y - ORIGIN[1]) / RESOLUTION)

        # Boundary checks
        if grid_x < 0 or grid_y < 0 or grid_x >= WIDTH or grid_y >= HEIGHT:
            return None, None

        return grid_x, grid_y
    except Exception as e:
        logging.error(f"Error in world_to_grid: {e}")
        return None, None


def grid_to_world(grid_x, grid_y):
    """Convert grid coordinates to world coordinates"""
    try:
        if grid_x < 0 or grid_y < 0 or grid_x >= WIDTH or grid_y >= HEIGHT:
            return None, None

        world_x = grid_x * RESOLUTION + ORIGIN[0]
        world_y = grid_y * RESOLUTION + ORIGIN[1]

        return world_x, world_y
    except Exception as e:
        logging.error(f"Error in grid_to_world: {e}")
        return None, None


def grid_to_ceiling(grid_x, grid_y):
    """Convert grid coordinates to ceiling coordinates with validation."""
    try:
        # Input validation
        if grid_x is None or grid_y is None:
            return None, None

        # Ensure integers
        grid_x = int(grid_x)
        grid_y = int(grid_y)

        # Boundary checks
        if not (0 <= grid_x < WIDTH and 0 <= grid_y < HEIGHT):
            return None, None

        ceiling_x = grid_x // CEILING_GRID_SIZE
        ceiling_y = grid_y // CEILING_GRID_SIZE

        # Check ceiling grid bounds
        max_ceiling_x = WIDTH // CEILING_GRID_SIZE
        max_ceiling_y = HEIGHT // CEILING_GRID_SIZE

        if not (0 <= ceiling_x < max_ceiling_x and 0 <= ceiling_y < max_ceiling_y):
            return None, None

        return ceiling_x, ceiling_y

    except Exception as e:
        logging.error(f"Error in grid_to_ceiling: {e}")
        return None, None
# Particle Filter Class


class ParticleFilter:
    def __init__(self, num_particles, grid_map, origin, resolution, ceiling_grid_size=6):
        self.num_particles = num_particles
        self.grid_map = grid_map  
        self.origin = origin
        self.resolution = resolution
        self.ceiling_grid_size = ceiling_grid_size
        self.particles = self.initialize_particles()
        self.particle_weights = np.ones(
            self.num_particles) / self.num_particles

    def initialize_particles(self):
        """
        Initialize particles randomly in free cells.
        """
        # particles = []
        # for _ in range(self.num_particles):
        #     px, py = self.grid_map['random_free_cell']()
        #     particles.append([px, py])
        # return np.array(particles, dtype=float)

        particles = []
        particles = np.random.uniform(
            low=[-15.4, -12.2], high=[0, 0], size=(self.num_particles, 2)
        )
        self.particles = particles
        return particles

    def move_particles(self, dx, dy, yaw, sigma=0.1):
        for i, particle in enumerate(self.particles):
            noise = np.random.normal(0, sigma, size=2)
            particle[0] += (dx * np.cos(yaw) - dy * np.sin(yaw)) + noise[0]
            particle[1] += (dx * np.sin(yaw) + dy * np.cos(yaw)) + noise[1]
            particle[0] = np.clip(particle[0], WIDTH_MIN, WIDTH_MAX)
            particle[1] = np.clip(particle[1], HEIGHT_MIN, HEIGHT_MAX)



    def image_similarity(self, expected_image, actual):
        """
        Compute similarity between expected and actual images using histogram comparison.
        """
        if expected_image is None or actual is None:
            return 0.0

        # Convert images to HSV
        actual_hsv = cv2.cvtColor(actual, cv2.COLOR_BGR2HSV)
        expected_hsv = cv2.cvtColor(expected_image, cv2.COLOR_BGR2HSV)

        # Calculate histograms
        actual_hist = cv2.calcHist([actual_hsv], [0, 1], None, [
                                   50, 60], [0, 180, 0, 256])
        expected_hist = cv2.calcHist([expected_hsv], [0, 1], None, [
                                     50, 60], [0, 180, 0, 256])

        # Normalize histograms
        cv2.normalize(actual_hist, actual_hist)
        cv2.normalize(expected_hist, expected_hist)

        # Compute similarity using Bhattacharyya distance
        similarity = 1 - \
            cv2.compareHist(actual_hist, expected_hist,
                            cv2.HISTCMP_BHATTACHARYYA)

        return max(0.0, similarity)

    def update_weights(self, actual_image):
        if actual_image is None:
            logging.error("Received None actual_image")
            return

        logging.info(f"Updating weights for {self.num_particles} particles")

        for i, particle in enumerate(self.particles):
            try:
                # Convert coordinates
                if np.isnan(particle[0]) or np.isnan(particle[1]):
                    logging.debug(f"Particle {i} has NaN coordinates")
                    self.particle_weights[i] = 0.0
                    continue

                grid_x, grid_y = world_to_grid(particle[0], particle[1])
                
                if grid_x is None or grid_y is None:
                    logging.debug(f"Invalid grid coordinates for particle {i}")
                    self.particle_weights[i] = 0.0
                    continue
               
                ceiling_x, ceiling_y = grid_to_ceiling(grid_x, grid_y)
            
                if ceiling_x is None or ceiling_y is None:
                    logging.debug(
                        f"Invalid ceiling coordinates for particle {i}")
                    self.particle_weights[i] = 0.0
                    continue

                ceiling_key = (ceiling_y, ceiling_x)
                if ceiling_key not in self.grid_map['ceiling_images']:
                    logging.debug(f"No ceiling images for key {ceiling_key}")
                    self.particle_weights[i] = 0.0
                    continue

                # Process images for valid particles
                similarities = []
                ceiling_images = self.grid_map['ceiling_images'][ceiling_key]

                # Debug visualization of actual image
                

                
                # Compare with each orientation
                for orientation, coord in ceiling_images.items():
                    image_path = os.path.join('ceiling_images', 
                                            f'{ceiling_key[0]}_{ceiling_key[1]}_{orientation}.jpg')
                    
                    if not os.path.exists(image_path):
                        logging.warning(f"Image not found: {image_path}")
                        continue

                    expected_image = cv2.imread(image_path)
                    if expected_image is not None:
                        # Debug visualization of expected image

            
                        
                        similarity = self.image_similarity(expected_image, actual_image)
                        similarities.append(similarity)
                        logging.debug(
                            f"Similarity for orientation {orientation}: {similarity}")
                    else:
                        logging.warning(f"Failed to load image: {image_path}")

                # Update weight based on best match
                if similarities:
                    self.particle_weights[i] = max(similarities)
                    logging.debug(
                        f"Particle {i} weight: {self.particle_weights[i]}")
                else:
                    self.particle_weights[i] = 0.0
                    logging.debug(f"No valid similarities for particle {i}")

            except Exception as e:
                logging.error(f"Error processing particle {i}: {e}")
                self.particle_weights[i] = 0.0

        # Normalize weights
        total_weight = np.sum(self.particle_weights)
        if total_weight > 0:
            self.particle_weights /= total_weight
            logging.info(
                f"Weights normalized. Max weight: {np.max(self.particle_weights)}")
        else:
            self.particle_weights = np.ones(
                self.num_particles) / self.num_particles
            logging.warning("All weights were zero, reset to uniform")

        # Log weight statistics
        logging.info(f"Weight stats - Mean: {np.mean(self.particle_weights):.6f}, "
                     f"Std: {np.std(self.particle_weights):.6f}, "
                     f"Max: {np.max(self.particle_weights):.6f}")
  

    def resample(self):
        """
        Resample particles based on their weights.
        """
        indices = np.random.choice(
            range(self.num_particles),
            self.num_particles,
            p=self.particle_weights
        )
        self.particles = self.particles[indices]
        self.particle_weights.fill(1.0 / self.num_particles)

    def estimate_position(self):
        """
        Estimate the robot's position as the weighted average of particle positions.
        """
       
        x_estimate = np.average(self.particles[:, 0], weights=self.particle_weights)
        y_estimate = np.average(self.particles[:, 1], weights=self.particle_weights)
        return x_estimate, y_estimate

    def draw_particles(self, estimated_position=None):
        """
        Visualize particles and estimated position, aligning with the map's orientation and real-world coordinates.
        """
        # Load map image
        map_img = plt.imread('lab_474_rotated_cropped.pgm')

        plt.clf()  # Clear the figure

        # Display the map with the real-world bounds
        plt.imshow(map_img, cmap='gray', origin='lower', extent=[0, -15.4, 0, -12.2])

        # Debug: Check particle values
        print("Particles (original):", self.particles)
        print("Estimated Position (original):", estimated_position)

        # Transform particle coordinates to the map's real-world bounds
        flipped_x = self.particles[:, 0]  # Particle X-coordinates
        flipped_y = self.particles[:, 1]  # Particle Y-coordinates

        # Plot particles
        plt.scatter(
            flipped_x,
            flipped_y,
            s=10,
            c=self.particle_weights,
            cmap='viridis',
            alpha=0.5,
            label='Particles'
        )

        # Plot estimated position
        if estimated_position is not None:
            flipped_estimated_x = estimated_position[0]
            flipped_estimated_y = estimated_position[1]
            plt.scatter(
                flipped_estimated_x,
                flipped_estimated_y,
                c='blue',
                marker='+',
                s=100,
                label='Estimated Position'
            )

        # Set axis limits to match the real-world bounds
        plt.xlim(0, -15.4)
        plt.ylim(0, -12.2)

        # Add labels, title, and grid
        plt.legend()
        plt.xlabel("World Y (meters) -")
        plt.ylabel("World X (meters) - ")
        plt.title("Particle Filter Localization")
        plt.grid(True)

        # Draw and save the visualization
        plt.draw()
        plt.pause(0.01)
        plt.savefig("testing_testing_testing.png")



def load_grid_map(pgm_path):
    """
    Load the grid map and prepare the ceiling images library.
    Assumes ceiling images are stored in 'ceiling_images/y_x_orientation.jpg/png'.
    """
    if not os.path.exists(pgm_path):
        raise FileNotFoundError(f"Map file {pgm_path} not found.")

    # Load the map image
    img = Image.open(pgm_path).convert('L')
    img = np.array(img, dtype=np.uint8)/255.0
    print(np.unique(img))

    THRESHOLD = np.unique(img)[1]
    height, width = img.shape

    grid_map = {
        "map_array": img,
        "height": height,
        "width": width,
        "ceiling_images": {}
    }

    ceiling_images_dir = 'ceiling_images'
    if not os.path.exists(ceiling_images_dir):
        raise FileNotFoundError(
            f"Ceiling images directory '{ceiling_images_dir}' not found.")

    # Regular expression to match the renamed filename pattern
    # Expected format: y_x_orientation.jpg/png
    pattern = re.compile(
        r'^(\d+)_(\d+)_(front|left|back|right)\.(jpg|png)$', re.IGNORECASE)

    for filename in os.listdir(ceiling_images_dir):
        if filename.lower().endswith('.jpg') or filename.lower().endswith('.png'):
            match = pattern.match(filename)
            if match:
                y = int(match.group(1))
                x = int(match.group(2))
                orientation = match.group(3).lower()
                extension = match.group(4).lower()

                # Skip blocked columns
                if x in [4, 5, 6]:
                    print(f"Skipping blocked column for file: {filename}")
                    continue

                # Initialize the dict for (y, x) if not already
                if (y, x) not in grid_map["ceiling_images"]:
                    grid_map["ceiling_images"][(y, x)] = {}

                # Assign the image path to the corresponding orientation
                ceiling_image_path = os.path.join(ceiling_images_dir, filename)
                grid_map["ceiling_images"][(
                    y, x)][orientation] = ceiling_image_path
                print(
                    f"Loaded image for grid ({y}, {x}) - {orientation}: {filename}")
            else:
                print(
                    f"Invalid ceiling image filename format: {filename}. Expected 'y_x_orientation.jpg/png'. Skipping.")
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
        self.previous_imu_yaw = None  # Previous IMU yaw in radians
        self.previous_left_distance = None  # Previous left encoder distance in meters
        self.previous_right_distance = None  # Previous right encoder distance in meters
        self.delta_theta_encoders = 0.0  # Change in orientation from encoders
        self.delta_d = 0.0 
        self.yaw = 0.0  # Current orientation in radians
        self.vx = 0.0  # Linear velocity in m/s
        self.vy = 0.0  # Linear velocity in m/s
        self.omega = 0.0  # Angular velocity in rad/s
        



    def __del__(self):
        """Cleanup when object is deleted"""
        if hasattr(self, 'websocket_client'):
            self.websocket_client.cleanup()
    

        

    def fetch_camera_image(self):
        """
        Fetch the latest camera image from Misty via HTTP.
        """

        misty.EnableCameraService()
        camera_url = f"http://{self.misty_ip}/api/cameras/rgb"
        headers = {"Accept": "image/jpeg"}

        
        try:
            response = requests.get(
                camera_url, headers=headers, stream=True, timeout=5)
            if response.status_code == 200:
                img_array = np.asarray(
                    bytearray(response.content), dtype=np.uint8)
                frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                if frame is not None:
                    # Resize the image if necessary
                    frame = cv2.resize(frame, (720, 1080)) 
                     # (width, height)
                    return frame
                else:
                    print("Error: Failed to decode the image. Frame is None.")
            else:
                print(
                    f"Error fetching frame: {response.status_code} - {response.text}")
        except requests.exceptions.RequestException as e:
            print(f"Request Exception while fetching image: {e}")
        return None

   
    import math
    def extract_movement(self, event_name, data):
        """
        Fuse encoder and IMU data to calculate deltas (dx, dy) and yaw.
        """
        wheel_base = 0.11  # Track width
        interval = 0.1     # Time step

        if event_name == "DriveEncoders":
            left_dist = data.get("leftDistance", 0.0) / 1000.0
            right_dist = data.get("rightDistance", 0.0) / 1000.0
            self.vx = (left_dist + right_dist) / 2.0
            self.omega = (right_dist - left_dist) / wheel_base

        elif event_name == "IMU":
            yaw_velocity = math.radians(data.get("yawVelocity", 0.0))
            self.yaw += yaw_velocity * interval
            self.vx += data.get("xAcceleration", 0.0) * interval

        dx = self.vx * math.cos(self.yaw) * interval
        dy = self.vx * math.sin(self.yaw) * interval
        return dx, dy
   
   
  
    def run(self, step=0):
        global data_queue
        """Main loop to process sensor data and update particle filter."""
        plt.figure(figsize=(8, 8))
        plt.ion()  # Interactive mode on
        
        # Step counters and intervals
        fusion_interval = 0.01  # 10 ms (100 Hz)
        particle_interval = 0.1  # 100 ms (10 Hz)
        plot_interval = 2  # Plot every 5 particle filter steps

        last_fusion_time = time.time()
        last_particle_time = time.time()

        while step < 50:
            with self.lock:
                try:
                    current_time = time.time()

                    # Process sensor fusion (encoders + IMU) at high frequency
                    if current_time - last_fusion_time >= fusion_interval:
                        if not data_queue.empty():
                            data= data_queue.get()
                            
                            event_type = data.get("event_type")
                            message = data.get("message")

                            if event_type == "DriveEncoders":
                                # Update position using encoder data
                                 self.extract_movement(event_type, message)

                            elif event_type == "IMU":
                                # Update orientation and refine velocities using IMU data
                                self.extract_movement(event_type,message)

                        last_fusion_time = current_time

                    # Update particle filter at lower frequency
                    if current_time - last_particle_time >= particle_interval:
                        # Move particles based on the fused dx, dy
                        self.pf.move_particles(self.dx, self.dy,self.yaw)
                        self.dx = 0.0
                        self.dy = 0.0
                        # Fetch latest camera image
                        actual_image = self.fetch_camera_image()
                        if actual_image is not None:
                            # Update weights and resample particles
                            self.pf.update_weights(actual_image)
                            estimated_position = self.pf.estimate_position()
                            self.pf.resample()
                            self.current_position = np.array(estimated_position)

                            # Plot particles at intervals
                            if step % plot_interval == 0:
                                self.pf.draw_particles(estimated_position)

                            step += 1
                            logging.info(f"Step: {step}, Estimated Position: {estimated_position}")
                        else:
                            logging.warning("Actual image is None. Skipping update.")

                        last_particle_time = current_time

                    # Brief pause for plotting
                    plt.pause(0.001)

                except Exception as e:
                    logging.error(f"Error in main loop: {e}")
                    continue


    
      
def on_message(ws, message):
    try:
        event_data = json.loads(message)
    except json.JSONDecodeError:
        logging.warning(f"Received non-JSON message: {message}")
        return  # Early exit since message is not in JSON format

    
    if event_data.get("eventName") == "IMUEvent":
        handle_sensor_data("IMU", event_data)
    elif event_data.get("eventName")== "EncoderEvent":
        handle_sensor_data("DriveEncoders", event_data)
    else:
        logging.warning(f"Unknown event received: {event_data}")


def on_error(ws, error):
    logging.error(f"WebSocket error: {error}")


def on_close(ws, close_status_code, close_msg):
    logging.info("WebSocket connection closed")


def on_open(ws):
    global subscribed  # Use if opting for the global flag approach
    logging.info("WebSocket connection opened")
    if not subscribed:
        ws.send(json.dumps({
            "Operation": "subscribe",
            "Type": "IMU",
            "DebounceMs": 50,
            "EventName": "IMUEvent"
        }))
        ws.send(json.dumps({
            "Operation": "subscribe",
            "Type": "DriveEncoders",
            "DebounceMs": 50,
            "EventName": "EncoderEvent"
        }))
        subscribed = True
        logging.info("Subscribed to IMUEvent and DriveEncoders")
    else:
        logging.info("Already subscribed to events")



def handle_sensor_data(event_type, message):
    """
    Callback to handle incoming sensor data and enqueue it for processing.
    """
    if message is None:
        logging.warning(
            f"No 'message' field in event data for event type: {event_type}")
        return

    else:
        data_dict = {
            "event_type": event_type,
            "message": message
        }
        data_queue.put(data_dict)
        logging.debug(f"Enqueued {event_type} data for processing")




# Entry Point
def main():
    # Configuration
    MAP_IMAGE_PATH = "lab_474_rotated_cropped.pgm"

    # Test coordinate transformations with error handling

    WS_URL = f"ws://{MISTY_IP}/pubsub"

    ws = websocket.WebSocketApp(
        WS_URL, on_message=on_message, on_error=on_error, on_close=on_close
    )
    ws.on_open = on_open
    threading.Thread(target=ws.run_forever).start()

    try:
        localizer = RobotLocalizer(
            map_image_path=MAP_IMAGE_PATH,
            misty_ip=MISTY_IP,
            num_particles=50
        )
    
        
        localizer.run()
    except Exception as e:
        logging.error(f"Error in main loop: {e}")
    
    finally:
         ws.close()

main()