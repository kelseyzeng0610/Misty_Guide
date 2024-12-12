import yaml
import numpy as np
from PIL import Image
import math
import time
import websocket
import json
import random
import cv2
import os
import requests
import matplotlib.pyplot as plt
from collections import deque
# from pydstarlite import DStarLite

from mistyPy.Robot import Robot
from mistyPy.Events import Events
import os 
import threading 

ORIGIN = np.array([-15.4, -12.2, 0.0])  # Real-world origin at the door
RESOLUTION = 0.05  # meters per pixel
CEILING_GRID_SIZE = 6

LOOP_INTERVAL = 0.5

occupied_threshold = 0.65
free_threshold = 0.196

current_image = None
wheel_base = 0.11  

def world_to_grid(world_x, world_y):
    grid_x = int((world_x - ORIGIN[0]) / RESOLUTION)
    grid_y = int((world_y - ORIGIN[1]) / RESOLUTION)
    return grid_x, grid_y
def grid_to_world(grid_x, grid_y):
    world_x = grid_x * RESOLUTION + ORIGIN[0]
    world_y = grid_y * RESOLUTION + ORIGIN[1]
    return world_x, world_y


def grid_to_ceiling(grid_x, grid_y):
    """
    Convert grid coordinates to ceiling map coordinates.
    """
    ceiling_x = grid_x // 6
    ceiling_y = grid_y // 6
    return ceiling_x, ceiling_y
    


class ParticleFilter:
    def __init__(self, num_particles, grid_map, origin, resolution, ceiling_grid_size=6):
        self.num_particles = num_particles
        self.grid_map = grid_map  # Dictionary with grid cell as key and image path as value
        self.origin = origin
        self.resolution = resolution
        self.ceiling_grid_size = ceiling_grid_size
        self.particles = self.initialize_particles()
        self.particle_weights = np.ones(self.num_particles) / self.num_particles
    def initialize_particles(self):
        particles = []
        for _ in range(self.num_particles):
            px, py = self.grid_map.random_free_cell()
            
            particles.append([px, py])
        return np.array(particles, dtype=float)
    
    def move_particles(self,dx,dy,sigma=0.1):
        noise = random.normal(0,sigma,self.num_particles.shape)
        self.particles += np.array([dx,dy]) + noise

    def image_similarity(self,expected_image,actual):
        measurement_hsv = cv2.cvtColor(actual, cv2.COLOR_RGB2HSV)
        expected_hsv = cv2.cvtColor(expected_image, cv2.COLOR_RGB2HSV)
        
        
        measurement_hist = cv2.calcHist([measurement_hsv], [0], None, [50], [0, 180]) 
        expected_hist = cv2.calcHist([expected_hsv], [0], None, [50], [0, 180])
        
        cv2.normalize(measurement_hist, measurement_hist)
        cv2.normalize(expected_hist, expected_hist)
        
        # Compute similarity using Bhattacharyya distance
        similarity = 1 - cv2.compareHist(measurement_hist, expected_hist, cv2.HISTCMP_BHATTACHARYYA)

   
        return max(0, similarity)
    def update_weights(self, actual_image):
        for i, particle in enumerate(self.particles):
            grid_x, grid_y = world_to_grid(particle[0], particle[1])
            ceiling_x, ceiling_y = grid_to_ceiling(grid_x, grid_y)
            ceiling_key = (ceiling_x, ceiling_y)

            if ceiling_key in self.grid_map['ceiling_images']:
                expected_image_path = self.grid_map['ceiling_images'][ceiling_key]
                expected_image = cv2.imread(expected_image_path)
                if expected_image is None:
                    self.particle_weights[i] = 0.0
                    continue
                similarity = self.image_similarity(expected_image, actual_image)
                self.particle_weights[i] = similarity
            else:
                # Assign a minimal weight if no corresponding ceiling image
                self.particle_weights[i] = 0.0

        # Normalize weights
        total_weight = np.sum(self.particle_weights)
        if total_weight == 0:
            # Avoid division by zero; assign uniform weights
            self.particle_weights = np.ones(self.num_particles) / self.num_particles
        else:
            # Add a small value to avoid zero weights
            self.particle_weights += 1e-6
            self.particle_weights /= np.sum(self.particle_weights)

    

    


    def estimate_positions(self):
        x_estimate = np.average(self.particles[:,0], weights=self.particle_weights)
        y_estimate = np.average(self.particles[:,1], weights=self.particle_weights)
        return x_estimate, y_estimate

    def resample(self):
        indices = np.random.choice(range(self.num_particles), self.num_particles, p=self.particle_weights)
        self.particles = self.particles[indices]
        self.particle_weights.fill(1.0 / self.num_particles)
    
    
    def draw_particles(self, estimated_position=None):
        """
        Visualize particles, true position, and estimated position.
        """
        plt.clf()
        # Plot particles
        plt.scatter(self.particles[:, 0], self.particles[:, 1], 
                    s=10, c=self.particle_weights, cmap='viridis', alpha=0.5, label='Particles')
        # Plot true position
        
        # Plot estimated position
        if estimated_position is not None:
            plt.scatter(estimated_position[0], estimated_position[1], c='blue', marker='+', label='Estimated Position')
        plt.legend()
        plt.xlabel("Grid X")
        plt.ylabel("Grid Y")
        plt.title("Particle Filter Localization")
        plt.grid(True)
        plt.pause(0.01)






class MistyWebSocketClient:
    def __init__(self, ip, on_message):
        self.ip = "10.5.11.234"
        self.websocket_uri = f"ws://{self.ip}/pubsub"
        self.ws = None
        self.ws.on_open = self.on_open
        self.queue = deque()

    def on_message(self,ws, message):
        event_data = json.loads(message)
        if event_data.get("eventName") == "IMUEvent":
            process_imu(event_data["message"])
        elif event_data.get("eventName") == "DriveEncoders":
            process_drive_encoders(event_data["message"])
            
    def on_open(self, ws):
        imu_subscription = {"Operation": "subscribe", "Type": "IMU", "DebounceMs": 200, "EventName": "IMUEvent"}
        encoders_subscription = {"Operation": "subscribe", "Type": "DriveEncoders", "DebounceMs": 200, "EventName": "DriveEncoders"}
        ws.send(json.dumps(imu_subscription))
        ws.send(json.dumps(encoders_subscription))
        print("Subscribed to IMU and DriveEncoders events.")
    def on_error(self,ws, error):
        print("Error:", error)
    def on_close(self,ws, close_status_code, close_msg):
        print("WebSocket closed.", close_msg)

    def run_forever(self):
        self.ws = websocket.WebSocketApp(self.websocket_uri, on_message=self.on_message, on_error=self.on_error,on_close=self.on_close)
        self.ws.on_open = self.on_open
        self.ws.run_forever().start()

class GridMap:

    def __init__(self,pgm_path):
        self.grid_map = self.load_grid_map(pgm_path)
        self.origin = ORIGIN
        self.resolution = RESOLUTION


    def load_grid_map(pgm_path):
        img = Image.open(pgm_path)
        img = np.array(img, dtype=np.uint8) / 255.0
        height,width,_ = img.shape

        grid_map = {
            "map_array": img,
            "height": height,
            "width": width,
            "ceilings_images":{}
        }

        ceiling_images_dir = 'ceiling_images'
        for filename in os.listdir(ceiling_images_dir):
            if filename.endswith('.jpg'):
                ceiling_x, ceiling_y = map(int, filename[:-4].split('_'))
                ceiling_image_path = os.path.join(ceiling_images_dir, filename)
                grid_map["ceiling_images"][(ceiling_x, ceiling_y)] = ceiling_image_path
        return grid_map

    def random_free_cell(self):
        while True:
            x = random.randint(0, self.grid_map["width"] - 1)
            y = random.randint(0, self.grid_map["height"] - 1)
            if (self.grid_map["map_array"][y, x]) /255.0 < occupied_threshold:
                return x, y
        return None


class RobotLocalizer:
    def __init__(self, map_image_path, websocket_uri, num_particles=300):
        self.grid_map = GridMap(map_image_path)
        self.origin = ORIGIN
        self.resolution = RESOLUTION
        self.pf = ParticleFilter(num_particles, self.grid_map, self.origin, self.resolution, CEILING_GRID_SIZE)
        self.data_queue = deque()
        self.websocket = MistyWebSocketClient(websocket_uri, self.on_message)
        self.current_position = np.array([0.0, 0.0])
        self.lock = threading.Lock()

    

    def start_websocket(self):
        self.websocket.run_forever()


        
    

    def fetch_camera(self):
        global current_image
        url = f"http://{MISTY_IP}/api/cameras/rgb"
        headers = {"Accept": "image/jpeg"}
        try:
            response = requests.get(url, headers=headers, stream=True)
            if response.status_code == 200:
                img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
                frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                if frame is not None:
                    # Resize the image to the expected dimensions
                    frame = cv2.resize(frame, (720, 1080))  # Ensure correct dimensions (width x height)
                    current_camera_image = frame
                    return frame
                else:
                    print("Error: Failed to decode the image. Frame is None.")
                    current_camera_image = None
            else:
                print("Error fetching frame:", response.status_code, response.text)
                current_image = None
        except Exception as e:
            print("Error:", e)
            current_image = None
        
        
    
    def run(self):
        plt.figure(figsize=(8,8))
        plt.ion()

        while True:
            with self.lock:
            if self.data_queue:
                data = self.data_queue.popleft()
                imu_data = data.get('IMU',{})
                encoders = 
                camera_image = data.get('CameraImage',None)

                dx,dy = self.extract_movement(encoders)
                self.pf.move_particles(dx,dy)

                actual_image = self.fetch_camera()

                if actual_image is not None:
                    self.pf.update_weights(actual_image)
                    self.pf.resample()
                    estimated_position = self.pf.estimate_positions()
                self.pf.draw_particles(estimated_position)
            plt.pause(0.01)
    
    def extract_movment(self,encoders):
        
        left_velocity_mm_s = encoders.get("leftVelocity", 0.0)
        right_velocity_mm_s = encoders.get("rightVelocity", 0.0)
        left_v = left_velocity_mm_s / 1000.0
        right_v = right_velocity_mm_s / 1000.0

        v = (left_v + right_v) / 2.0 
        omega = (right_v - left_v) / wheel_base
        
        dt = LOOP_INTERVAL
        dx = v*math.cos(omega*dt)*dt
        dy = v*math.sin(omega*dt)*dt
        return dx,dy
    
        
    




class ParticleFilter:

    def __init__(self, grid_world, num_particles=100):
        self.grid = grid_world
        self.num_particles = num_particles
        self.particles = []
        self.weights = np.ones(num_particles) / num_particles
        self.initialize_particles()
    

    
            

    def initialize_particles(self):
        for _ in range(self.num_particles):
            px, py = self.grid.random_free_cell()
            theta = random.uniform(-math.pi, math.pi)
            self.particles.append([px, py, theta])
        self.particles = np.array(self.particles, dtype=float)
    


    def predict(self,dx,dy,dtheta):
        sigma_motion = 0.2
        for i in range(self.num_particles):
            px, py, theta = self.particles[i]
            gdx = dx + np.random.normal(0, sigma_motion)
            gdy = dy + np.random.normal(0, sigma_motion)
            gdtheta = dtheta + np.random.normal(0, sigma_motion)
            new_px = px + gdx * math.cos(theta) - gdy * math.sin(theta)
            new_py = py + gdx * math.sin(theta) + gdy * math.cos(theta)
            new_theta = (theta + gdtheta + math.pi) % (2 * math.pi) - math.pi
            if self.grid.is_free(int(new_px), int(new_py)):
                self.particles[i] = [new_px, new_py, new_theta]
    

    


def process_imu(message):
    global imu_v, imu_omega, last_time
    now = time.time()
    dt = now - last_time
    if dt <= 0:
        dt = LOOP_INTERVAL
    last_time = now
    imu_v += message.get("xAcceleration", 0.0) * dt
    imu_omega = math.radians(message.get("yawVelocity", 0.0))


wheel_base = 0.11  # Wheel base (track width) in meters
def process_drive_encoders(message):
    global encoder_v, encoder_omega
    left_velocity_mm_s = message.get("leftVelocity", 0.0)
    right_velocity_mm_s = message.get("rightVelocity", 0.0)
    left_v = left_velocity_mm_s / 1000.0
    right_v = right_velocity_mm_s / 1000.0
    
    encoder_v = (left_v + right_v) / 2.0
    encoder_omega = (right_v - left_v) / wheel_base
    return encoder_v, encoder_omega





if __name__ == "__main__":
    import os
    import base64
    import io

    MISTY_IP = "10.5.11.234"
    WS_URL = f"ws://{MISTY_IP}/pubsub"

    localizer = RobotLocalizer("rotate_lab_474.pgm", WS_URL)
    localizer.start_websocket()

    try:
        localizer.run()
    except KeyboardInterrupt:
        print("Stopping the program...")



