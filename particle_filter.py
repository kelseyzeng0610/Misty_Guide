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
# from pydstarlite import DStarLite

from mistyPy.Robot import Robot
from mistyPy.Events import Events
import os 
import threading 




"""
grid = Grid.from_matrix(GRID_MAP)

start = (0, 0)
goal_pos = (5, 5)


dstar.set_start(start_pos)
dstar.set_goal(goal_pos)

# Compute the initial path
path = dstar.compute_shortest_path()

# Check if a path exists
if path is None:
    logging.error("No path found from start to goal.")
else:
    logging.info(f"Path found: {path}")




# Define possible orientations
ORIENTATIONS = ['N', 'E', 'S', 'W']  # Clockwise
ORIENTATION_DEGREES = {'N': 0, 'E': 90, 'S': 180, 'W': 270}
"""
# def calculate_turn(current_orientation, desired_orientation):
#     """
#     Calculates the action needed to turn from current to desired orientation.
#     Since only left turns are allowed, determine the number of left turns required.
#     """
#     current_idx = ORIENTATIONS.index(current_orientation)
#     desired_idx = ORIENTATIONS.index(desired_orientation)
#     turns = (current_idx - desired_idx) % len(ORIENTATIONS)
#     return 'left' * turns  # Returns a string like 'leftleft' for two left turns

# def path_to_actions(path, initial_orientation='E'):
#     """
#     Converts a list of grid positions to movement actions based on orientation.
    
#     :param path: List of tuples representing grid positions [(x1, y1), (x2, y2), ...]
#     :param initial_orientation: Starting orientation ('N', 'E', 'S', 'W')
#     :return: List of actions ['forward', 'left', ...]
#     """
#     actions = []
#     current_orientation = initial_orientation
    
#     for i in range(1, len(path)):
#         current_pos = path[i-1]
#         next_pos = path[i]
        
#         dx = next_pos[0] - current_pos[0]
#         dy = next_pos[1] - current_pos[1]
        
#         # Determine desired orientation based on movement
#         if dx == 1 and dy == 0:
#             desired_orientation = 'E'
#         elif dx == -1 and dy == 0:
#             desired_orientation = 'W'
#         elif dx == 0 and dy == 1:
#             desired_orientation = 'S'
#         elif dx == 0 and dy == -1:
#             desired_orientation = 'N'
#         else:
#             logging.error(f"Invalid movement from {current_pos} to {next_pos}")
#             continue  # Skip invalid movements
        
#         # Calculate required turns (only left turns are allowed)
#         while current_orientation != desired_orientation:
#             actions.append('left')
#             current_orientation = ORIENTATIONS[(ORIENTATIONS.index(current_orientation) - 1) % len(ORIENTATIONS)]
        
#         # Move forward
#         actions.append('forward')
    
#     return actions

# # Convert path to actions
# actions = path_to_actions(path, initial_orientation='E')
# logging.info(f"Action sequence: {actions}")


# Robot and Map Configuration
MISTY_IP = "10.5.11.234"
misty = Robot(MISTY_IP)
WS_URL = f"ws://{MISTY_IP}/pubsub"

FRONT_TOF_SENSOR_ID = "toffc"
LOOP_INTERVAL = 0.1

# Global Variables
fused_omega = 0.0
fused_v = 0.0
imu_v = 0.0
imu_omega = 0.0
encoder_v = 0.0
encoder_omega = 0.0
last_time = time.time()
current_camera_image = None

# Map Parameters
origin = [-15.4, -12.2, 0.0]  # Map origin in world coordinates
resolution = 0.05  # Map resolution (meters per pixel)
occupied_thresh = 0.65
free_thresh = 0.196
wheel_base = 0.11  # Wheel base (track width) in meters

### WebSocket Callbacks ###
def process_drive_encoders(message):
    global encoder_v, encoder_omega
    left_velocity_mm_s = message.get("leftVelocity", 0.0)
    right_velocity_mm_s = message.get("rightVelocity", 0.0)
    left_v = left_velocity_mm_s / 1000.0
    right_v = right_velocity_mm_s / 1000.0
    encoder_v = (left_v + right_v) / 2.0
    encoder_omega = (right_v - left_v) / wheel_base

def process_imu(message):
    global imu_v, imu_omega, last_time
    now = time.time()
    dt = now - last_time
    if dt <= 0:
        dt = LOOP_INTERVAL
    last_time = now
    imu_v += message.get("xAcceleration", 0.0) * dt
    imu_omega = math.radians(message.get("yawVelocity", 0.0))

def on_message(ws, message):
    event_data = json.loads(message)
    if event_data.get("eventName") == "IMUEvent":
        process_imu(event_data["message"])
    elif event_data.get("eventName") == "DriveEncoders":
        process_drive_encoders(event_data["message"])

def on_open(ws):
    imu_subscription = {"Operation": "subscribe", "Type": "IMU", "DebounceMs": 200, "EventName": "IMUEvent"}
    encoders_subscription = {"Operation": "subscribe", "Type": "DriveEncoders", "DebounceMs": 200, "EventName": "DriveEncoders"}
    ws.send(json.dumps(imu_subscription))
    ws.send(json.dumps(encoders_subscription))
    print("Subscribed to IMU and DriveEncoders events.")

def on_error(ws, error):
    print("Error:", error)

def on_close(ws, close_status_code, close_msg):
    print("WebSocket closed.", close_msg)

### Map and GridWorld ###
def load_map(pgm_path):
    img = Image.open(pgm_path)
    return np.array(img, dtype=np.uint8)

class GridWorld:
    def __init__(self, map_data, origin, resolution, occupied_threshold, free_threshold):
        self.map = map_data
        self.origin = origin
        self.resolution = resolution
        self.occupied_threshold = occupied_threshold
        self.free_threshold = free_threshold

        # Calculate grid dimensions
        self.height, self.width = self.map.shape
        self.grid_rows = int(self.height * self.resolution)  # Number of rows in the grid
        self.grid_cols = int(self.width * self.resolution)  # Number of columns in the grid

        self.free_cells = self._identify_free_cells()

    def _identify_free_cells(self):
        """
        Precompute a list of free cells in the grid.
        :return: List of tuples representing free cell indices (px, py).
        """
        free_cells = []
        for y in range(self.grid_rows):
            for x in range(self.grid_cols):
                if self.is_free(x, y):
                    free_cells.append((x, y))
        return free_cells

    def is_free(self, grid_x, grid_y):
        """
        Check if a grid cell is free.
        :param grid_x: Grid column index.
        :param grid_y: Grid row index.
        :return: True if the cell is free, False otherwise.
        """
        if 0 <= grid_x < self.grid_cols and 0 <= grid_y < self.grid_rows:
            world_x, world_y = self.grid_to_world(grid_x, grid_y)
            px, py = self.world_to_image(world_x, world_y)
            cell_value = self.map[py, px] / 255.0  # Normalize pixel values
            return cell_value <= self.free_threshold
        return False

    def world_to_grid(self, world_x, world_y):
        """
        Convert world coordinates to grid coordinates.
        :param world_x: X-coordinate in world space.
        :param world_y: Y-coordinate in world space.
        :return: Tuple (grid_x, grid_y).
        """
        grid_x = int((world_x - self.origin[0]) / self.resolution)
        grid_y = int((world_y - self.origin[1]) / self.resolution)
        return grid_x, grid_y

    def grid_to_world(self, grid_x, grid_y):
        """
        Convert grid coordinates to world coordinates.
        :param grid_x: X-coordinate in grid space.
        :param grid_y: Y-coordinate in grid space.
        :return: Tuple (world_x, world_y).
        """
        world_x = grid_x * self.resolution + self.origin[0]
        world_y = grid_y * self.resolution + self.origin[1]
        return world_x, world_y

    def world_to_image(self, world_x, world_y):
        """
        Convert world coordinates to image pixel coordinates.
        :param world_x: X-coordinate in world space.
        :param world_y: Y-coordinate in world space.
        :return: Tuple (image_x, image_y).
        """
        image_x = int((world_x - self.origin[0]) / self.resolution * self.width / self.grid_cols)
        image_y = int((world_y - self.origin[1]) / self.resolution * self.height / self.grid_rows)
        return image_x, image_y

    def random_free_cell(self):
        """
        Select a random free cell from the precomputed free cells.
        :return: Tuple (grid_x, grid_y).
        """
        if not self.free_cells:
            raise ValueError("No free cells available in the map.")
        return random.choice(self.free_cells)

def measurement_func(px, py, ceiling_map, grid_world):
    """
    Compute the weight of a particle by comparing its expected measurement with the actual image.
    """
    global current_camera_image

    # Map GridWorld coordinates to CeilingMap grid
    norm_x, norm_y = normalize_to_ceiling_grid(px, py, ceiling_map.grid_width, ceiling_map.grid_height, grid_world)

    # Retrieve the expected image from the CeilingMap
    expected_img = ceiling_map.get_expected_image(norm_x, norm_y)

    if expected_img is None:  # Blocked column or missing image
        return 0.0

    # Compare the expected image with the current camera image
    img_likelihood = image_similarity(current_camera_image, expected_img) if current_camera_image is not None else 0.5

    return img_likelihood


### CeilingMap ###
class CeilingMap:
    def __init__(self, image_dir, grid_width, grid_height, blocked_columns, standard_size=(1080, 720)):
        """
        Initialize the CeilingMap, skipping blocked columns and associating images with grid coordinates.
        """
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.blocked_columns = blocked_columns
        self.image_db = {}
        self.standard_size = standard_size

        image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".jpg")])
        img_index = 0

        for gx in range(grid_width):
            if gx in blocked_columns:
                continue  # Skip blocked columns

            for gy in range(grid_height):
                if img_index >= len(image_files):
                    print("Warning: Not enough images to populate the grid.")
                    break

                img_path = os.path.join(image_dir, image_files[img_index])
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, self.standard_size, interpolation=cv2.INTER_AREA)
                    self.image_db[(gx, gy)] = img
                else:
                    print(f"Failed to load image: {img_path}")
                img_index += 1

    def get_expected_image(self, gx, gy):
        """
        Retrieve the expected image for a CeilingMap grid coordinate.
        If the column is blocked, return None.
        """
        return self.image_db.get((gx, gy), None)



def capture_image():
    global current_camera_image
    url = f"http://{MISTY_IP}/api/cameras/rgb"
    headers = {"Accept": "image/jpeg"}
    image_dir = "captured_images"
    os.makedirs(image_dir,exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    filemame = os.path.join(image_dir, f"h_{timestamp}.jpg")

    try:
        response = requests.get(url, headers=headers, stream=True)
        if response.status_code == 200:
            img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if frame is not None:
                # Resize the image to the expected dimensions
                frame = cv2.resize(frame, (720, 1080)) 
                cv2.imwrite(filename,frame) # Ensure correct dimensions (width x height)
                current_camera_image = frame
            else:
                print("Error: Failed to decode the image. Frame is None.")
                current_camera_image = None
        else:
            print("Error fetching frame:", response.status_code, response.text)
            current_camera_image = None
    except Exception as e:
        print("Error:", e)
        current_camera_image = None

### Particle Filter ###
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

    def predict(self, dx, dy, dtheta):
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

    def update_weights(self, measurement_func, ceiling_map, grid_world):
        for i, (px, py, _) in enumerate(self.particles):
            self.weights[i] = measurement_func(px, py, ceiling_map, grid_world)
        self.weights += 1e-6  # Avoid division by zero
        self.weights /= np.sum(self.weights)


    def resample(self):
        indices = np.random.choice(range(self.num_particles), self.num_particles, p=self.weights)
        self.particles = self.particles[indices]
        self.weights.fill(1.0 / self.num_particles)

    def estimate_pose(self):
        """
        Estimate the robot's pose (x, y, theta) based on the weighted average of particles.
        :return: Tuple (world_x, world_y, theta) representing the estimated pose.
        """
        x, y, sin_theta, cos_theta = 0, 0, 0, 0
        for (px, py, theta), weight in zip(self.particles, self.weights):
            x += px * weight
            y += py * weight
            sin_theta += math.sin(theta) * weight
            cos_theta += math.cos(theta) * weight

        # Convert to world coordinates
        grid_x = x
        grid_y = y
        world_x, world_y = self.grid.grid_to_world(grid_x, grid_y)

        # Calculate average orientation
        theta = math.atan2(sin_theta, cos_theta)
        return world_x, world_y, theta

    
def image_similarity(img1, img2):
    if img1 is None or img2 is None:
        return 0.0
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    img1_hsv = cv2.cvtColor(img1, cv2.COLOR_RGB2HSV)
    img2_hsv = cv2.cvtColor(img2, cv2.COLOR_RGB2HSV)

    hist1 = cv2.calcHist([img1_hsv],[0],None,[50],[0,180])
    hist2 = cv2.calcHist([img2_hsv],[0],None,[50],[0,180])
    cv2.normalize(hist1,hist1)
    cv2.normalize(hist2,hist2)

    similarity = 1 - cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)
    return max(similarity, 0.0)

# def measurement_func(px, py, ceiling_map):
#     global current_camera_image


#     expected_img = ceiling_map.get_expected_image(px, py)
#     img_likelihood = image_similarity(current_camera_image, expected_img) if current_camera_image is not None else 0.5

#     weight = img_likelihood
#     return weight

### Main Loop ###
def main_loop(pf, grid, ceiling_map, map_data, step, trajectory):
    global fused_v,fused_omega, imu_v, imu_omega, encoder_v, encoder_omega
    capture_image()
    dt = time.time() - last_time
    fused_v = 0.7 * encoder_v + 0.3 * imu_v
    fused_omega = 0.7 * encoder_omega + 0.3 * imu_omega

    dx = fused_v * dt
    dtheta = fused_omega * dt
    pf.predict(dx, 0.0, dtheta)
    pf.update_weights(measurement_func, ceiling_map, grid)  # Pass ceiling_map and grid_world
    pf.resample()

    wx, wy, theta = pf.estimate_pose()
    print(f"Estimated Position: ({wx:.2f}, {wy:.2f}), Orientation: {math.degrees(theta):.2f}°")
    trajectory.append((wx, wy))

    # visualize_localization(grid.map, pf, (wx, wy), step, trajectory)


    import matplotlib.pyplot as plt
def normalize_to_ceiling_grid(px, py, grid_width, grid_height, grid_world):
    """
    Normalize GridWorld (fine resolution) grid coordinates to CeilingMap's fixed 6x6 grid.
    :param px: GridWorld x-coordinate.
    :param py: GridWorld y-coordinate.
    :param grid_width: CeilingMap grid width (6).
    :param grid_height: CeilingMap grid height (6).
    :param grid_world: GridWorld object for reference.
    :return: Normalized coordinates (gx, gy) for the CeilingMap.
    """
    # Convert GridWorld (grid_x, grid_y) to world coordinates
    world_x, world_y = grid_world.grid_to_world(px, py)

    # Map world coordinates to CeilingMap grid
    norm_x = int((world_x - origin[0]) / ((grid_world.width * grid_world.resolution) / grid_width))
    norm_y = int((world_y - origin[1]) / ((grid_world.height * grid_world.resolution) / grid_height))

    # Ensure normalized values fall within valid range
    norm_x = max(0, min(grid_width - 1, norm_x))
    norm_y = max(0, min(grid_height - 1, norm_y))

    return norm_x, norm_y



if __name__ == "__main__":
    # Load the map and initialize GridWorld
   
    # Initialize CeilingMap
    image_dir = "ceiling_images"
    grid_width = 6
    grid_height = 6
    blocked_columns = [0, 1]


    

    # Parameters from particle_filter2.py
    origin = [-15.4, -12.2, 0.0]  # Map origin in world coordinates
    resolution = 0.05  # Map resolution (meters per pixel)
    grid_width = 6
    grid_height = 6
    blocked_columns = [4,5]

    map_data = load_map("lab_474.pgm")
    grid = GridWorld(
        map_data=map_data,
        origin=origin,
        resolution=resolution,
        occupied_threshold=occupied_thresh,
        free_threshold=free_thresh
    )

    # Initialize CeilingMap
    image_dir = "ceiling_images"
    ceiling_map = CeilingMap(
        image_dir=image_dir,
        grid_width=6,
        grid_height=6,
        blocked_columns=[0, 1]
    )

    def print_coordinate_mappings(grid_world, ceiling_map):
        """
        Print mappings between GridWorld and CeilingMap for verification.
        """
        print("\nGrid to CeilingMap Coordinate Mappings:")
        print(f"{'GridWorld (x,y)':<20} {'CeilingMap (gx,gy)':<20} {'Status':<10}")
        print("-" * 50)

        for py in range(grid_world.grid_rows):
            for px in range(grid_world.grid_cols):
                norm_x, norm_y = normalize_to_ceiling_grid(px, py, ceiling_map.grid_width, ceiling_map.grid_height, grid_world)
                status = "BLOCKED" if norm_x in ceiling_map.blocked_columns else "FREE"
                print(f"({px:2d},{py:2d}){' '*10} ({norm_x:2d},{norm_y:2d}){' '*10} {status}")


    print("\nGrid System Information:")
    print(f"Origin: {origin[:2]}")
    print(f"Resolution: {resolution} meters/cell")
    print(f"Grid Size: {grid_width}x{grid_height}")
    print(f"Blocked Columns: {blocked_columns}")

    print_coordinate_mappings(grid, ceiling_map)

    

    # Helper function to normalize to ceiling grid
    def normalize_to_ceiling_grid(px, py, grid_width, grid_height, grid_world):
        """
        Normalize GridWorld (dynamic resolution) grid coordinates to CeilingMap's fixed grid (6x6).
        :param px: GridWorld x-coordinate.
        :param py: GridWorld y-coordinate.
        :param grid_width: CeilingMap grid width.
        :param grid_height: CeilingMap grid height.
        :param grid_world: GridWorld object.
        :return: Normalized coordinates for CeilingMap.
        """
        world_x, world_y = grid_world.grid_to_world(px, py)
        norm_x = int((world_x - origin[0]) / (resolution * grid_world.width) * grid_width)
        norm_y = int((world_y - origin[1]) / (resolution * grid_world.height) * grid_height)
        return max(0, min(grid_width - 1, norm_x)), max(0, min(grid_height - 1, norm_y))

    # Initialize ParticleFilter
    pf = ParticleFilter(grid_world=grid, num_particles=100)

    # Initialize trajectory list for visualization
    trajectory = []

    # Set up WebSocket connection
    ws = websocket.WebSocketApp(
        WS_URL, on_message=on_message, on_error=on_error, on_close=on_close
    )
    ws.on_open = on_open
    threading.Thread(target=ws.run_forever).start()

    # Main loop to run the particle filter
    for step in range(100):
        main_loop(pf, grid, ceiling_map, map_data, step, trajectory)
        time.sleep(1)

    print("Localization complete.")

