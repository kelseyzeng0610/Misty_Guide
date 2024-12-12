from mistyPy.Robot import Robot
from mistyPy.Events import Events
import time

MISTY_IP = "10.5.9.252"
misty = Robot(MISTY_IP)

# Global variables for tracking sensor data
current_tof_distance = 0.0
current_v = 0.0
current_omega = 0.0

# IMU event callback
# def imu_callback(event_data):
#     global current_v, current_omega
#     msg = event_data["message"]
#     yaw_vel = msg.get("yawVelocity", 0.0)  # Angular velocity (degrees/s)
#     x_acc = msg.get("xAcceleration", 0.0)  # Linear acceleration (m/s^2)

#     # Process data (e.g., integration or printing)
#     print(f"IMUEvent triggered. Yaw Vel: {yaw_vel} deg/s, X Acc: {x_acc} m/s^2")

# ToF event callback
def tof_callback(event_data):
    # global current_tof_distance
    # msg = event_data["message"]
    # distance = msg.get("distanceInMeters", None)
    # if distance is not None:
    #     current_tof_distance = distance
    #     print(f"ToFEvent triggered. Distance: {current_tof_distance} meters")
    print(event_data["message"])

def imu_callback(event_data):
    print(event_data["message"])


if __name__ == "__main__":
    try:
        # First create the robot object
        ip_address ="10.5.9.252"
        misty = Robot(ip_address)

        # Register the event, which has a minimum of 2 parameters: the user defined name of the event, and the event type
        misty.RegisterEvent(Events.IMU, "IMU", callback_function=imu_callback)

        # Start recording speech to get an event message
        misty.DriveTime(30, 30, 2000)

        # Use the keep_alive function to keep running the main python thread until all events are closed, or the script is killed due to an exception
        misty.KeepAlive()

    except Exception as ex:
        print(ex)
    finally:
        # Unregister events if they aren't all unregistered due to an error
        misty.UnregisterAllEvents()



# MISTY_IP = "10.5.11.234"

# misty = Robot(MISTY_IP)

# CURRENTMAP = misty.GetMap()


# map_info = misty.GetMap().json().get("result")
# if map_info:
#     # Debugging: Print entire map_info
    
#     # print("Map Information:", json.dumps(map_info, indent=4))
#     # Extract the grid and some metadata
#     grid = map_info.get("grid")
#     height = map_info.get("height")
#     width = map_info.get("width")
#     meters_per_cell = map_info.get("metersPerCell")
#     origin_x = map_info.get("originX")
#     origin_y = map_info.get("originY")
    
#     try:
#         grid_array = np.array(grid, dtype=np.float32)
#         print("Grid Array Shape:", grid_array.shape)
#         print("Grid Array Dtype:", grid_array.dtype)
#         plt.imshow(grid_array, cmap='gray', origin='lower')
#         plt.title("Misty's SLAM Map")
#         plt.colorbar(label='Occupancy Value')
#         plt.savefig("map.png")
#     except Exception as e:
#         print("Error converting grid to NumPy array:", e)
#         grid_array = None


# endpoint = f"http://{MISTY_IP}/api/drive/path"
# path= {
#     "Path": "110:150",
#     "Velocity": 0.5,          # Optional: Move at 50% of max speed
#     "FullSpinDuration": 15,   # Optional: Full spin duration in seconds
#     "WaypointAccuracy": 0.1,  # Optional: How close Misty must get to each waypoint (meters)
#     "RotateThreshold": 10     # Optional: Angle in degrees before Misty pivots toward next waypoint
# }


# endpoint_track =  f"http://{MISTY_IP}/api/slam/track/start"
# response1 = requests.post(endpoint_track)
# print("Status Code:", response1.status_code)
# print("Response:", response1.text)

# time.sleep(5)
# # Call FollowPath
# response = requests.post(endpoint, json=path)
# print("Status Code:", response.status_code)
# print("Response:", response.text)

# time.sleep(10)


# misty.DriveTime(1,4,2000)
# # When done, stop tracking
