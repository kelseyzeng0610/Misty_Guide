

# import requests

# def drive_misty():
#     # Misty robot's IP and endpoint
#     robot_ip = "10.5.11.234"
#     drive_url = f"http://{robot_ip}/api/drive"

#     # Drive command payload
#     payload = {
#         "linearVelocity": 5,
#         "angularVelocity": 5,
#     }

#     try:
#         # Send POST request with a 10-second timeout
#         response = requests.post(drive_url, json=payload, timeout=10)
        
#         # Check if the request was successful
#         if response.ok:
#             json_data = response.json()
#             print("Response from Misty:", json_data)
#         else:
#             print("Error:", response.status_code, response.text)
#     except requests.Timeout:
#         print("Request timed out after 10 seconds.")

# if __name__ == "__main__":
#     drive_misty()
# import sys, os
# sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import mistyPy 
from mistyPy.Robot import Robot



def start_skill():
    current_response = misty.Drive(5,5)
    print(current_response)
    print(current_response.status_code)
    print(current_response.json())
    print(current_response.json()["result"])

if __name__ == "__main__":
    ipAddress = "10.5.11.234"
    misty = Robot(ipAddress)
    start_skill()