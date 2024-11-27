
import mistyPy 
import requests 
import numpy as np
import cv2
from mistyPy.Robot import Robot
import math
from ultralytics import YOLO


MISTY_IP = "10.5.11.234"

file_path = "images"

# 02945

model = YOLO("yolo_weights/yolo11n.pt")


classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]


def start_skill():
    current_response = misty.Drive(5,5)
    print(current_response)
    print(current_response.status_code)
    print(current_response.json())
    print(current_response.json()["result"])


def fetch_misty_camera_frame():
    """Fetches the latest frame from Misty's RGB camera."""
    url = f"http://{MISTY_IP}/api/cameras/rgb"
    headers = {"Accept": "image/jpeg"}  
    
    try:
        response = requests.get(url, headers=headers, stream=True)
        if response.status_code == 200:
            
            img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            img = cv2.imwrite("images/misty_frame.jpg", frame)
           
            return img
        else:
            print("Error fetching frame:", response.status_code, response.text)
            return None
    except Exception as e:
        print("Error:", e)
        return None

    

def fetch_and_feed_into_yolo():

    frame = "images/misty_frame.jpg"

    
    if frame is not None:
        frame = cv2.imread(frame)

        results = model(frame,stream = False)
        # coordinates
        for r in results:
            boxes = r.boxes

            for box in boxes:
                # Bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # Convert to int

                # Draw the bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)

                # Confidence
                confidence = math.ceil((box.conf[0] * 100)) / 100
                print("Confidence --->", confidence)

                # Class name
                cls = int(box.cls[0])
                print("Class name -->", classNames[cls])

                # Display label on the bounding box
                label = f"{classNames[cls]} {confidence:.2f}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                label_x = x1
                label_y = y1 - 10 if y1 - 10 > 10 else y1 + 20 
                org = (x1, y1 - 10)  # Position for text (above the box)
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 0.6
                color = (255, 0, 0)
                thickness = 2

                cv2.rectangle(
                    frame,
                    (label_x, label_y - label_size[1]),
                    (label_x + label_size[0], label_y + 5),
                    (0, 0, 0),  # Black background
                    -1,  # Filled rectangle
                )

                cv2.putText(
                    frame, label, (label_x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
                )  # White text with thickness 2


        # Display the result
        cv2.imshow('YOLO Detection Result', frame)
        cv2.waitKey(0)  
        cv2.destroyAllWindows()
        
        


    



if __name__ == "__main__":
    ipAddress = "10.5.11.234"
    misty = Robot(ipAddress)
    # fetch_misty_camera_frame()
    fetch_and_feed_into_yolo()
    
