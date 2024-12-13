
import mistyPy
import os
import time
import requests
import numpy as np
import cv2
import base64
from mistyPy.Robot import Robot
import math
from ultralytics import YOLO

import webcolors
from gtts import gTTS  # Used for doing text to speech
import nltk
from nltk.corpus import words
nltk.download('words')
word_list = set(words.words())


MISTY_IP = "10.5.9.252"
DEBUG_JSON_REQUESTS = False
misty = Robot(MISTY_IP)

file_path = "images"

# 02945

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


def start_skill():
    current_response = misty.Drive(5, 5)
    print(current_response)
    print(current_response.status_code)
    print(current_response.json())
    print(current_response.json()["result"])


def fetch_misty_camera_frame():
    """Fetches the latest frame from Misty's RGB camera."""
    url = f"http://{MISTY_IP}/api/cameras/rgb"
    headers = {"Accept": "image/jpeg"}

    try:
        misty.EnableCameraService()
        response = requests.get(url, headers=headers, stream=True)
        if response.status_code == 200:

            img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if not os.path.exists("images"):
                os.makedirs("images")
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

        results = model(frame, stream=False)
        # coordinates
        for r in results:
            boxes = r.boxes

            for box in boxes:
                # Bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(
                    x2), int(y2)  # Convert to int

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
                label_size, _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
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


def closest_color(requested_color):
    min_colors = {}
    for name in webcolors.names('css3'):
        r_c, g_c, b_c = webcolors.name_to_rgb(name)
        rd = (r_c - requested_color[0]) ** 2
        gd = (g_c - requested_color[1]) ** 2
        bd = (b_c - requested_color[2]) ** 2
        min_colors[(rd + gd + bd)] = name
    return min_colors[min(min_colors.keys())]


def get_color_name(rgb_tuple):
    # split_result = dictionary_based_split(closest_color(rgb_tuple))
    # return split_result
    return closest_color(rgb_tuple)


def dictionary_based_split(text):
    def word_match(text, word_list):
        matches = []
        while text:
            for i in range(len(text), 0, -1):
                if text[:i] in word_list:
                    matches.append(text[:i])
                    text = text[i:]
                    break
        return matches
    return ' '.join(word_match(text.lower(), word_list))


def describe_object(x1, x2, y1, y2, frame):
    """
    use the confidence score to threshold when the description of the object is provided
    """
    roi = frame[y1:y2, x1:x2]
    hist = cv2.calcHist([roi], [0, 1, 2], None, [
                        256, 256, 256], [0, 256, 0, 256, 0, 256])
    max_bin = np.unravel_index(hist.argmax(), hist.shape)
    dominant_color = (int(max_bin[0]), int(max_bin[1]), int(max_bin[2]))
    color_name = get_color_name(dominant_color)

    return color_name


def JSON_response_to_dictionary(response):
    API_Data = response.json()
    if DEBUG_JSON_REQUESTS:
        for key in API_Data:
            {print(key, ":", API_Data[key])}
    return API_Data


def misty_description_demo():
    """
    Demo for the ability of our system to have Misty announce
    descriptions of obstacles surrounding her
    """
    # retrieve the camera frame
    image = fetch_misty_camera_frame()
    breakpoint()
    frame_path = "images/misty_frame.jpg"
    frame = cv2.imread(frame_path)

    results = model(frame, stream=False)
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(
                x2), int(y2)

            confidence = math.ceil((box.conf[0] * 100)) / 100
            print("Confidence --->", confidence)

            # Class name
            cls = int(box.cls[0])
            print("Class name -->", classNames[cls])

            if confidence > 0.5:
                color_description = describe_object(x1, x2, y1, y2, frame)
                text = f"I see a {classNames[cls]} in front of me. The most dominant color of the object is {color_description}."
                breakpoint()
                language = "en"
                speech = gTTS(text=text, lang=language, slow=False)
                file_name = "audio_demo.mp3"
                speech.save(file_name)
                ENCODING = 'utf-8'
                encode_string = base64.b64encode(open(file_name, "rb").read())
                base64_string = encode_string.decode(ENCODING)

                save_audio_response = misty.SaveAudio(
                    file_name, data=base64_string, overwriteExisting=True, immediatelyApply=True)
                save_audio = JSON_response_to_dictionary(save_audio_response)
                print("Saving Audio Response: " + str(save_audio))
                misty.PlayAudio(file_name, volume=10)
                time.sleep(5)


def main():
    # ipAddress = "10.5.11.234"
    # misty = Robot(ipAddress)
    # fetch_misty_camera_frame()
    # fetch_and_feed_into_yolo()
    frame_path = "/Users/nfalicov/Documents/tufts/probabilistic robotics/CityMap.png"
    frame = cv2.imread(frame_path)
    print(describe_object(100, 150, 100, 150, frame))
    cv2.rectangle(
        frame,
        (100, 100),
        (150, 150),
        (0, 0, 0),  # Black background
        -1,  # Filled rectangle
    )
    cv2.imshow('YOLO Detection Result', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# if __name__ == "__main__":
#     main()
