# Simon Says: "Avoid Obstacle Ahead!"
**Creating a Robotic System Capable of Object Description and Avoidance**


Welcome to the **Misty_Guide** repository! This project presents a Python-based system designed for autonomous robot navigation, obstacle avoidance, and interaction using the Misty II robot platform.


## Project Structure(core files)

. 
├── integration.py # Main integration of all components (PF and OA)
├── oa_and_announce.py # Obstacle avoidance and announcements 
├── particle_filter.py # Localization implementation 
└── requirements.txt # Project dependencies


## Core Features

- **Localization**: Uses particle filter-based localization
- **Obstacle Avoidance**: Real-time detection and avoidance of obstacles
- **Object Recognition**: YOLO-based object detection and announcement
- **Text-to-Speech**: Voice announcements using gTTS
- **Sensor Integration**: Processes IMU, encoders, and time-of-flight data

## Components

### Particle Filter (`particle_filter.py`)
- Implements robot localization using particle filter algorithm
- Processes ceiling images for position estimation
- Handles sensor fusion from IMU and encoder data

### Obstacle Avoidance (`oa_and_announce.py`)
- Manages time-of-flight sensor data
- Implements obstacle detection and avoidance maneuvers
- Integrates YOLO object detection
- Handles voice announcements


### Integration (`integration.py`)
- Main system integration
- Coordinates particle filter, obstacle avoidance, and movement
- Manages WebSocket connections for sensor data
- Handles multi-threading for concurrent operations


## Setup
1. clone git directory
```bash
git clone https://github.com/kelseyzeng0610/Misty_Guide.git
cd Misty_Guide
```


3. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure robot IP:
Update MISTY_IP in relevant files
Default: "10.5.9.252"
Set up directories:
-make sure ceiling images are present
-Ensure yolo11n.pt is present
## Usage

Run obstacle avoidance(you can run this anywhere!):
```bash
python oa_and_announce.py
``` 

Run Particle Filter(you can ONLY run this @JCC 474):
```bash
python particle_filter.py
```
Run Integration(only @JCC 474)
```bash
python integration.py

```


Acknowledgement:

This project is a final project for CS141-Probabilistic Robotics (Fall 2024), taught by Prof. Jivko Sinapov. 
