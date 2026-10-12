# Vision-Guided Pick (UR5e + 2F-85 + D435i)

MuJoCo simulation for a vision-guided pick task on a UR5e arm with a
wrist-mounted RealSense D435i RGB-D camera and a Robotiq 2F-85 gripper.

## Setup

    pip install -r requirements.txt

## Run

    python main.py

## Layout

- `main.py` — entry point, orchestrates the pipeline
- `src/arm.py` — IK solver code for arm steering
- `src/camera.py` — wrapper for RealSense D435i wrist camera, handles capture and deprojection 
- `src/config.py` — Global flags and variables
- `src/floor.py` — tested but currently unused. Filters and fits floor plane.
- `src/model_loader.py` — composes UR5e + gripper + wrist camera + floor
- `src/perception.py` — object perception functions
- `src/pick_place.py` — creates scene, sequences robot moves.
- `requirements.txt` — dependencies

## Status
- Composed model loader & main file boilerplate
- Mounted wrist camera with RGB-D capture