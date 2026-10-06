# Vision-Guided Pick (UR5e + 2F-85 + D435i)

MuJoCo simulation for a vision-guided pick task on a UR5e arm with a
wrist-mounted RealSense D435i RGB-D camera and a Robotiq 2F-85 gripper.

## Setup

    pip install -r requirements.txt

## Run

    python main.py

## Layout

- `main.py` — entry point, orchestrates the pipeline
- `src/model_loader.py` — composes UR5e + gripper + camera + floor
- `src/camera.py` — RGB-D capture and deprojection
- `requirements.txt` — dependencies

## Status
- Composed model loader & main file boilerplate
- Mounted wrist camera with RGB-D capture