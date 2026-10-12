# Vision-Guided Pick (UR5e + 2F-85 + D435i)

MuJoCo simulation for a vision-guided pick task on a UR5e arm with a
wrist-mounted RealSense D435i RGB-D camera and a Robotiq 2F-85 gripper.

## Setup

    pip install -r requirements.txt

## Run

    python main.py

## Usage

 - `E key` — Pause / Resume running pick and place sequence.
 - `P key` — Place a new object in a random location. Limit of 8 per scene.
 - `R key` — Resets scene. Places four object at random on table and begins pick and place.
 


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
- Robot will pick objects from table and place them in a chute.
- The robot will seach for objects that are partially hidden.
- The search routine will fail on full hidden objects.
- Move speeds and gripper strength are configurable.
- The table surface is filtered by Z height.
- User controls are responsive and reliable.
- Behaviour is deterministic and repeatable.