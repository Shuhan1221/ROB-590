Soft Robot Pipe-Climbing and Kresling Origami Simulation

Overview

This repository contains simulation and analysis scripts for soft-robot pipe climbing, volume-rendered tube trajectories, and orientation correction using an Extended Kalman Filter (EKF). It also includes tools to evaluate the constant-curvature assumption (PCC) for Kresling origami modules and a MATLAB utility to convert roll‑pitch‑yaw angles into curvature parameters.

Features

Unconstrained Soft Robot Trajectory: Simulate a soft robot in free space without any external constraints.

Volume-rendered Tube Trajectory: Visualize the robot as a tube with real volume.

Pipe-constrained Trajectory: Simulate the robot moving inside a rigid pipe.

EKF Correction: Apply an Extended Kalman Filter to correct trajectory and orientation errors.

PCC Validation Notebook: Jupyter notebook to test the Piecewise Constant Curvature (PCC) assumption on Kresling origami.

RPY-to-Angles Conversion: MATLAB function to compute bending (theta), directional (phi), and end-effector translation (T) from roll‑pitch‑yaw.

Prerequisites

Python 3.6+

MATLAB (or GNU Octave) for running the .m script

Jupyter Notebook for the PCC validation

Python Dependencies

Install via pip:

pip install numpy matplotlib jupyter

Optionally, you can create and activate a virtual environment:

python -m venv venv
source venv/bin/activate    # macOS/Linux
venv\Scripts\activate     # Windows
pip install -r requirements.txt

(Create a requirements.txt with numpy and matplotlib if desired.)

Usage

1. Unconstrained Soft Robot Trajectory

Run the simulation without any environmental constraints:

python newnewpiperobot.py

2. Volume-Rendered Tube Trajectory

Visualize the robot with real volume as a cyan tube:

python oldpiperobot.py

3. Pipe-Constrained Trajectory

Simulate the soft robot climbing inside a pipe with boundary constraints:

python piperobotinpipe.py

4. EKF-Based Correction

Apply an Extended Kalman Filter to the raw trajectory for state and orientation correction:

python ekf_new.py

5. PCC Validation Notebook

Launch the Jupyter notebook to verify the PCC assumption on three Kresling origami modules:

jupyter notebook test_pcc.ipynb

6. Roll-Pitch-Yaw to Curvature Angles (MATLAB)

Call the conversion function in MATLAB or Octave:

rpy = [roll; pitch; yaw];
l = segment_length;
[theta, phi, T] = from_rpy_to_angles(rpy, l);

File Descriptions

newnewpiperobot.py  : Generate an unconstrained soft robot trajectory.

oldpiperobot.py     : Render the soft robot as a cyan tube representing its real volume.

piperobotinpipe.py  : Simulate the soft robot trajectory under pipe constraints.

ekf_new.py          : Apply EKF for trajectory and orientation correction.

test_pcc.ipynb      : Jupyter notebook to test PCC on Kresling origami modules.

from_rpy_to_angles.m: MATLAB script to compute bending and directional angles from RPY.
