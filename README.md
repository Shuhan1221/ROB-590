# Soft Robot Pipe-Climbing and Kresling Origami Simulation

## Overview
This repository contains simulation and analysis scripts for soft-robot pipe climbing, volume-rendered tube trajectories, and orientation correction using an Extended Kalman Filter (EKF). It also includes tools to evaluate the constant-curvature assumption (PCC) for Kresling origami modules and a MATLAB utility to convert roll-pitch-yaw angles into curvature parameters.

## Features
- **Unconstrained Soft Robot Trajectory**: Simulate a soft robot in free space without any external constraints.
- **Volume-rendered Tube Trajectory**: Visualize the robot as a tube with real volume.
- **Pipe-constrained Trajectory**: Simulate the robot moving inside a rigid pipe.
- **EKF Correction**: Apply an Extended Kalman Filter to correct trajectory and orientation errors.
- **PCC Validation Notebook**: Jupyter notebook to test the Piecewise Constant Curvature (PCC) assumption on Kresling origami.
- **RPY-to-Angles Conversion**: MATLAB function to compute bending (`theta`), directional (`phi`), and end-effector translation (`T`) from roll-pitch-yaw.

## Prerequisites
- **Python 3.6+**
- **MATLAB** (or GNU Octave) for running the `.m` script
- **Jupyter Notebook** for the PCC validation

### Python Dependencies
Install via pip:
```bash
pip install numpy matplotlib jupyter
