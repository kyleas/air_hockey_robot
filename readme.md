# Air Hockey Robot

This project implements a 2-axis CoreXY air hockey robot designed as part of the ME507 embedded systems course. The robot plays against a human by tracking the puck with computer vision and responding with predictive paddle movements.

The documentation pages are at https://kyleas.github.io/air_hockey_robot/

![Air Hockey Table](images/overall.jpg)

## Overview

The system is divided into two main components:

- **Sensing & Vision (Raspberry Pi)**  
  Uses OpenCV to detect the puck and paddle on the board using HSV filtering, contour detection, and predictive path planning.

- **Motion Control (STM32)**  
  A CoreXY-style robot arm driven by two stepper motors, controlled by an STM32 microcontroller using custom C firmware.

Key features include:
- Limit switch-based auto-homing
- UART command interface between Pi and STM32
- Predictive reflection-based path planning
- CoreXY motion control with synchronized velocity planning

## Repository Structure

```text
├── Core/
│   ├── Inc/
│   └── Src/
│       ├── main.c               # Top-level control loop (calibration + UART parsing)
│       ├── stepper_driver.c     # Low-level stepper motor control
│       └── stepper_manager.c    # CoreXY motion planning + FSM
├── Pi_Side/
│   └── airhockey.py             # OpenCV puck detection and trajectory prediction
├── docs/
│   └── mainpage.md              # Doxygen top-level documentation file
├── README.md
├── LICENSE
└── .gitignore
