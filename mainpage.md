@mainpage Air Hockey Robot Documentation

# Air Hockey Robot

As part of the ME507 class project, an air hockey robot was created. A human player is situated on one side of the board, and a 2-axis, CoreXY style robot is on the other side. The robot will automatically predict where the puck will go and move the handle to that location. The robot is not quite fast enough to play the game as intended, so the operator will wait until the robot is in location. The robot is also not quite fast enough to hit it back, so if the puck becomes stuck, the robot will automatically push the puck to the operators side. 

\image html overall.jpg

## Robot Overview

The robot is separated into two separate functions: movement and sensing. 

## Movement 

The movement portion is a typical CoreXY style robot where two steppers use differential control to move the handle to each location. The steppers are both stationary and use one belt for movement. The movement portion automatically zeros itself upon each restart. This involves using two limit switches to zero the X and Y axis. The movement is all controled by an STM32 with TMC2209 stepper drivers. 

\image html board.jpg

### PCB
The robot itself is controlled through a custom PCB with a couple of important features. The main processor is an STM32F411CEU6 using C/C++. There are header pins on the board to program the chip with an ST-LINK from a Nucelo board. The board also features two TMC2209 stepper driver chips. These chips were selected for their quiet operation. These chips are only using the DIR/STEP pins to define motion, but have the functionality for full control with UART. There are two limit switch inputs in the board as well to zero. The board can take in 8V-18V as it has a built in buck to give a constant 5V and 3.3V rail. The power input comes from 1x XT-30 input to give the functionality for either a power supply or LiPo battery. 

\image html pcb.jpg 

### STM32 Code 
The STM32 firmware is written in C using the STM32CubeIDE and follows a structured, modular style. It is splot across three main components: 

### \ref main.c "main.c" System Control and UART-Based Command loop
The \ref main.c "main.c" file initializes all hardware (GPIOs, UARTs, timer, etc.) and executes the core application loop. It supports two operational modes: 

1. Calibration Mode: Runs at startup and homes the X and Y axes using limit switches. The handle moves until contact in both axes, backs off a configurable distance, and then resets its internal coordates to zero. 
2. Positioning Mode: The system listens for movement commands from a Raspberry Pi over UART. Each command encodes a new target (x, y) position. The main loop parses the commands, calls the high-level motion planner, and manages status responses for debugging. 

The firmware employs a responsive, interrupt-driven UART scheme using HAL callbacks and circular buffers, enabling non-blocking serial communication with both a PC and the Pi simultaneously. 

### \ref stepper_manager.c "stepper_manager.c" CoreXY High-Level Motion planner
The \ref stepper_manager.c "stepper_manager.c" file implements coordinated motion planning and CoreXY kinematics. It manages: 

1. Calibration Sequencing: Executes a multi-step homing routing using two limit switches, including precise backoff and position zeroing.
2. Soft Limit Enforcement: Ensures motion commands respect physical bounds of the board. 
3. Path Planning: Converts global X/Y commands into CoreXY A/B motor commands with velocity synchronization. 
4. Speed Scaling: Dynamically adjusts speed and acceleration based on travel distance. 
5. Limit Switch Detection: Calls an emergency stop on all steppers when a limit switch is detected in the positioning mode to prevent hardware damage. 

All logic is handled in a simple finite-state machine (FSM) pattern (IDLE, MOVING, CALIBRATING) and is updated through a central StepperManager_Update() call in the main loop. 

### \ref stepper_driver.c "stepper_driver.c" Low-Level Stepper Motor Controller
This file provides the real-time control of individual stepper motors. Features include: 

\li Constant speed stepping with configurable max speed and acceleration. 
\li Direction and step pin toggling via GPIO. 
\li Position tracking and step count management. 
\li Non-blocking move execution, compatible with timer-driven HAL_TIM_PeriodElapsedCallback() ISR. 
\li Safe emergency stop behavior on limit switch triggers.

This driver is intentionally kept hardware-focused and timing-tight, isolating motion primitives from higher-level path logic. 


## Sensing

The sensing portion executed using a Raspberry Pi 4 with a camera attached. The Raspberry Pi uses OpenCv for image detection. The process for getting the final detection takes place through multiple steps in the \ref airhockey.py "airhockey file": 

\htmlonly
<h3>Image Processing Pipeline</h3>

<ol>

  <li>
    <strong>Frame Calibration:</strong>  
    The camera's field of view is larger than the board itself, so the first step is to define the boundaries of the air hockey board.  
    The mode <code>'calibrate_frame'</code> allows the user to select two points per edge (top, bottom, left, right), which form a quadrilateral bounding box.  
    This selection defines a transformation matrix that isolates the board area from the rest of the image for further processing.

    <img src="frame_calibration.png" style="max-width: 100%; height: auto; margin-top: 1em;">
    <div style="font-style: italic; font-size: 90%; color: #ccc;">User selects 2 points per board edge to define a perspective transform for cropping.</div>
  </li>

  <li>
    <strong>HSV Calibration:</strong>  
    Once the board region is defined, the next task is to identify the puck and handle in the image using color detection.  
    The system uses HSV (Hue, Saturation, Value) filtering to isolate specific color ranges.  
    In <code>'calibrate_hsv'</code> mode, the user clicks on multiple image points corresponding to the desired object colors.  
    As points are added, the HSV range is dynamically expanded to include all selected values, and a buffer of ±10 is applied to each HSV channel.

    <img src="initial_hsv.png" style="max-width: 100%; height: auto; margin-top: 1em;">
    <div style="font-style: italic; font-size: 90%; color: #ccc;">Before HSV values are selected — no filtering is applied.</div>

    <img src="final_hsv.png" style="max-width: 100%; height: auto; margin-top: 1em;">
    <div style="font-style: italic; font-size: 90%; color: #ccc;">After selecting HSV points — puck and handle are now isolated clearly.</div>
  </li>

  <li>
    <strong>Contour Filtering:</strong>  
    After HSV filtering, the system uses OpenCV's contour detection to locate connected blobs in the binary mask.  
    Only contours with an area above a specified threshold are retained, to eliminate noise and small objects.  
    The centroid (center of mass) of each valid contour is calculated and displayed with a point marker.

  </li>

  <li>
    <strong>Vector Prediction:</strong>  
    Once the puck and handle are located, a vector is drawn between them, representing the expected path of the puck if struck by the handle.  
    The system also checks if the vector would intersect a wall and, if so, calculates a reflection vector assuming a perfect bounce.  
    These vectors are drawn in real time for visualization.

    <img src="2025-06-10-121049_1920x1080_scrot.png" style="max-width: 100%; height: auto; margin-top: 1em;">
    <div style="font-style: italic; font-size: 90%; color: #ccc;">Puck trajectory and reflected path are visualized using vector lines.</div>
  </li>

  <li>
    <strong>Velocity Thresholding:</strong>  
    If the puck’s velocity exceeds a set threshold, it is assumed to be in motion.  
    The trajectory vector of the puck is then projected forward to anticipate where it will go.  
    The robot uses this predictive path to move its paddle into position before the puck arrives.

  </li>

</ol>
\endhtmlonly