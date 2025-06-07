/*
 * stepper_manager.h
 *
 *  Created on: Jun 2, 2025
 *      Author: kyle
 */

#ifndef INC_STEPPER_MANAGER_H_
#define INC_STEPPER_MANAGER_H_

#include "stepper_driver.h"
#include "stm32f4xx_hal.h"
#include <stdbool.h>
#include <stdint.h>

/**
 * @brief  Manager states for the XY pair.
 */
typedef enum {
    MANAGER_IDLE,
    MANAGER_CALIBRATING,
    MANAGER_MOVING
} ManagerState;

/**
 * @brief  A simple XY‐stage manager.
 *         - Takes in two StepperMotor pointers (motor_x, motor_y).
 *         - Takes in exactly two limit switch pins: one for X, one for Y.
 *         - Exposes: StartCalibration(), MoveTo(x_mm, y_mm), and Update().
 *         - Conversion from mm ↔ steps uses a single `steps_per_mm`.
 *         - Enforces soft limits in mm.
 *         - Calibration: drives both axes toward their *negative* limit switches until each is pressed,
 *           then backs off by `calib_backoff_mm` and sets that axis’s current_steps = 0.
 *         - During a normal MoveTo, clamps targets to soft limits and moves both axes simultaneously
 *           with each axis’s own trapezoidal profile (computed by its own StepperMotor).
 *         - Non‐blocking: you must call StepperManager_Update() periodically in your main loop
 *           to finish homing and detect when both axes have stopped.
 */
typedef struct {
    StepperMotor *motor_x;
    StepperMotor *motor_y;

    GPIO_TypeDef *limit_x_port;
    uint16_t      limit_x_pin;
    GPIO_TypeDef *limit_y_port;
    uint16_t      limit_y_pin;

    float position_x_mm;   // “Commanded” X position in mm (only valid when state==IDLE)
    float position_y_mm;   // “Commanded” Y position in mm

    float soft_limit_x_min_mm;
    float soft_limit_x_max_mm;
    float soft_limit_y_min_mm;
    float soft_limit_y_max_mm;

    float steps_per_mm;    // Conversion factor: [steps] = steps_per_mm * [mm]

    ManagerState state;

    bool calib_x_homed;    // True once X‐axis limit was hit & backed off
    bool calib_y_homed;    // True once Y‐axis limit was hit & backed off

    float calib_backoff_mm; // After hitting switch, move forward by this many mm, then zero
} StepperManager;

/**
 * @brief  Initialize the XY manager.
 * @param  mgr                   Pointer to StepperManager
 * @param  motor_x, motor_y      Two StepperMotor pointers (already initialized with Init())
 * @param  limit_x_port, limit_x_pin:   GPIO for X‐axis “home” switch (pressed=LOW)
 * @param  limit_y_port, limit_y_pin:   GPIO for Y‐axis “home” switch (pressed=LOW)
 * @param  steps_per_mm          How many steps correspond to 1 mm
 * @param  soft_limit_x_min_mm   Minimum X (mm) that we allow (e.g. 0.0f)
 * @param  soft_limit_x_max_mm   Maximum X (mm) that we allow
 * @param  soft_limit_y_min_mm   Minimum Y (mm)
 * @param  soft_limit_y_max_mm   Maximum Y (mm)
 * @param  calib_backoff_mm      After hitting a limit switch, back off by this many mm, then set pos=0
 */
void StepperManager_Init(
    StepperManager *mgr,
    StepperMotor   *motor_x,
    StepperMotor   *motor_y,
    GPIO_TypeDef   *limit_x_port,
    uint16_t        limit_x_pin,
    GPIO_TypeDef   *limit_y_port,
    uint16_t        limit_y_pin,
    float           steps_per_mm,
    float           soft_limit_x_min_mm,
    float           soft_limit_x_max_mm,
    float           soft_limit_y_min_mm,
    float           soft_limit_y_max_mm,
    float           calib_backoff_mm
);

/**
 * @brief  Begin a homing‐calibration sequence.
 *         This drives both axes toward their respective limit switches (negative direction) at a slow, fixed speed,
 *         waits for each to be pressed, then backs off by `calib_backoff_mm` and sets that axis’s step‐count=0.
 * @note   While homing is in progress, calls to MoveTo() are ignored.
 */
void StepperManager_StartCalibration(StepperManager *mgr);

/**
 * @brief  Request a simultaneous move of X→x_mm, Y→y_mm (both in mm).
 *         Clamps them individually to their soft limits.
 *         Does nothing if a calibration or a previous move is still in progress.
 */
void StepperManager_MoveTo(StepperManager *mgr, float x_mm, float y_mm);

/**
 * @brief  Must be called periodically in the main loop (non‐blocking).
 *         - If state==CALIBRATING, polls limit switches, backs off, then closes out homing.
 *         - If state==MOVING, polls to see if both StepperMotor_IsMoving()==false to declare IDLE.
 *         - Also watches for “unexpected” limit hits during a normal move (and immediately stops that axis).
 */
void StepperManager_Update(StepperManager *mgr);


#endif /* INC_STEPPER_MANAGER_H_ */
