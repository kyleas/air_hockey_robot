/*
 * stepper_driver.c
 *
 *  Created on: Jun 2, 2025
 *      Author: kyle
 */

#include "stepper_driver.h"
#include <math.h>

/**
 * @brief  Plan a motion at constant speed, set DIR pin, etc.
 * @param  motor: Pointer to StepperMotor object
 * @note   Called internally by MoveTo().
 */
static void StepperMotor_PlanMotion(StepperMotor *motor)
{
    int32_t delta = motor->target_position - motor->current_position;
    if (delta == 0) {
        motor->moving = false;
        return;
    }

    // Set direction
    motor->direction = (delta > 0) ? +1 : -1;
    
    // Drive DIR pin high or low
    if (motor->direction > 0) {
        HAL_GPIO_WritePin(motor->dir_port, motor->dir_pin, GPIO_PIN_SET);
    } else {
        HAL_GPIO_WritePin(motor->dir_port, motor->dir_pin, GPIO_PIN_RESET);
    }

    motor->total_steps = fabsf(delta);
    motor->step_count = 0;

    // Set to constant speed (max_speed)
    motor->current_speed = motor->max_speed;

    // Calculate fixed step period for constant speed
    uint32_t raw = (uint32_t)((float)TIMER_FREQUENCY_HZ / motor->current_speed);
    motor->step_period = (raw < 1) ? 1 : raw;
    
    // Initialize timer
    motor->step_timer = 0;
    motor->pulse_high = false;
    motor->moving = true;
}

/**
 * @see stepper_driver.h
 */
void StepperMotor_Init(
    StepperMotor *motor,
    GPIO_TypeDef *dir_port, uint16_t dir_pin,
    GPIO_TypeDef *step_port, uint16_t step_pin,
    float max_speed,
    float accel
) {
    motor->dir_port      = dir_port;
    motor->dir_pin       = dir_pin;
    motor->step_port     = step_port;
    motor->step_pin      = step_pin;
    motor->current_position = 0;
    motor->target_position  = 0;
    motor->moving         = false;
    motor->direction      = +1;
    motor->step_timer     = 0;
    motor->step_period    = 0;
    motor->total_steps    = 0;
    motor->step_count     = 0;
    motor->accel_steps    = 0;  // Not used with constant speed
    motor->decel_steps    = 0;  // Not used with constant speed
    motor->current_speed  = 0.0f;
    motor->max_speed      = max_speed;
    motor->accel          = accel;  // Not used with constant speed
    motor->pulse_high     = false;

    // Initialize GPIO state: DIR=LOW, STEP=LOW
    HAL_GPIO_WritePin(motor->dir_port, motor->dir_pin, GPIO_PIN_RESET);
    HAL_GPIO_WritePin(motor->step_port, motor->step_pin, GPIO_PIN_RESET);
    // (We assume the limit pin is already configured as input with pull-up externally.)
}

/**
 * @see stepper_driver.h
 */
void StepperMotor_MoveTo(StepperMotor *motor, int32_t target_position)
{
    motor->target_position = target_position;
    StepperMotor_PlanMotion(motor);
}

/**
 * @see stepper_driver.h
 */
void StepperMotor_Stop(StepperMotor *motor)
{
    motor->moving    = false;
    motor->pulse_high = false;
    // Ensure STEP pin is low
    HAL_GPIO_WritePin(motor->step_port, motor->step_pin, GPIO_PIN_RESET);
}

/**
 * @see stepper_driver.h
 */
bool StepperMotor_IsMoving(StepperMotor *motor)
{
    return motor->moving;
}

/**
 * @see stepper_driver.h
 */
void StepperMotor_Update(StepperMotor *motor)
{
    if (!motor->moving) {
        return;
    }

    // If we are currently holding the STEP pin HIGH from the last ISR, pull it LOW now.
    if (motor->pulse_high) {
        HAL_GPIO_WritePin(motor->step_port, motor->step_pin, GPIO_PIN_RESET);
        motor->pulse_high = false;
        return;
    }

    // Count down step_timer. When it reaches zero, generate the next pulse.
    if (motor->step_timer > 0) {
        motor->step_timer--;
    }
    if (motor->step_timer == 0) {
        // Reload step_timer with fixed period (constant speed)
        motor->step_timer = motor->step_period;

        // Issue a single step pulse: STEP→HIGH now; next ISR tick will drop it low
        HAL_GPIO_WritePin(motor->step_port, motor->step_pin, GPIO_PIN_SET);
        motor->pulse_high = true;

        // Update bookkeeping: advance position & step_count
        motor->current_position += motor->direction;
        motor->step_count++;

        // If we have reached the desired total, stop motion completely
        if (motor->step_count >= motor->total_steps) {
            StepperMotor_Stop(motor);
        }
    }
}


