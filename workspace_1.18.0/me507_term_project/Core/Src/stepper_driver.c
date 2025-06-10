/**
 * @file stepper_driver.c
 * @brief Low-level stepper motor driver implementation
 * @details Provides direct control of individual stepper motors with constant speed
 *          operation, including step pulse generation, direction control, and 
 *          position tracking. Designed for timer interrupt-driven operation.
 * @author Kyle Schumacher
 * @date Jun 9, 2025
 */

#include "stepper_driver.h"
#include <math.h>

/**
 * @brief Plan a motion at constant speed and configure motor direction
 * @param motor Pointer to the StepperMotor structure
 * @details Internal function that calculates motion parameters for a move:
 *          - Sets direction pin based on target vs current position
 *          - Calculates total steps required for the move
 *          - Sets constant speed operation (no acceleration/deceleration)
 *          - Calculates fixed step timing period
 *          - Initializes motion state variables
 * @note This is a static function called internally by StepperMotor_MoveTo()
 * @note Motor runs at max_speed parameter throughout the entire move
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
 * @brief Initialize a stepper motor instance
 * @param motor Pointer to the StepperMotor structure to initialize
 * @param dir_port GPIO port for direction pin
 * @param dir_pin GPIO pin number for direction control
 * @param step_port GPIO port for step pin
 * @param step_pin GPIO pin number for step pulse generation
 * @param max_speed Maximum speed in steps per second
 * @param accel Acceleration parameter (not used in constant speed mode)
 * @details Configures the motor structure with hardware connections and motion
 *          parameters. Initializes all state variables to safe defaults and
 *          sets GPIO pins to their initial states (DIR=LOW, STEP=LOW).
 * @note The accel parameter is retained for compatibility but not used
 * @note GPIO pins must be configured as outputs before calling this function
 * @warning Ensure GPIO pins are properly configured in your hardware setup
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
 * @brief Start a move to the specified absolute position
 * @param motor Pointer to the StepperMotor structure
 * @param target_position Target position in steps (absolute coordinate)
 * @details Initiates a non-blocking move to the target position at constant speed.
 *          The motor will immediately start moving towards the target using the
 *          configured max_speed. Position tracking is automatically maintained.
 * @note This function returns immediately; use StepperMotor_IsMoving() to check status
 * @note Multiple calls will override the previous target (no queuing)
 * @note Position is tracked in steps relative to the initialization point
 */
void StepperMotor_MoveTo(StepperMotor *motor, int32_t target_position)
{
    motor->target_position = target_position;
    StepperMotor_PlanMotion(motor);
}

/**
 * @brief Immediately stop the motor and clear motion state
 * @param motor Pointer to the StepperMotor structure
 * @details Stops all motor motion immediately without deceleration. Clears the
 *          moving flag and ensures the STEP pin is in a safe LOW state. Current
 *          position tracking is preserved.
 * @note This is an emergency stop - no deceleration is performed
 * @note Position tracking remains accurate up to the point of stopping
 * @warning Motor will stop abruptly which may cause mechanical stress
 */
void StepperMotor_Stop(StepperMotor *motor)
{
    motor->moving    = false;
    motor->pulse_high = false;
    // Ensure STEP pin is low
    HAL_GPIO_WritePin(motor->step_port, motor->step_pin, GPIO_PIN_RESET);
}

/**
 * @brief Check if the motor is currently moving
 * @param motor Pointer to the StepperMotor structure
 * @return true if motor is moving, false if stopped or idle
 * @details Returns the current motion status. Useful for coordination between
 *          multiple motors or determining when a move has completed.
 * @note This function is fast and safe to call frequently
 */
bool StepperMotor_IsMoving(StepperMotor *motor)
{
    return motor->moving;
}

/**
 * @brief Update motor state and generate step pulses (call from timer ISR)
 * @param motor Pointer to the StepperMotor structure
 * @details This function must be called periodically from a timer interrupt at
 *          TIMER_FREQUENCY_HZ rate. Handles the step pulse generation state machine:
 *          - Manages step pulse timing based on configured speed
 *          - Generates step pulses with proper HIGH/LOW timing
 *          - Updates position tracking after each step
 *          - Stops motion when target position is reached
 * @note Call this function from your timer interrupt service routine
 * @note Timer frequency must match TIMER_FREQUENCY_HZ definition
 * @note Uses a two-phase approach: HIGH pulse, then LOW pulse on next call
 * @warning This function assumes it's called at the correct timer frequency
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


