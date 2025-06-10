/**
 * @file stepper_driver.h
 * @brief Low-level stepper motor driver interface
 * @details Provides direct control interface for individual stepper motors with
 *          constant speed operation. Includes step pulse generation, direction
 *          control, position tracking, and timer interrupt integration.
 * @author Kyle Schumacher
 * @date Jun 9, 2025
 */

#ifndef INC_STEPPER_DRIVER_H_
#define INC_STEPPER_DRIVER_H_

#include "stm32f4xx_hal.h"
#include <stdbool.h>
#include <stdint.h>

/**
 * @brief Timer interrupt frequency in Hz
 * @details The application must configure a hardware timer (e.g. TIM2) to
 *          generate update interrupts at exactly this rate. In the ISR,
 *          call StepperMotor_Update() for each motor instance.
 * @note Higher frequencies allow faster step rates but increase CPU overhead
 * @note Must match your actual timer configuration
 */
#define TIMER_FREQUENCY_HZ  10000U

/**
 * @brief Stepper motor control structure
 * @details Contains all state and configuration data for a single stepper motor.
 *          Manages GPIO pins for direction and step control, maintains position
 *          tracking, and handles constant-speed motion profiles.
 * @note Initialize with StepperMotor_Init() before use
 * @note All position values are in steps relative to initialization point
 */
typedef struct {
    /* GPIO pin assignments */
    GPIO_TypeDef *dir_port;     ///< GPIO port for direction control pin
    uint16_t      dir_pin;      ///< GPIO pin number for direction control
    GPIO_TypeDef *step_port;    ///< GPIO port for step pulse pin
    uint16_t      step_pin;     ///< GPIO pin number for step pulse generation
    GPIO_TypeDef *limit_port;   ///< GPIO port for limit switch (not used in current implementation)
    uint16_t      limit_pin;    ///< GPIO pin number for limit switch (not used in current implementation)

    /* Position and motion state */
    int32_t current_position;   ///< Current position in steps (signed, relative to init)
    int32_t target_position;    ///< Target position in steps for current move
    bool    moving;             ///< True if motor is currently executing a move
    int8_t  direction;          ///< Movement direction: +1 (forward) or -1 (reverse)

    /* Timing and step generation */
    uint32_t step_timer;        ///< Countdown timer for next step pulse (in timer ticks)
    uint32_t step_period;       ///< Timer ticks between step pulses (determines speed)
    int32_t  total_steps;       ///< Total number of steps required for current move
    int32_t  step_count;        ///< Number of steps completed in current move
    int32_t  accel_steps;       ///< Number of acceleration steps (unused in constant speed mode)
    int32_t  decel_steps;       ///< Number of deceleration steps (unused in constant speed mode)

    /* Speed configuration */
    float current_speed;        ///< Current operating speed in steps per second
    float max_speed;            ///< Maximum configured speed in steps per second
    float accel;                ///< Acceleration value (unused in constant speed mode)

    /* Step pulse state */
    bool pulse_high;            ///< True when STEP pin is currently HIGH (pulse active)
} StepperMotor;

/**
 * @brief Initialize a stepper motor instance
 * @param motor Pointer to StepperMotor structure to initialize
 * @param dir_port GPIO port for direction control pin
 * @param dir_pin GPIO pin number for direction control
 * @param step_port GPIO port for step pulse generation pin
 * @param step_pin GPIO pin number for step pulse generation
 * @param max_speed Maximum operating speed in steps per second
 * @param accel Acceleration parameter (retained for compatibility, unused in constant speed mode)
 * @details Configures the motor structure with hardware pin assignments and
 *          motion parameters. Sets initial safe states for all variables and
 *          initializes GPIO pins to LOW state.
 * @note GPIO pins must be configured as outputs before calling this function
 * @note The accel parameter is retained for API compatibility but not used
 * @warning Ensure proper GPIO configuration in your hardware initialization
 */
void StepperMotor_Init(
    StepperMotor *motor,
    GPIO_TypeDef *dir_port, uint16_t dir_pin,
    GPIO_TypeDef *step_port, uint16_t step_pin,
    float max_speed,
    float accel
);

/**
 * @brief Start a non-blocking move to absolute position
 * @param motor Pointer to StepperMotor structure
 * @param target_position Target position in steps (absolute coordinate)
 * @details Initiates constant-speed motion to the specified absolute position.
 *          Movement starts immediately and runs at the configured max_speed.
 *          Function returns immediately; use StepperMotor_IsMoving() to monitor
 *          completion status.
 * @note Position is absolute, not relative to current position
 * @note Multiple calls will override previous target (no move queuing)
 * @note No acceleration/deceleration - motor runs at constant speed
 */
void StepperMotor_MoveTo(StepperMotor *motor, int32_t target_position);

/**
 * @brief Immediately stop motor motion
 * @param motor Pointer to StepperMotor structure
 * @details Performs emergency stop of motor motion without deceleration.
 *          Clears motion state and ensures STEP pin is in safe LOW state.
 *          Position tracking remains accurate up to the point of stopping.
 * @note This is an immediate stop - no deceleration ramp
 * @note Current position tracking is preserved
 * @warning Abrupt stopping may cause mechanical stress at high speeds
 */
void StepperMotor_Stop(StepperMotor *motor);

/**
 * @brief Check if motor is currently moving
 * @param motor Pointer to StepperMotor structure
 * @return true if motor is executing a move, false if stopped or idle
 * @details Returns current motion status for coordination with other motors
 *          or determining when moves have completed.
 * @note Fast and safe to call frequently from main loop
 * @note Useful for multi-motor synchronization
 */
bool StepperMotor_IsMoving(StepperMotor *motor);

/**
 * @brief Update motor state and generate step pulses (ISR function)
 * @param motor Pointer to StepperMotor structure
 * @details Core motor update function that must be called from timer ISR
 *          at TIMER_FREQUENCY_HZ rate. Handles step pulse timing, GPIO
 *          control, position tracking, and move completion detection.
 *          
 *          Step pulse generation uses two-phase approach:
 *          - Phase 1: Set STEP pin HIGH, start pulse
 *          - Phase 2: Set STEP pin LOW, complete pulse
 * @note MUST be called from timer interrupt at TIMER_FREQUENCY_HZ rate
 * @note Timer frequency must exactly match TIMER_FREQUENCY_HZ definition
 * @note Keep ISR processing minimal for consistent timing
 * @warning Incorrect timer frequency will cause speed errors
 */
void StepperMotor_Update(StepperMotor *motor);

#endif /* INC_STEPPER_DRIVER_H_ */
