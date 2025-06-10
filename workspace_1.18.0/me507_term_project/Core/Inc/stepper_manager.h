/**
 * @file stepper_manager.h
 * @brief High-level CoreXY stepper motor management interface
 * @details Provides coordinated control of dual stepper motors in CoreXY configuration
 *          with features including automatic calibration, soft limits, position tracking,
 *          and intelligent motion planning with interruption capabilities.
 * @author Kyle Schumacher
 * @date Jun 9, 2025
 */

#ifndef INC_STEPPER_MANAGER_H_
#define INC_STEPPER_MANAGER_H_

#include "stepper_driver.h"
#include "stm32f4xx_hal.h"
#include <stdbool.h>
#include <stdint.h>

/**
 * @brief Stepper manager state machine states
 * @details Defines the operational states of the stepper manager for proper
 *          sequencing of calibration, movement, and idle operations.
 */
typedef enum {
    MANAGER_IDLE,        ///< Ready for new commands, no active operations
    MANAGER_CALIBRATING, ///< Performing automatic homing calibration sequence
    MANAGER_MOVING       ///< Executing coordinated motor movements
} ManagerState;

/**
 * @brief CoreXY stepper motor management structure
 * @details Manages dual stepper motors in CoreXY configuration with automatic
 *          calibration, soft limits, position tracking, and coordinated movement.
 *          Provides millimeter-based positioning with automatic step conversion.
 * 
 * @note CoreXY Kinematics:
 *       - Motor A (motor_x): Controls X+Y movement (A = X + Y)
 *       - Motor B (motor_y): Controls X-Y movement (B = X - Y)
 *       - Forward: X = (A + B)/2, Y = (A - B)/2
 *       - Inverse: A = X + Y, B = X - Y
 */
typedef struct {
    /* Motor references */
    StepperMotor *motor_x;  ///< Motor A in CoreXY system (handles X+Y movement)
    StepperMotor *motor_y;  ///< Motor B in CoreXY system (handles X-Y movement)

    /* Limit switch configuration */
    GPIO_TypeDef *limit_x_port;  ///< GPIO port for X-axis limit switch
    uint16_t      limit_x_pin;   ///< GPIO pin for X-axis limit switch
    GPIO_TypeDef *limit_y_port;  ///< GPIO port for Y-axis limit switch
    uint16_t      limit_y_pin;   ///< GPIO pin for Y-axis limit switch

    /* Position tracking (in millimeters) */
    float position_x_mm;    ///< Current commanded X position in millimeters
    float position_y_mm;    ///< Current commanded Y position in millimeters

    /* Soft limit boundaries (in millimeters) */
    float soft_limit_x_min_mm;  ///< Minimum allowed X position
    float soft_limit_x_max_mm;  ///< Maximum allowed X position
    float soft_limit_y_min_mm;  ///< Minimum allowed Y position
    float soft_limit_y_max_mm;  ///< Maximum allowed Y position

    /* Conversion and calibration */
    float steps_per_mm;         ///< Conversion factor: steps = mm × steps_per_mm
    float calib_backoff_mm;     ///< Distance to back off after hitting limit switch

    /* State management */
    ManagerState state;         ///< Current operational state

    /* Calibration status */
    bool calib_x_homed;         ///< True when X-axis homing is complete
    bool calib_y_homed;         ///< True when Y-axis homing is complete

    /* Default motion parameters */
    float default_speed_a;      ///< Default speed for motor A (steps/second)
    float default_speed_b;      ///< Default speed for motor B (steps/second)
    float default_accel_a;      ///< Default acceleration for motor A (steps/second²)
    float default_accel_b;      ///< Default acceleration for motor B (steps/second²)
} StepperManager;

/**
 * @brief Initialize the CoreXY stepper motor manager
 * @param mgr Pointer to StepperManager structure
 * @param motor_x Pointer to motor A (already initialized with StepperMotor_Init)
 * @param motor_y Pointer to motor B (already initialized with StepperMotor_Init)
 * @param limit_x_port GPIO port for X-axis limit switch
 * @param limit_x_pin GPIO pin for X-axis limit switch (active LOW)
 * @param limit_y_port GPIO port for Y-axis limit switch
 * @param limit_y_pin GPIO pin for Y-axis limit switch (active LOW)
 * @param steps_per_mm Conversion factor from millimeters to motor steps
 * @param soft_limit_x_min_mm Minimum allowed X position in millimeters
 * @param soft_limit_x_max_mm Maximum allowed X position in millimeters
 * @param soft_limit_y_min_mm Minimum allowed Y position in millimeters
 * @param soft_limit_y_max_mm Maximum allowed Y position in millimeters
 * @param calib_backoff_mm Distance to back away from limit switches after contact
 * @details Configures the manager with motor references, limit switches, coordinate
 *          system parameters, and soft limits. Motors must be pre-initialized.
 * @note Limit switches are expected to be active LOW (pressed = GPIO_PIN_RESET)
 * @note Soft limits are enforced during all normal movements
 * @warning Motors must be initialized with StepperMotor_Init() before calling this
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
 * @brief Start automatic homing calibration sequence
 * @param mgr Pointer to StepperManager structure
 * @details Initiates a two-phase automatic calibration sequence:
 *          1. X-axis homing: Move toward X limit switch, back off, zero position
 *          2. Y-axis homing: Move toward Y limit switch, back off, zero position
 *          
 *          The sequence uses slow, controlled speeds for accuracy and sets the
 *          final position to the configured limit boundaries.
 * @note Call StepperManager_Update() continuously to advance the calibration
 * @note Movement commands are ignored during calibration
 * @note Motors operate at reduced speed during calibration for safety
 * @warning Ensure limit switches are properly configured and functional
 */
void StepperManager_StartCalibration(StepperManager *mgr);

/**
 * @brief Request coordinated move to specified position
 * @param mgr Pointer to StepperManager structure
 * @param x_mm Target X position in millimeters
 * @param y_mm Target Y position in millimeters
 * @details Executes intelligent coordinated movement with advanced features:
 *          - Automatic soft limit clamping
 *          - Motion interruption for responsive control
 *          - Real-time position tracking during interruption
 *          - Distance-based speed optimization
 *          - CoreXY kinematic transformation
 *          - Minimum movement filtering (1mm threshold)
 * @note Movements less than 1mm total distance are automatically ignored
 * @note New commands interrupt current movements seamlessly
 * @note Calibration commands take priority over movement requests
 * @note Position coordinates are absolute, not relative
 */
void StepperManager_MoveTo(StepperManager *mgr, float x_mm, float y_mm);

/**
 * @brief Update manager state machine (call from main loop)
 * @param mgr Pointer to StepperManager structure
 * @details Core state machine update function that must be called continuously
 *          from the main application loop. Handles:
 *          
 *          **CALIBRATING State:**
 *          - Monitors limit switch activation during homing
 *          - Controls backoff movements after limit contact
 *          - Sets final calibrated position coordinates
 *          - Transitions to IDLE when complete
 *          
 *          **MOVING State:**
 *          - Monitors for unexpected limit switch activation
 *          - Detects movement completion
 *          - Provides emergency stop on limit contact
 *          - Transitions to IDLE when both motors stop
 *          
 *          **IDLE State:**
 *          - Ready to accept new commands
 *          - No active processing required
 * @note This function is non-blocking and must be called frequently
 * @note Limit switch contact during normal moves triggers immediate stop
 * @warning Continuous calling is required for proper operation
 */
void StepperManager_Update(StepperManager *mgr);

#endif /* INC_STEPPER_MANAGER_H_ */
