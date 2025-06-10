/**
 * @file stepper_manager.c
 * @brief CoreXY stepper motor manager implementation
 * @details Provides high-level motion control for CoreXY kinematics, including
 *          calibration, soft limits, position tracking, and coordinated moves.
 * @author Kyle Schumacher
 * @date Jun 9, 2025
 */

#include "stepper_manager.h"
#include <math.h>

/**
 * @brief Initialize the stepper manager with motor configuration and limits
 * @param mgr Pointer to the StepperManager structure
 * @param motor_x Pointer to motor A (X+Y axis motor in CoreXY)
 * @param motor_y Pointer to motor B (X-Y axis motor in CoreXY)
 * @param limit_x_port GPIO port for X-axis limit switch
 * @param limit_x_pin GPIO pin for X-axis limit switch
 * @param limit_y_port GPIO port for Y-axis limit switch
 * @param limit_y_pin GPIO pin for Y-axis limit switch
 * @param steps_per_mm Motor steps per millimeter conversion factor
 * @param soft_limit_x_min_mm Minimum X position in mm (soft limit)
 * @param soft_limit_x_max_mm Maximum X position in mm (soft limit)
 * @param soft_limit_y_min_mm Minimum Y position in mm (soft limit)
 * @param soft_limit_y_max_mm Maximum Y position in mm (soft limit)
 * @param calib_backoff_mm Distance to back off from limit switches after homing
 * @note All position limits are in millimeters relative to home position
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
) {
    mgr->motor_x          = motor_x;
    mgr->motor_y          = motor_y;
    mgr->limit_x_port     = limit_x_port;
    mgr->limit_x_pin      = limit_x_pin;
    mgr->limit_y_port     = limit_y_port;
    mgr->limit_y_pin      = limit_y_pin;
    mgr->steps_per_mm     = steps_per_mm;
    mgr->soft_limit_x_min_mm = soft_limit_x_min_mm;
    mgr->soft_limit_x_max_mm = soft_limit_x_max_mm;
    mgr->soft_limit_y_min_mm = soft_limit_y_min_mm;
    mgr->soft_limit_y_max_mm = soft_limit_y_max_mm;
    mgr->calib_backoff_mm = calib_backoff_mm;

    mgr->position_x_mm = 0.0f;
    mgr->position_y_mm = 0.0f;
    mgr->calib_x_homed = false;
    mgr->calib_y_homed = false;
    mgr->state         = MANAGER_IDLE;

    mgr->default_speed_a = motor_x->max_speed;
    mgr->default_accel_a = motor_x->accel;
    mgr->default_speed_b = motor_y->max_speed;
    mgr->default_accel_b = motor_y->accel;
}

/**
 * @brief Start the homing calibration sequence
 * @param mgr Pointer to the StepperManager structure
 * @details Initiates a two-phase homing sequence:
 *          1. Move towards X limit switch, back off, and zero X position
 *          2. Move towards Y limit switch, back off, and zero Y position
 *          The sequence runs in slow calibration speed for accuracy.
 * @note Call StepperManager_Update() repeatedly to progress the calibration
 * @warning Do not send move commands during calibration
 */
void StepperManager_StartCalibration(StepperManager *mgr)
{
    // 1) Clear any previous flags
    mgr->calib_x_homed = false;
    mgr->calib_y_homed = false;

    // 2) Pick a slow "calibration" speed/accel
    float calib_speed = 100.0f;   // steps/sec
    float calib_accel = 200.0f;   // steps/sec²

    // Override the low‐level motors' speed/accel
    mgr->motor_x->max_speed = calib_speed;
    mgr->motor_x->accel     = calib_accel;
    mgr->motor_y->max_speed = calib_speed;
    mgr->motor_y->accel     = calib_accel;

    // 3) Home X (positive‐X direction on CoreXY).
    //
    int32_t big_pos = 1000000000;
    StepperMotor_MoveTo(mgr->motor_x, mgr->motor_x->current_position + big_pos);
    StepperMotor_MoveTo(mgr->motor_y, mgr->motor_y->current_position + big_pos);

    mgr->state = MANAGER_CALIBRATING;
}

/**
 * @brief Request a coordinated move to the specified position
 * @param mgr Pointer to the StepperManager structure
 * @param new_x_mm Target X position in millimeters
 * @param new_y_mm Target Y position in millimeters
 * @details Performs intelligent motion planning with the following features:
 *          - Automatic soft limit enforcement
 *          - Motion interruption for new commands
 *          - Position tracking during interruption
 *          - Minimum movement threshold (1mm) filtering
 *          - Speed scaling based on move distance
 *          - CoreXY kinematic transformation
 * @note Movements less than 1mm total distance are ignored
 * @note Calibration commands take priority and will ignore move requests
 */
void StepperManager_MoveTo(StepperManager *mgr, float new_x_mm, float new_y_mm)
{
    // 1) If we're calibrating, ignore the new request
    if (mgr->state == MANAGER_CALIBRATING) {
        return;
    }

    // 2) Clamp the desired (x,y) to the soft limits:
    if (new_x_mm < mgr->soft_limit_x_min_mm) new_x_mm = mgr->soft_limit_x_min_mm;
    if (new_x_mm > mgr->soft_limit_x_max_mm) new_x_mm = mgr->soft_limit_x_max_mm;
    if (new_y_mm < mgr->soft_limit_y_min_mm) new_y_mm = mgr->soft_limit_y_min_mm;
    if (new_y_mm > mgr->soft_limit_y_max_mm) new_y_mm = mgr->soft_limit_y_max_mm;

    // 3) If we're already moving, update our current position tracking
    if (mgr->state == MANAGER_MOVING) {
        // Calculate current CoreXY position based on motor steps
        // A = motor_x, B = motor_y
        int32_t currentA = mgr->motor_x->current_position;
        int32_t currentB = mgr->motor_y->current_position;
        
        // CoreXY inverse kinematics: X = (A+B)/2, Y = (A-B)/2
        float steps_x = (currentA + currentB) / 2.0f;
        float steps_y = (currentA - currentB) / 2.0f;
        
        // Update our position tracking to match actual motor positions
        mgr->position_x_mm = steps_x / mgr->steps_per_mm;
        mgr->position_y_mm = steps_y / mgr->steps_per_mm;
    }

    // Recompute delta from current position (which may have just been updated)
    float dx_mm = new_x_mm - mgr->position_x_mm;
    float dy_mm = new_y_mm - mgr->position_y_mm;

    // Skip small moves - prevents unnecessary motor recalculation
    // Increased threshold to 1.0mm to ignore minor movements
    float total_dist_mm = sqrtf(dx_mm*dx_mm + dy_mm*dy_mm);
    if (total_dist_mm < 1.0f) {
        // If we were moving and stopped for a trivial move, go back to IDLE
        if (mgr->state == MANAGER_MOVING) {
            mgr->state = MANAGER_IDLE;
        }
        return;
    }

    // 4) Convert each delta to integer steps
    int32_t steps_dx = (int32_t)lrintf(dx_mm * mgr->steps_per_mm);
    int32_t steps_dy = (int32_t)lrintf(dy_mm * mgr->steps_per_mm);

    // 5) CoreXY formulas for motor A & B
    int32_t deltaA = steps_dx + steps_dy;
    int32_t deltaB = steps_dx - steps_dy;

    // 6) Compute each motor's absolute target steps
    int32_t targetA = mgr->motor_x->current_position + deltaA; 
    int32_t targetB = mgr->motor_y->current_position + deltaB;  

    // 7) Kick off both motors 
    float distA = fabsf((float)deltaA);
    float distB = fabsf((float)deltaB);

    // Calculate the total distance - used for speed scaling
    total_dist_mm = sqrtf(dx_mm*dx_mm + dy_mm*dy_mm);
    
    // For short moves, use higher acceleration but lower max speed
    float speed_scale = 1.0f;
    float accel_scale = 1.0f;
    
    if (total_dist_mm < 10.0f) {
        // For very short moves, we want quick acceleration but limited top speed
        speed_scale = 0.7f;
        accel_scale = 1.5f;
    } else if (total_dist_mm > 50.0f) {
        // For long moves, allow higher top speed
        speed_scale = 1.2f;
        accel_scale = 1.0f;
    }

    // pick the limiting axis
    float vA, vB;
    if (distA >= distB) {
        // A is limiting: run A at its default max, scale B
        vA = mgr->default_speed_a * speed_scale;
        vB = (distB / distA) * mgr->default_speed_a * speed_scale;
        // clamp B to its own max if needed
        if (vB > mgr->default_speed_b * speed_scale) vB = mgr->default_speed_b * speed_scale;
    } else {
        // B is limiting
        vB = mgr->default_speed_b * speed_scale;
        vA = (distA / distB) * mgr->default_speed_b * speed_scale;
        if (vA > mgr->default_speed_a * speed_scale) vA = mgr->default_speed_a * speed_scale;
    }

    // scale accelerations proportionally
    float aA = (vA / (mgr->default_speed_a * speed_scale)) * mgr->default_accel_a * accel_scale;
    float aB = (vB / (mgr->default_speed_b * speed_scale)) * mgr->default_accel_b * accel_scale;

    // override the motors' settings just for this move
    mgr->motor_x->max_speed = vA;
    mgr->motor_x->accel     = aA;
    mgr->motor_y->max_speed = vB;
    mgr->motor_y->accel     = aB;

    // now kick off both moves
    StepperMotor_MoveTo(mgr->motor_x, targetA);
    StepperMotor_MoveTo(mgr->motor_y, targetB);

    // 8) Update the manager's "commanded" position so we know where the carriage will be when this finishes
    mgr->position_x_mm = new_x_mm;
    mgr->position_y_mm = new_y_mm;

    // 9) Switch state so Update() can watch for completion
    mgr->state = MANAGER_MOVING;
}

/**
 * @brief Update the stepper manager state machine
 * @param mgr Pointer to the StepperManager structure
 * @details This function must be called continuously from the main loop.
 *          Handles state transitions for:
 *          - CALIBRATING: Manages the two-phase homing sequence
 *          - MOVING: Monitors move completion and limit switch safety
 *          - IDLE: No action required
 * @note In CALIBRATING state:
 *       - Monitors limit switches for homing detection
 *       - Handles backoff moves after limit contact
 *       - Sets final position coordinates after calibration
 * @note In MOVING state:
 *       - Checks for unexpected limit switch activation
 *       - Transitions to IDLE when both motors complete their moves
 * @warning Limit switch activation during normal moves will immediately stop all motion
 */
void StepperManager_Update(StepperManager *mgr)
{
    if (mgr->state == MANAGER_CALIBRATING) {
        // ─── COREXY HOMING: X‐axis first ───────────────────────────────────
        if (!mgr->calib_x_homed) {
            // If X endstop is pressed (active low), stop both motors immediately:
            if (HAL_GPIO_ReadPin(mgr->limit_x_port, mgr->limit_x_pin) == GPIO_PIN_RESET) {
                StepperMotor_Stop(mgr->motor_x);
                StepperMotor_Stop(mgr->motor_y);

                mgr->calib_x_homed = true;

                // Back off in +X by calib_backoff_mm
                int32_t backoff_steps = (int32_t) lrintf(mgr->calib_backoff_mm * mgr->steps_per_mm);
                // ΔX = +backoff, ΔY = 0  ⇒  ΔA = +backoff, ΔB = +backoff
                int32_t targetA = mgr->motor_x->current_position - backoff_steps;
                int32_t targetB = mgr->motor_y->current_position - backoff_steps;
                StepperMotor_MoveTo(mgr->motor_x, targetA);
                StepperMotor_MoveTo(mgr->motor_y, targetB);
            }
        }
        else {
            // Once back‐off is finished, set both motor step counts = 0
            if (!StepperMotor_IsMoving(mgr->motor_x) &&
                !StepperMotor_IsMoving(mgr->motor_y))
            {
                mgr->motor_x->current_position = 0;
                mgr->motor_y->current_position = 0;
            }
        }

        // ─── COREXY HOMING: Y‐axis second ───────────────────────────────────
        if (mgr->calib_x_homed && !mgr->calib_y_homed) {
            // We only need to start that move once when we detect X is done. So:
            if (!StepperMotor_IsMoving(mgr->motor_x) &&
                !StepperMotor_IsMoving(mgr->motor_y))
            {
                int32_t big_neg = -1000000000;
                // For Y: ΔX=0, ΔY = –∞ ⇒ ΔA = –∞, ΔB = +∞
                StepperMotor_MoveTo(mgr->motor_x, mgr->motor_x->current_position + big_neg); // A backward
                StepperMotor_MoveTo(mgr->motor_y, mgr->motor_y->current_position - big_neg); // B forward
            }

            // Check if Y endstop is pressed:
            if (HAL_GPIO_ReadPin(mgr->limit_y_port, mgr->limit_y_pin) == GPIO_PIN_RESET) {
                StepperMotor_Stop(mgr->motor_x);
                StepperMotor_Stop(mgr->motor_y);
                mgr->calib_y_homed = true;

                // Back off in +Y (which, in CoreXY, is "A_forward & B_forward")
                int32_t backoff_steps = (int32_t) lrintf(mgr->calib_backoff_mm * mgr->steps_per_mm);
                int32_t targetA = mgr->motor_x->current_position + backoff_steps;
                int32_t targetB = mgr->motor_y->current_position - backoff_steps;
                StepperMotor_MoveTo(mgr->motor_x, targetA);
                StepperMotor_MoveTo(mgr->motor_y, targetB);
            }
        }
        else if (mgr->calib_y_homed) {
            // Once both back-off moves complete, zero out step counts, report (0,0)
            if (!StepperMotor_IsMoving(mgr->motor_x) &&
                !StepperMotor_IsMoving(mgr->motor_y))
            {
            	float x_mm = mgr->soft_limit_x_max_mm;
            	float y_mm = mgr->soft_limit_y_min_mm;
            	float x_steps = (int32_t)lrintf(x_mm * mgr->steps_per_mm);
            	float y_steps = (int32_t)lrintf(y_mm * mgr->steps_per_mm);
            	int32_t A = x_steps + y_steps;
            	int32_t B = x_steps - y_steps;


                mgr->motor_x->current_position = A;
                mgr->motor_y->current_position = B;
                mgr->position_x_mm = mgr->soft_limit_x_max_mm;
                mgr->position_y_mm = mgr->soft_limit_y_min_mm;
                mgr->state = MANAGER_IDLE;
            }
        }
    }
    else if (mgr->state == MANAGER_MOVING) {
        // ─── During normal moves, if any limit switch is unexpectedly hit, stop that axis ─────
        // For CoreXY, if X or Y endstop is tripped during a normal move, you still want to
        // immediately kill both motors—otherwise the carriage may try to plow into the endstop.
        if (HAL_GPIO_ReadPin(mgr->limit_x_port, mgr->limit_x_pin) == GPIO_PIN_RESET ||
            HAL_GPIO_ReadPin(mgr->limit_y_port, mgr->limit_y_pin) == GPIO_PIN_RESET)
        {
            StepperMotor_Stop(mgr->motor_x);
            StepperMotor_Stop(mgr->motor_y);
            mgr->state = MANAGER_IDLE;
            return;
        }

        // If both motors finished their planned trapezoidal runs → done
        if (!StepperMotor_IsMoving(mgr->motor_x) &&
            !StepperMotor_IsMoving(mgr->motor_y))
        {
            mgr->state = MANAGER_IDLE;
        }
    }
}


